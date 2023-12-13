import base64
from datetime import datetime

from log import create_logger
from data_pipeline import parse_finra_eod

# https://www.finra.org/sites/default/files/BTDS-specs-v4.6A.pdf

logger = create_logger('finra')
_LOG_COLUMNS = ['BSYM', 'CUSIP', 'DATE_TIME', 'EXECUTION_DATE_TIME', 'SIDE', 'PRICE', 'YIELD', 'QUANTITY']
logger.debug(f'log columns for parsed messages: {_LOG_COLUMNS}')

# Tuples with column name, length in bytes

finra_dtypes = {
        'SHORT_INT': 7,
        'LONG': 14,
        'FLOAT': 11,
        'BIG_FLOAT': 13,
        'DATE': 8,
        'DATE_TIME': 14
}

field_sizes = {
        'MESSAGE_CATEGORY': 1,
        'MESSAGE_TYPE': 1,
        'RESERVED': 1,
        'RETRANSMISSION_REQUESTER': 2,
        'MESSAGE_SEQUENCE_NUMBER': 'SHORT_INT',
        'MARKET_CENTER_ORIGINATOR_ID': 1,
        'DATE_TIME': 'DATE_TIME',

        'SYMBOL': 14,
        'CUSIP': 9,
        'BSYM': 12,
        'SUB_PRODUCT_TYPE': 5,
        'ORIGINAL_DISSEMINATION_DATE': 'DATE',
        'ORIGINAL_MESSAGE_SEQUENCE_NUMBER': 'SHORT_INT',
        'FUNCTION': 1,
        'QUANTITY_INDICATOR': 1,
        'QUANTITY': 14,
        'PRICE': 'FLOAT',
        'REMUNERATION': 1,
        'SPECIAL_PRICE_INDICATOR': 1,
        'SIDE': 1,
        'AS_OF_INDICATOR': 1,
        'EXECUTION_DATE_TIME': 'DATE_TIME',
        'FUTURE_USE': 2,
        'SALE_CONDITION_3': 1,
        'SALE_CONDITION_4': 1,
        'SETTLEMENT_DATE': 'DATE',
        'YIELD_DIRECTION': 1,
        'YIELD': 'BIG_FLOAT',
        'WHEN_ISSUED_INDICATOR': 1,
        'REPORTING_PARTY_TYPE': 1,
        'CONTRA_PARTY_TYPE': 1,
        'ATS_INDICATOR': 1,
        'CHANGE_INDICATOR': 1,

        'HIGH_PRICE': 'FLOAT',
        'HIGH_YIELD_DIRECTION': 1,
        'HIGH_YIELD': 'BIG_FLOAT',
        'LOW_PRICE': 'FLOAT',
        'LOW_YIELD_DIRECTION': 1,
        'LOW_YIELD': 'BIG_FLOAT',
        'LAST_SALE_PRICE': 'FLOAT',
        'LAST_SALE_YIELD_DIRECTION': 1,
        'LAST_SALE_YIELD': 'BIG_FLOAT',
        'CHANGE_INDICATOR': 1,

        'DAILY_HIGH_PRICE': 'FLOAT',
        'DAILY_HIGH_YIELD': 'BIG_FLOAT',
        'DAILY_LOW_PRICE': 'FLOAT',
        'DAILY_LOW_YIELD': 'BIG_FLOAT',
        'DAILY_CLOSE_PRICE': 'FLOAT',
        'CLOSE_YIELD_DIRECTION': 1,
        'DAILY_CLOSE_YIELD': 'BIG_FLOAT',

        'ISSUER': 30,
        'ACTION': 1,
        'ACTION_DATE_TIME': 'DATE_TIME',
        'HALT_REASON': 4
}


header_fields = [
        'MESSAGE_CATEGORY',
        'MESSAGE_TYPE',
        'RESERVED',
        'RETRANSMISSION_REQUESTER',
        'MESSAGE_SEQUENCE_NUMBER',
        'MARKET_CENTER_ORIGINATOR_ID',
        'DATE_TIME'
]

field_groups = {
        'label': ['SYMBOL', 'CUSIP', 'BSYM', 'SUB_PRODUCT_TYPE'],
        'trade_additional_information': ['ORIGINAL_DISSEMINATION_DATE'],
        'additional_information': ['ORIGINAL_DISSEMINATION_DATE', 'ORIGINAL_MESSAGE_SEQUENCE_NUMBER', 'FUNCTION'],
        'trade_information': ['QUANTITY_INDICATOR', 'QUANTITY', 'PRICE', 'REMUNERATION', 'SPECIAL_PRICE_INDICATOR',
                                       'SIDE', 'AS_OF_INDICATOR', 'EXECUTION_DATE_TIME', 'FUTURE_USE', 'SALE_CONDITION_3',
                                       'SALE_CONDITION_4', 'SETTLEMENT_DATE', 'YIELD_DIRECTION', 'YIELD', 'WHEN_ISSUED_INDICATOR',
                                       'REPORTING_PARTY_TYPE', 'CONTRA_PARTY_TYPE', 'ATS_INDICATOR'],
        'trade_summary_information': ['CHANGE_INDICATOR'],
        'summary_information': ['HIGH_PRICE', 'HIGH_PRICE_DIRECTION', 'HIGH_YIELD', 'LOW_PRICE', 'LOW_YIELD_DIRECTION',
                                'LOW_YIELD', 'LAST_SALE_PRICE', 'LAST_SALE_YIELD_DIRECTION', 'LAST_SALE_YIELD', 'CHANGE_INDICATOR'],
        'daily_trade_summary': ['WHEN_ISSUED_INDICATOR',
                                'DAILY_HIGH_PRICE', 'HIGH_YIELD_DIRECTION', 'DAILY_HIGH_YIELD', 'DAILY_LOW_PRICE', 'LOW_YIELD_DIRECTION',
                                'DAILY_LOW_YIELD', 'DAILY_CLOSE_PRICE', 'CLOSE_YIELD_DIRECTION', 'DAILY_CLOSE_YIELD'],
        'trading_halt': ['ISSUER', 'ACTION', 'ACTION_DATE_TIME', 'HALT_REASON'],
        'totals': ['ALL_SECURITIES', 'INVESTMENT_GRADE', 'HIGH_YIELD', 'CONVERTIBLES'],
}

# We have message categories
# within which we have the message
# types.
message_categories = {
        'T': { # Trade
            'M': [ # Trade Report
                'label',
                'trade_additional_information',
                'trade_information',
                'trade_summary_information',
            ],
            'N': [ # Trade Cancelled
                'label',
                'additional_information',
                'trade_information',
                'summary_information'
            ],
            'O': [ # Trade Correction
                'label',
                'additional_information',
                ('trade_information', 'original_trade_information'),
                ('trade_information', 'correction_trade_information'),
                'summary_information'
            ]
        },
        'C': { # Control

        },
        'A': { # Administrative
            'E': [ # Daily Trade Summary
                'label',
                'daily_trade_summary'
            ],
            'H': [ # Trading Halt
                'label',
                'trading_halt'
            ],

            # A, "General Administrative Message" not implemented yet as of the v4.6A specification
            
            '1': [ # End of Day Market Aggregate Data
                ('totals', 'total_securities_traded'),
                ('totals', 'advances'),
                ('totals', 'declines'),
                ('totals', 'unchanged'),
                ('totals', '52_week_high'),
                ('totals', '52_week_low'),
                ('totals', 'total_volume')
            ]
        }
}


def get_indexed_value(data, i, length):    
    return data[i:(i+length)].decode('ascii')


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# Returns tuple
# pandas dtype name and value converted to that dtype
def parse_dtype(dtype, data, i):
    if isinstance(dtype, int):
        value = get_indexed_value(data, i, dtype).strip()
        if value == '':
            value = None
        return dtype, 'string', value
    dtype_length = finra_dtypes[dtype]
    value = get_indexed_value(data, i, dtype_length)
    value = value.strip()
    if value == '':
        return dtype_length, None, None
    if dtype == 'SHORT_INT' or dtype == 'LONG':
        return dtype_length, 'int', int(value)
    if dtype == 'FLOAT' or dtype == 'BIG_FLOAT':
        return dtype_length, 'float', float(value)
    if dtype == 'DATE':
        return dtype_length, 'datetime', value
    if dtype == 'DATE_TIME':
        return dtype_length, 'datetime', datetime.strptime(value, '%Y%m%d%H%M%S')
    raise NotImplementedError(f'Unrecognized dtype \"{dtype}\"')


def parse_fields(fields, data, i=0):
    parsed = {}
    for field in fields:
        dtype = field_sizes[field]
        l, _, value = parse_dtype(dtype, data, i)
        parsed[field] = value
        i += l
    return i, parsed


def parse_data(raw_data, header_only=False) -> dict:
    start_char_idx = raw_data.index(bytes.fromhex('01'))
    end_char_idx = raw_data.index(bytes.fromhex('03'))

    raw_data = raw_data[(start_char_idx + 1):end_char_idx]
    i, header = parse_fields(header_fields, raw_data)

    if header_only:
        return header

    message_category = header['MESSAGE_CATEGORY']
    message_type = header['MESSAGE_TYPE']
    if message_category != 'T' or message_type != 'M':
        # This is not a trade report, skip for now. We can take
        # care of the other message types later.
        return {'not_trade_report': 'Not trade report for the message above'}

    message = header

    if message_category not in message_categories or \
            message_type not in message_categories[message_category]:
        raise NotImplementedError(f'Unrecognized message category \'{message_category}\' or message type \'{message_type}\'')

    message_spec = message_categories[header['MESSAGE_CATEGORY']][header['MESSAGE_TYPE']]

    for message_type_field_group in message_spec:
        if message_type_field_group not in field_groups:
            raise NotImplementedError(f'Field group '
                                      f'{message_type_field_group} not found in the field_groups list')
        field_group = field_groups[message_type_field_group]
        i, field_group_data = parse_fields(field_group, raw_data, i)
        # Append the field group data to the message
        message = {**message, **field_group_data}

    return message


def parse_messages(base64_messages: list[str], trade_source: str):
    if not base64_messages:
        return None
    messages = []
    for m in base64_messages:
        logger.debug(m)
        try:
            raw_data = base64.b64decode(m)
            header = parse_data(raw_data, header_only=True)
            if header['MESSAGE_CATEGORY'] != 'T' or header['MESSAGE_TYPE'] != 'M':
                logger.debug(f'skipping')
                continue
            message = parse_data(raw_data)
            messages.append(message)
            logger.debug(','.join(str(message[k]) for k in _LOG_COLUMNS))
        except:
            logger.exception(f'error parsing {m}')

    if not messages or len(messages) <= 0:
        return None

    # Convert messages from list of dicts to dict of lists
    messages = {k: [m[k] for m in messages] for k in messages[0].keys()}

    return parse_finra_eod.parse_from_dict(messages, trade_source)
