from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from data_pipeline.csv_parser import parse_csv_files
from data_pipeline.helpers import get_file_paths_of_cached_files
from performance_helper import create_profiler
import s3
import settings

# FINRA data spec:
# https://www.finra.org/sites/default/files/finra-adds-system-user-guide.pdf

_delimiter: Literal[','] = ','
_quote_char: Literal['"'] = '"'
_column_types: dict[str, pa.DataType] = {
    'MESSAGE_SEQUENCE_NUMBER': pa.uint32(),
    'DATE_TIME': pa.timestamp('ns'),
    'BSYM': pa.string(),
    'CUSIP': pa.string(),
    'QUANTITY': pa.string(),
    'PRICE': pa.float64(),
    'SIDE': pa.string(),
    'EXECUTION_DATE_TIME': pa.timestamp('ns'),
    'YIELD': pa.float64(),
    'YIELD_DIRECTION': pa.string(),
    'CONTRA_PARTY_TYPE': pa.string(),
    'ATS_INDICATOR': pa.string(),
    'MESSAGE_CATEGORY': pa.string(),
    'MESSAGE_TYPE': pa.string(),
    'SUB_PRODUCT_TYPE': pa.string(),
    'AS_OF_INDICATOR': pa.string(),
    'WHEN_ISSUED_INDICATOR': pa.string()
}

def _create_process(trade_source: str):
    def _process(table: pa.table) -> pa.table:
        return pa.table({
                'as_of_indicator': table.column('AS_OF_INDICATOR'),
                'ats_indicator': table.column('ATS_INDICATOR').fill_null('N'),
                'buy_sell': table.column('SIDE'),
                'cusip': table.column('CUSIP'),
                'execution_date': pc.assume_timezone(table.column('EXECUTION_DATE_TIME'),
                                                    settings.get('$.finra_timezone')).cast(pa.int64()),
                'figi': table.column('BSYM'),
                'message_category': table.column('MESSAGE_CATEGORY'),
                'message_type': table.column('MESSAGE_TYPE'),
                'price': table.column('PRICE'),
                'quantity': pc.cast(
                                pc.if_else(
                                    pc.equal(table.column('QUANTITY'), '5MM+'),
                                    '5000000',
                                    pc.if_else(
                                        pc.equal(table.column('QUANTITY'), '1MM+'),
                                        '1000000',
                                        table.column('QUANTITY'))),
                                pa.float64()),
                'record_num': table.column('MESSAGE_SEQUENCE_NUMBER'),
                'report_date': pc.assume_timezone(table.column('DATE_TIME'),
                                                settings.get('$.finra_timezone')).cast(pa.int64()),
                'side': table.column('CONTRA_PARTY_TYPE'),
                'sub_product': table.column('SUB_PRODUCT_TYPE'),
                'trade_original_format': pa.nulls(len(table), pa.string()).fill_null('finra_eod'),
                'trade_status': pa.nulls(len(table), pa.string()), # finra historical needs this column
                'trade_source': pa.nulls(len(table), pa.string()).fill_null(trade_source),
                # change 'W' to 'Y' and fill nulls with 'N', if it's neither then leave it as-is
                'when_issued_indicator': pc.if_else(pc.equal(table.column('WHEN_ISSUED_INDICATOR'), 'W'),
                                            'Y',
                                            table.column('WHEN_ISSUED_INDICATOR')).fill_null('N'),
                'yield': pc.multiply(
                            pc.if_else(
                                pc.is_null(table.column('YIELD_DIRECTION')), 1, -1),
                            table.column('YIELD'))})
    return _process

def parse_finra_eod(data_path: str, trade_source: str) -> pa.table:
    with create_profiler('parse all finra eod')() as p:
        mapping = s3.get_all_with_prefix(data_path)
        file_paths = get_file_paths_of_cached_files(mapping)
        file_paths = [f for f in file_paths if f.lower().endswith('.csv')]
        result = parse_csv_files(file_paths, _column_types, delimiter=_delimiter, quote_char=_quote_char, process=_create_process(trade_source))
        p.print(f'rows: {result.num_rows}')
        return result

def parse_from_dict(data: dict[str, list], trade_source: str) -> pa.table:
    data = {k: pa.array(data[k], type=_column_types[k]) for k in _column_types.keys()}
    return _create_process(trade_source)(pa.Table.from_pydict(data))
