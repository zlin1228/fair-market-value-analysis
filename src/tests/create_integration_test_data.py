import sys
sys.path.insert(0, '..')

import os
import re
import tempfile
from typing import Callable, cast, Literal
import xml.etree.ElementTree as ET
import zipfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

from data_pipeline.csv_parser import _invalid_row_handler as invalid_row_handler_default
from data_pipeline.parse_finra_eod import _invalid_row_handler as invalid_row_handler_eod, _delimiter as delimiter_eod
from data_pipeline.parse_finra_historical import _invalid_row_handler as invalid_row_handler_historical, _delimiter as delimiter_historical
from data_pipeline.parse_moodys import namespaces
import settings
import s3
from tests.helpers import BONDS, CleanEnvironment

cusips = pa.array([bond[0] for bond in BONDS], pa.string())

# In order to preserve the data exactly how it exists we have to create a schema to tell PyArrow to treat all data
# as strings so it doesn't automatically infer types.
# To create the schema we need a list of all the columns.
finra_historical_columns = ['Record Count Num', 'Reference Number', 'Trade Status', 'TRACE Symbol', 'CUSIP', 'Bloomberg Identifier',
                    'Sub Product', 'When Issued Indicator', 'Remuneration', 'Quantity', 'Price', 'Yield Direction', 'Yield', 'As Of Indicator',
                    'Execution Date', 'Execution Time', 'Trade Report Date', 'Trade Report Time', 'Settlement Date', 'Trade Modifier 3',
                    'Trade Modifier 4', 'Buy/Sell Indicator', 'Buyer Commission', 'Buyer Capacity', 'Seller Commission', 'Seller Capacity',
                    'Reporting Party Type', 'Contra Party Indicator', 'Locked In Indicator', 'ATS Indicator', 'Special Price Indicator',
                    'Trading Market Indicator', 'Dissemination Flag', 'Prior Trade Report Date', 'Prior Reference Number', 'First Trade Control Date',
                    'First Trade Control Number']
finra_eod_columns = ['MESSAGE_CATEGORY', 'MESSAGE_TYPE', 'MESSAGE_SEQUENCE_NUMBER', 'MARKET_CENTER', 'DATE_TIME', 'SYMBOL',
                    'CUSIP', 'BSYM', 'SUB_PRODUCT_TYPE', 'ORIGINAL_DISSEMINATION_DATE', 'QUANTITY_INDICATOR', 'QUANTITY', 'PRICE',
                    'REMUNERATION', 'SPECIAL_PRICE_INDICATOR', 'SIDE', 'AS_OF_INDICATOR', 'EXECUTION_DATE_TIME', 'SALE_CONDITION_3',
                    'SALE_CONDITION_4', 'SETTLEMENT_DATE', 'YIELD_DIRECTION', 'YIELD', 'WHEN_ISSUED_INDICATOR', 'REPORTING_PARTY_TYPE',
                    'CONTRA_PARTY_TYPE', 'ATS_INDICATOR', 'CHANGE_INDICATOR', 'ORIGINAL_MESSAGE_SEQUENCE_NUMBER', 'FUNCTION', 'HIGH_PRICE',
                    'HIGH_YIELD_DIRECTION', 'HIGH_YIELD', 'LOW_PRICE', 'LOW_YIELD_DIRECTION', 'LOW_YIELD', 'LAST_SALE_PRICE',
                    'LAST_SALE_YIELD_DIRECTION', 'LAST_SALE_YIELD', 'ORIGINAL_QUANTITY_INDICATOR', 'ORIGINAL_QUANTITY', 'ORIGINAL_PRICE',
                    'ORIGINAL_REMUNERATION', 'ORIGINAL_SPECIAL_PRICE_INDICATOR', 'ORIGINAL_SIDE', 'ORIGINAL_AS_OF_INDICATOR',
                    'ORIGINAL_EXECUTION_DATE_TIME', 'ORIGINAL_SALE_CONDITION_3', 'ORIGINAL_SALE_CONDITION_4', 'ORIGINAL_SETTLEMENT_DATE',
                    'ORIGINAL_YIELD_DIRECTION', 'ORIGINAL_YIELD', 'ORIGINAL_WHEN_ISSUED_INDICATOR', 'ORIGINAL_REPORTING_PARTY_TYPE',
                    'ORIGINAL_CONTRA_PARTY_TYPE', 'ORIGINAL_ATS_INDICATOR', 'ISSUER', 'ACTION', 'ACTION_DATE_TIME', 'HALT_REASON',
                    'DAILY_HIGH_PRICE', 'DAILY_HIGH_YIELD', 'DAILY_LOW_PRICE', 'DAILY_LOW_YIELD', 'DAILY_CLOSE_PRICE',
                    'CLOSE_YIELD_DIRECTION', 'DAILY_CLOSE_YIELD', 'TEXT', 'TOTAL_SECURITIES_TRADED_ALL_SECURITIES',
                    'TOTAL_SECURITIES_TRADED_INVESTMENT_GRADE', 'TOTAL_SECURITIES_TRADED_HIGH_YIELD', 'TOTAL_SECURITIES_TRADED_CONVERTIBLES',
                    'ADVANCES_ALL_SECURITIES', 'ADVANCES_INVESTMENT_GRADE', 'ADVANCES_HIGH_YIELD', 'ADVANCES_CONVERTIBLES',
                    'DECLINES_ALL_SECURITIES', 'DECLINES_INVESTMENT_GRADE', 'DECLINES_HIGH_YIELD', 'DECLINES_CONVERTIBLES',
                    'UNCHANGED_ALL_SECURITIES', 'UNCHANGED_INVESTMENT_GRADE', 'UNCHANGED_HIGH_YIELD', 'UNCHANGED_CONVERTIBLES',
                    'f_52_WEEK_HIGH_ALL_SECURITIES', 'f_52_WEEK_HIGH_INVESTMENT_GRADE', 'f_52_WEEK_HIGH_HIGH_YIELD', 'f_52_WEEK_HIGH_CONVERTIBLES',
                    'f_52_WEEK_LOW_ALL_SECURITIES', 'f_52_WEEK_LOW_INVESTMENT_GRADE', 'f_52_WEEK_LOW_HIGH_YIELD', 'f_52_WEEK_LOW_CONVERTIBLES',
                    'TOTAL_VOLUME_ALL_SECURITIES', 'TOTAL_VOLUME_INVESTMENT_GRADE', 'TOTAL_VOLUME_HIGH_YIELD', 'TOTAL_VOLUME_CONVERTIBLE',
                    'TOTAL_NUMBER_OF_TRANSACTIONS_ALL_SECURITIES', 'TOTAL_NUMBER_OF_TRANSACTIONS_CUSTOMER_BUY', 'TOTAL_SECURITIES_TRADED_CUSTOMER_BUY',
                    'TOTAL_VOLUME_CUSTOMER_BUY', 'TOTAL_NUMBER_OF_TRANSACTIONS_CUSTOMER_SELL', 'TOTAL_SECURITIES_TRADED_CUSTOMER_SELL',
                    'TOTAL_VOLUME_CUSTOMER_SELL', 'TOTAL_NUMBER_OF_TRANSACTIONS_AFFILIATE_BUY', 'TOTAL_SECURITIES_TRADED_AFFILIATE_BUY',
                    'TOTAL_VOLUME_AFFILIATE_BUY', 'TOTAL_NUMBER_OF_TRANSACTIONS_AFFILIATE_SELL', 'TOTAL_SECURITIES_TRADED_AFFILIATE_SELL',
                    'TOTAL_VOLUME_AFFILIATE_SELL', 'TOTAL_NUMBER_OF_TRANSACTIONS_INTER_DEALER', 'TOTAL_SECURITIES_TRADED_INTER_DEALER',
                    'TOTAL_VOLUME_INTER_DEALER']
cusip_to_rating_columns = ['cusip', 'date', 'rating']
cbonds_emissions_columns = ['Issuer(id)', 'Full name of the issuer (eng)', 'id issue', 'Issue name (eng)', 'Issue amount (prospectus)',
                    'FIGI / FIGI RegS', 'Bloomberg Ticker', 'Amortizing security (yes/no)', 'CFI / CFI RegS', 'CFI 144A', 'Сonvertable (yes/no)',
                    'Bond type (id)', 'Coupon rate (eng)', 'Coupon frequency', 'Currency (id)', 'Currency ', 'End of placement date', 'Start of trading',
                    'Start of placement date', 'Early redemption date', 'Day count convention (id)', 'Day count convention', 'Country of the issuer (id)',
                    'Country of the issuer (eng)', 'Calculation amount for international bonds', 'Floating rate (yes/no)', 'Indexation (eng)',
                    'Integral multiple', 'ISIN / ISIN RegS', 'ISIN 144A', 'Isin code 3', 'Kind (id)', 'Kind (eng)', 'Margin', 'Maturity date',
                    'Mortgage bonds (yes/no)', 'Minimum settlement amount / Calculation amount', 'Issue number (eng)',
                    'Next offer date (put/call) - to be specified', 'Outstanding calculation amount', 'Perpetual (yes/no)', 'Outstanding amount ',
                    'Precision of rounding of some fields (ACE, outstanding par value)', 'Reference rate (id)', 'Registration date',
                    'Volume in circulation on outstanding nominal', 'Restructing (yes/no)', 'Debt resructuring date', 'Covered debt (yes/no)', 'SEDOL',
                    'Settlement date', 'State registration number', 'Issue status (id)', 'Issue status (eng)', 'Structured products (yes/no)',
                    'Bond subtype (id)', 'Bond subtype (eng)', 'Subordinated debt (yes/no)', 'Sovereign bonds type (id)', 'Bond Issue form (id)',
                    'Bond rank (id)', 'Date until which the bond can be converted', 'Conditions of convertation (eng)', 'Current coupon date',
                    'Current coupon rate', 'Current coupon amount', 'Dual currency bond', 'Foreign bond', 'The flag that paper green bonds',
                    'Type of emission indexation', 'Type of emission indexation (eng)', 'Period in days  to which the conversion is not allowed',
                    'Additional information (eng)', 'Next offer date (call)', 'Next offer date (put)', 'Payment currency', 'pik', 'Placement type (eng)',
                    'Price at primary placement', 'Non-market issue (yes/no)', 'Placement method (eng)', 'Redemption Linked (yes/no)', 'Reference rate (eng)',
                    'Retail bonds', 'Securitisation', 'Series of issue', 'Issue form status (id)', 'Sukuk (yes/no)', 'Trace-eligible securities',
                    'Class of underlying asset (eng)', 'ISIN of underlying asset', 'Equivalent of the volume of emission in USD', 'formal issuer (id)',
                    'formal issuer (eng)', 'Updating date', 'Update time', 'Аmount in circulation (excluding redemptions and buyback)',
                    'CUSIP 144A', 'CUSIP / CUSIP RegS']
cbonds_issuers_columns = ['Issuer id', 'Vat number (for Russia)', 'Issuer short name (eng)', 'Full issuer name (eng)', 'Group of issuers (id)',
                    'Group of issuers (eng)', 'Issuer sector (id)', 'Issuer sector (eng)', 'Industry (id)', 'Industry (eng)',
                    'Company doesn`t exist (yes/no)', 'Issuer address (eng)', 'Issuer email', 'Issuer fax', 'Issuer phone number',
                    'Legal issuer address (eng)', 'Issuer website', 'SPV (yes/no)', 'Company profile (eng)', 'Acquiring company (id)',
                    'Individual taxpayer number (TIN, EDRPOU, etc.)', 'Country of risk (id)', 'Country of risk (eng)', 'CIK number', 'LEI code',
                    'International issuer (yes/no)', 'Inverstor relation website (eng)', 'Renewal date', 'SWIFT code',
                    'VAT number for companies from Europe', 'Sanctions issuer']
bondcliq_columns = ['279', '269', '278', '48', '22', '541', '223', '107', '270', '218', '271', '272', '273', '126', '448']

expanded_re = re.compile(r'^.*/expanded/(.*)$')

def create_filter(column: str):
    """Creates a helper function that filters a table by only keeping rows where the value in 'column' is also in 'cusips'"""
    def filter(t: pa.table, _):
        return t.filter(pc.is_in(t[column], cusips))
    return filter

def process(s3url, simulated_s3url=None, file_filter: Callable[[list[str]], list[str]]=lambda l: l, table_filter: Callable[[pa.table,str], pa.table]=create_filter('cusip'),
                    get_columns: Callable[[str], list[str]]=lambda _: [], delimiter=',', quote_char: str|Literal[False]=False, invalid_row_handler: Callable=invalid_row_handler_default):
    print(f'processing {s3url}')
    with tempfile.TemporaryDirectory() as temp_dir:
        upload = False
        zip_path = os.path.join(temp_dir, 'the_new_zip_archive.zip')
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED,
                                compresslevel=settings.get('$.data_compression_level')) as zip:
            file_paths: list[str] = cast(list[str], s3.get_object(s3url))
            for file_path in file_filter(file_paths):
                columns = get_columns(os.path.basename(file_path))
                # treat all columns as strings so PyArrow doesn't infer data types and alter the data representation
                column_types = {k: pa.string() for k in columns}
                print(f'filtering {file_path}')
                t = table_filter(csv.read_csv(file_path,
                    parse_options=csv.ParseOptions(
                        delimiter=delimiter,
                        invalid_row_handler=invalid_row_handler,
                        quote_char=quote_char
                    ),
                    convert_options=csv.ConvertOptions(
                        column_types=column_types,
                        include_columns=[k for k in column_types.keys()],
                        strings_can_be_null=True
                    )), os.path.basename(file_path))
                if len(t) > 0:
                    temp_file_name = os.path.join(temp_dir, 'tempfile')
                    if quote_char:
                        # if quotes are allowed we can use pyarrow
                        csv.write_csv(t, temp_file_name, csv.WriteOptions(delimiter=delimiter, quoting_style='needed'))
                    else:
                        # otherwise we have to manually output sub-optimal csv data
                        with open(temp_file_name, 'w') as f:
                            f.write(f"{delimiter.join(columns)}\n")
                            i = 0
                            while i < len(t):
                                f.write(f"{delimiter.join([v if v is not None else '' for v in [t[c][i].as_py() for c in columns]])}\n")
                                i += 1
                    zip.write(temp_file_name, arcname=expanded_re.search(file_path).group(1))
                    upload = True
        if upload:
            match = s3.s3url_re.search(simulated_s3url if simulated_s3url else s3url)
            if match:
                newurl = f's3://deepmm.test.data/{match.group(1)}/{match.group(2)}'
                print(f'newurl: {newurl}')
                s3.upload_file(zip_path, newurl)
            else:
                raise ValueError(f'No regex match for {s3url}')

def process_multiple(s3_base_url, zip_filter: Callable[[list[str]], list[str]]=lambda l: l, file_filter: Callable[[list[str]], list[str]]=lambda l: l,
                    table_filter: Callable[[pa.table,str], pa.table]=create_filter('cusip'), get_columns: Callable[[str], list[str]]=lambda _: [], delimiter=',',
                    quote_char: str|Literal[False]=False, invalid_row_handler: Callable=invalid_row_handler_default):
    for s3path in zip_filter([p for p in cast(list[str], s3.get_all_paths_with_prefix(s3_base_url)) if p.lower().endswith('.zip')]):
        process(s3path, file_filter=file_filter, table_filter=table_filter, get_columns=get_columns, delimiter=delimiter,
                        quote_char=quote_char, invalid_row_handler=invalid_row_handler)

def create_until_filter(s: str, pre_process: Callable[[str], str]=lambda v: v):
    """Creates a helper function that filters a list by including only the entries where
        calling the 'pre_process' function on the entry returns a value less than or equal to
        the target value 's'"""
    def filter(l: list[str]):
        return [val for val in l if pre_process(val) <= s]
    return filter

def main():
    with CleanEnvironment():
        upper_filter = create_filter('CUSIP')
        lower_filter = create_filter('cusip')
        process(
                        's3://deepmm.data/finra/historical/trace_cbc.zip',
                        table_filter = upper_filter,
                        get_columns = lambda _: finra_historical_columns,
                        delimiter = delimiter_historical,
                        quote_char = False,
                        invalid_row_handler = invalid_row_handler_historical)
        process(
                        's3://deepmm.data/finra/historical/trace_cb_144ac.zip',
                        table_filter = upper_filter,
                        get_columns = lambda _: finra_historical_columns,
                        delimiter = delimiter_historical,
                        quote_char = False,
                        invalid_row_handler = invalid_row_handler_historical)
        process_multiple(
                        's3://deepmm.data/finra/trace_cbc/',
                        zip_filter = create_until_filter('MPPBTDSS_CUSIP_20230302.zip', os.path.basename),
                        table_filter = upper_filter,
                        get_columns = lambda _: finra_eod_columns,
                        delimiter = delimiter_eod,
                        quote_char = False,
                        invalid_row_handler = invalid_row_handler_eod)
        process_multiple(
                        's3://deepmm.data/finra/trace_cb_144ac/',
                        zip_filter = create_until_filter('MPPBT14S_CUSIP_20230302.zip', os.path.basename),
                        table_filter = upper_filter,
                        get_columns = lambda _: finra_eod_columns,
                        delimiter = delimiter_eod,
                        quote_char = False,
                        invalid_row_handler = invalid_row_handler_eod)
        def cbonds_filter(t: pa.table, file_name: str):
            if file_name.startswith('emissions'):
                return t.filter(pc.or_kleene(pc.is_in(t['CUSIP 144A'], cusips), pc.is_in(t['CUSIP / CUSIP RegS'], cusips)))
            elif file_name.startswith('issuers'):
                return t
            else:
                return pa.table({})
        def get_cbonds_columns(file_name: str):
            if file_name.startswith('emissions'):
                return cbonds_emissions_columns
            elif file_name.startswith('issuers'):
                return cbonds_issuers_columns
        process(
                        's3://deepmm.data/cbonds/2023-03-02.zip',
                        file_filter = lambda file_paths: [f for f in file_paths if os.path.basename(f).startswith('emissions') or os.path.basename(f).startswith('issuers')],
                        table_filter = cbonds_filter,
                        get_columns = get_cbonds_columns,
                        delimiter = ';',
                        quote_char = False)
        process_multiple(
                        's3://deepmm.data/bondcliq/',
                        zip_filter = create_until_filter('pretrade_0804_20230302.zip', os.path.basename),
                        file_filter = lambda file_paths: [f for f in file_paths if os.path.basename(f).startswith('pretrade_')],
                        table_filter = create_filter('48'),
                        get_columns = lambda _: bondcliq_columns,
                        delimiter = ',',
                        quote_char = False)
        file_list = s3.get_object('s3://deepmm.data/moodys/xbrl100-2023-02-15.zip')
        cusip_set = set(bond[0] for bond in BONDS)
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'tempfile')
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED,
                                    compresslevel=settings.get('$.data_compression_level')) as zip:
                for input_file in (file for file in file_list if file.lower().endswith('.xml')):
                    print(f'processing {os.path.basename(input_file)}')
                    tree = ET.parse(input_file)
                    root = tree.getroot()
                    for cusip in root.iterfind('./ns:ROCRA/ns:ISD/ns:IND/ns:CUSIP', namespaces):
                        if cusip.text in cusip_set:
                            zip.write(input_file, arcname=expanded_re.search(input_file).group(1))
                            break
            s3.upload_file(zip_path, 's3://deepmm.test.data/deepmm.data/moodys/xbrl100-2023-02-15.zip')

if __name__ == '__main__':
    main()
