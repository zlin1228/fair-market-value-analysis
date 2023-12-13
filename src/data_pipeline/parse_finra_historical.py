import re
from typing import cast, Dict, List, Literal, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from data_pipeline.csv_parser import parse_csv_files
from performance_helper import create_profiler
import s3
import settings

# https://www.finra.org/filing-reporting/trace/historic-data-file-layout-post-2612-files

_delimiter: Literal['|'] = '|'
_quote_char: Literal['"'] = '"'
_column_types: Dict[str, pa.DataType] = {
    'Record Count Num': pa.uint32(),
    'Execution Date': pa.string(),
    'Execution Time': pa.string(),
    'Trade Report Date': pa.string(),
    'Trade Report Time': pa.string(),
    'Bloomberg Identifier': pa.string(),
    'CUSIP': pa.string(),
    'Quantity': pa.float64(),
    'Buy/Sell Indicator': pa.string(),
    'Contra Party Indicator': pa.string(),
    'ATS Indicator': pa.string(),
    'Price': pa.float64(),
    'Yield': pa.float64(),
    'Yield Direction': pa.string(),
    'Trade Status': pa.string(),
    'Sub Product': pa.string(),
    'When Issued Indicator': pa.string(),
    'Dissemination Flag': pa.string(),
    'As Of Indicator': pa.string()
}
def _parse_finra_historical_date(table: pa.table, columns: Tuple[str,str]) -> pa.array:
    return pc.assume_timezone(
                pc.strptime(
                    pc.binary_join_element_wise(
                        table.column(columns[0]), table.column(columns[1]), 'T'
                    ),
                    format='%Y%m%dT%H%M%S',
                    unit='ns'
                ),
            settings.get('$.finra_timezone')).cast(pa.int64())

_digits_re = re.compile(r'^\d+$')
def _invalid_row_handler(row) -> Literal['skip', 'error']:
    if _digits_re.match(row.text):
        return 'skip'
    return 'error'

def _create_process(trade_source: str):
    def _process(table: pa.table) -> pa.table:
        # The Enhanced Historic Time and Sales dataset includes disseminated and non-disseminated
        # transactions, indicated by the Dissemination Flag. Inter-Dealer Buys
        # (Contra Party Indicator = D, Buy/Sell Indicator = B) and Inter-Dealer Sells
        # (Contra Party Indicator = D, Buy/Sell Indicator = S) reflect two sides of the same trade,
        # reported from each member firm’s perspective. Only the Inter-Dealer Sell trade report is
        # included in real-time dissemination. As long as the Inter-Dealer Sell trade meets the
        # eligibility criteria for dissemination, the Dissemination Flag in the dataset shall reflect
        # the value “Y.”
        # Dissemination Flag
        # Indicates whether the trade was disseminated (via BTDS, or ATDS for Agency Bonds) or not.
        # Applicable values are:
        # Y   =   Trade was disseminated
        # N   =   Trade was not disseminated
        table = table.filter(pc.equal(table.column('Dissemination Flag'), 'Y'))
        return pa.table({
                'as_of_indicator': table.column('As Of Indicator'),
                'ats_indicator': table.column('ATS Indicator').fill_null('N'),
                'buy_sell': table.column('Buy/Sell Indicator'),
                'cusip': table.column('CUSIP'),
                'execution_date': _parse_finra_historical_date(table, ('Execution Date', 'Execution Time')),
                'figi': table.column('Bloomberg Identifier'),
                'message_category': pa.nulls(len(table), pa.string()), # finra eod needs this column
                'message_type': pa.nulls(len(table), pa.string()), # finra eod needs this column
                'price': table.column('Price'),
                'quantity': table.column('Quantity'),
                'record_num': table.column('Record Count Num'),
                'report_date': _parse_finra_historical_date(table, ('Trade Report Date', 'Trade Report Time')),
                # NOTE: this field has B,S,N as possible values where EOD does not
                'side': table.column('Contra Party Indicator'),
                'sub_product': table.column('Sub Product'),
                'trade_original_format': pa.nulls(len(table), pa.string()).fill_null('finra_historical'),
                'trade_status': table.column('Trade Status'),
                'trade_source': pa.nulls(len(table), pa.string()).fill_null(trade_source),
                'when_issued_indicator': table.column('When Issued Indicator'),
                'yield': pc.multiply_checked(
                            pc.if_else(
                                pc.is_null(table.column('Yield Direction')), 1, -1),
                            table.column('Yield'))})
    return _process

def parse_finra_historical(data_path: str, trade_source: str) -> pa.table:
    with create_profiler('parse all finra historical')() as p:
        file_paths: List[str] = cast(List[str], s3.get_object(data_path))
        result = parse_csv_files(file_paths, _column_types, delimiter=_delimiter,
                        quote_char=_quote_char, process=_create_process(trade_source), invalid_row_handler=_invalid_row_handler)
        p.print(f'rows: {result.num_rows}')
        return result
