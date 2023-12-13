import re

import pyarrow as pa
import pyarrow.compute as pc

from data_pipeline.csv_parser import parse_csv_files
from data_pipeline.openfigi import update_and_load_openfigi_cache
import s3
import settings

_delimiter=';'
# NOTE: Cbonds uses ; as delimiter and does not seem to have a quote character
# but does include quotes mid-value
_quote_char=False

def _parse_cbonds_date_column(column) -> pa.array:
    return pc.assume_timezone(pc.strptime(column, format='%Y-%m-%d', unit='ns'),
                                settings.get('$.finra_timezone')).cast(pa.int64())

_emissions_column_types: dict[str, pa.DataType] = {
                    # 'Bloomberg Ticker': pa.string(),
                    'Coupon rate (eng)': pa.string(),
                    'Coupon frequency': pa.uint8(),
                    'Currency ': pa.string(), # trailing space intentional
                    'Current coupon amount': pa.float64(),
                    'Current coupon date': pa.string(),
                    'Current coupon rate': pa.float64(),
                    'CUSIP 144A': pa.string(),
                    'CUSIP / CUSIP RegS': pa.string(),
                    'Day count convention': pa.string(),
                    'End of placement date': pa.string(),
                    # 'FIGI / FIGI RegS': pa.string(),
                    'Full name of the issuer (eng)': pa.string(),
                    'ISIN 144A': pa.string(),
                    'ISIN / ISIN RegS': pa.string(),
                    'Issuer(id)': pa.string(),
                    'Maturity date': pa.string(),
                    'Outstanding amount ': pa.float64(), # trailing space intentional
                    'Settlement date': pa.string()}
def _process_emissions(table: pa.table) -> pa.table:
    return pa.table({
            # 'bloomberg_ticker': table.column('Bloomberg Ticker'),
            'coupon_frequency': table.column('Coupon frequency'),
            'coupon_string': table.column('Coupon rate (eng)'),
            'currency': table.column('Currency '),
            'current_coupon_amount': table.column('Current coupon amount'),
            'current_coupon_date': _parse_cbonds_date_column(table.column('Current coupon date')),
            'current_coupon_rate': table.column('Current coupon rate'),
            'cusip_144a': table.column('CUSIP 144A'),
            'cusip_regs': table.column('CUSIP / CUSIP RegS'),
            'day_count_convention': table.column('Day count convention'),
            'end_of_placement_date': _parse_cbonds_date_column(table.column('End of placement date')),
            # 'figi': table.column('FIGI / FIGI RegS'),
            'isin_144a': table.column('ISIN 144A'),
            'isin_regs': table.column('ISIN / ISIN RegS'),
            'issue_date': _parse_cbonds_date_column(table.column('Settlement date')),
            'issuer': table.column('Full name of the issuer (eng)'),
            'issuer_id': table.column('Issuer(id)'),
            'maturity': _parse_cbonds_date_column(table.column('Maturity date')),
            'outstanding': table.column('Outstanding amount ')})

_issuers_column_types: dict[str, pa.DataType] = {
                    'Issuer id': pa.string(),
                    'Issuer sector (eng)': pa.string(),
                    'Industry (eng)': pa.string()}
def _process_issuers(table: pa.table) -> pa.table:
    return pa.table({
            'issuer_id': table['Issuer id'],
            'sector': table['Issuer sector (eng)'],
            'industry': table['Industry (eng)']})

emissions_re = re.compile(r'^.*/expanded/emissions(_\d+)?\.csv$')
issuers_re = re.compile(r'^.*/expanded/issuers\.csv$')

def parse_cbonds():
    paths = s3.get_all_paths_with_prefix(f"{settings.get('$.raw_data_path')}cbonds/")
    paths = [p for p in paths if p.lower().endswith('.zip')]
    paths.sort()
    file_list = s3.get_object(paths[-1])
    cbonds_data = parse_csv_files([file for file in file_list if emissions_re.match(file)],
                                _emissions_column_types, delimiter=_delimiter,
                                quote_char=_quote_char, process=_process_emissions)
    issuers = parse_csv_files([file for file in file_list if issuers_re.match(file)],
                                _issuers_column_types, delimiter=_delimiter,
                                quote_char=_quote_char, process=_process_issuers)
    # NOTE: WE ONLY KEEP BONDS THAT HAVE A cusip_regs AND/OR A cusip_144a
    # BECAUSE WE USE CUSIP TO JOIN WITH THE OTHER DATA SOURCES
    # regs
    cbonds_data_regs = cbonds_data.filter(pc.true_unless_null(cbonds_data['cusip_regs']))
    cbonds_data_regs = pa.table({'cusip': cbonds_data_regs['cusip_regs'],
                                 'isin': cbonds_data_regs['isin_regs'],
                                 **{ c: cbonds_data_regs[c] for c in cbonds_data_regs.column_names if c not in
                                        ['cusip_144a', 'cusip_regs', 'isin_144a', 'isin_regs'] }})
    cbonds_data_regs = cbonds_data_regs.group_by(cbonds_data_regs.column_names).aggregate([])
    # 144a
    cbonds_data_144a = cbonds_data.filter(pc.true_unless_null(cbonds_data['cusip_144a']))
    cbonds_data_144a = pa.table({'cusip': cbonds_data_144a['cusip_144a'],
                                 'isin': cbonds_data_144a['isin_144a'],
                                 **{ c: cbonds_data_144a[c] for c in cbonds_data_144a.column_names if c not in
                                        ['cusip_144a', 'cusip_regs', 'isin_144a', 'isin_regs'] }})
    cbonds_data_144a = cbonds_data_144a.group_by(cbonds_data_144a.column_names).aggregate([])
    # combine them back together
    cbonds_data = pa.concat_tables([cbonds_data_regs, cbonds_data_144a])
    # join with issuers
    cbonds_data = cbonds_data.join(issuers, 'issuer_id', join_type='left outer')
    # join with openfigi
    openfigi_cache = update_and_load_openfigi_cache(cbonds_data['cusip'])
    # only pull these columns from OpenFigi
    openfigi_cache = pa.table({ c: openfigi_cache[c] for c in ['cusip', 'exchCode', 'figi', 'marketSector', 'securityType', 'securityType2', 'ticker']})
    cbonds_data = cbonds_data.join(openfigi_cache, 'cusip', 'cusip', 'left outer')
    # remove any duplicates
    cbonds_data = cbonds_data.group_by(cbonds_data.column_names).aggregate([])
    return cbonds_data
