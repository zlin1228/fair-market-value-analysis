import sys
sys.path.insert(0, '../src')
import settings
settings.override_if_main(__name__, 1)

import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.csv as csv

from data_pipeline.helpers import ordinal_re, table_re
from pyarrow_helpers import filter_by_value_in_column
import s3

def read_parquet():
    tables = {}
    ordinals = {}
    file_list = s3.get_object(settings.get('$.data_path'))
    for file in file_list:
        if match := ordinal_re.search(file):
            ordinal = match.group(1)
            ordinals[ordinal] = pq.read_table(file)
    for file in file_list:
        if match := table_re.search(file):
            table = match.group(1)
            tables[table] = pq.read_table(file)
    for table in tables:
        tables[table] = pa.table({
                **{ ordinal: ordinals[ordinal].take(tables[table][ordinal])[ordinal] for ordinal in ordinals.keys() if ordinal in tables[table].column_names },
                **{ non_ordinal: tables[table][non_ordinal] for non_ordinal in tables[table].column_names if non_ordinal not in ordinals.keys() }})
    return tables

def read_openfigi_cache():
    file_list = s3.get_object(f"{settings.get('$.raw_data_path')}openfigi/openfigi_cache.zip")
    return csv.read_csv(file_list[0])

def print_index(table, index):
    columns = table.column_names
    columns.sort()
    for column in columns:
        print(f'{column}: "{table[column][index].as_py()}"')

OUTPUT_DIR = './parquet_output/'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
def write_csv(table: pa.table, file: str):
    csv.write_csv(table, os.path.join(OUTPUT_DIR, file))

tables = read_parquet()

t = pa.table({ c: tables['trades'][c] for c in ['figi', 'cusip']})
b = tables['bonds']
FILTER_BONDS = False
if FILTER_BONDS:
    for c in ['figi', 'cusip', 'maturity', 'coupon_string', 'issuer', 'issue_date', 'outstanding', 'sector', 'industry']:
        b = b.filter(pc.true_unless_null(b[c]))
    b = b.filter(pc.equal(b['sector'], 'corporate'))
b = pa.table({ c: b[c] for c in ['ticker', 'figi', 'cusip', 'maturity', 'current_coupon_rate', 'coupon_string']})

t_count = len(t)
tuc = pc.unique(t['cusip'])
tuc_count = len(tuc)

coupon = pc.extract_regex(b['coupon_string'], r'^(?P<coupon>\d+(?:\.\d+)?)%$')
coupon = pc.struct_field(coupon, [0])
b = b.append_column('coupon', coupon.cast(pa.float64()))

tjb = t.join(b, 'cusip', 'cusip', 'left outer', '', '_b')

def cusip_report(table, prompt):
    trade_count = len(table)
    cusip_count = len(pc.drop_null(pc.unique(table['cusip'])))
    print(f'{prompt}:')
    print(f"    {'{0:0.3f}'.format(100 * trade_count / t_count)}% of all trades ({trade_count} / {t_count})")
    print(f"    represents {'{0:0.3f}'.format(100 * cusip_count / tuc_count)}% of all trade cusips ({cusip_count} / {tuc_count})")

def unique_with_counts(table, column_names=None, count_column=None):
    column_names = column_names if column_names else table.column_names
    count_column = count_column if count_column else column_names[0]
    return table.group_by(column_names).aggregate([(count_column, 'count')]).sort_by([(f'{count_column}_count', 'descending')])

matched_trades = tjb.filter(pc.equal(tjb['figi'], tjb['figi_b']))
cusip_report(matched_trades, 'trades with cusip matching cbonds and openfigi and figi matching openfigi')

matched_trades_matched_coupon_string = matched_trades.filter(pc.true_unless_null(matched_trades['coupon']))
cusip_report(matched_trades_matched_coupon_string, "trades we can match that have coupon_string that matches r'^(?P<coupon>\d+(?:\.\d+)?)%$'")
write_csv(unique_with_counts(matched_trades_matched_coupon_string, ['cusip', 'figi', 'figi_b', 'ticker', 'coupon_string'], 'cusip'), 'matched_trades_matched_coupon_string.csv')

matched_trades_unmatched_coupon_string = matched_trades.filter(pc.is_null(matched_trades['coupon']))
cusip_report(matched_trades_unmatched_coupon_string, "trades we can match that do NOT have coupon_string that matches r'^(?P<coupon>\d+(?:\.\d+)?)%$'")
write_csv(unique_with_counts(matched_trades_unmatched_coupon_string, ['cusip', 'figi', 'figi_b', 'ticker', 'coupon_string'], 'cusip'), 'matched_trades_unmatched_coupon_string.csv')

trades_we_cannot_match_or_that_have_unmatched_coupon_string = tjb.filter(pc.or_kleene(pc.or_kleene(pc.is_null(tjb['figi_b']), pc.invert(pc.equal(tjb['figi'], tjb['figi_b']))), pc.is_null(tjb['coupon'])))
cusip_report(trades_we_cannot_match_or_that_have_unmatched_coupon_string, "trades we cannot match or that do not have coupon_string that matches r'^(?P<coupon>\d+(?:\.\d+)?)%$'")
write_csv(unique_with_counts(trades_we_cannot_match_or_that_have_unmatched_coupon_string, ['cusip', 'figi', 'figi_b', 'ticker', 'coupon_string'], 'cusip'), 'trades_we_cannot_match_or_that_have_unmatched_coupon_string.csv')

def process_common_bond_category(category):
    cat = csv.read_csv(f'./{category}.csv')
    cusip = pc.extract_regex(cat['ISIN'], r'^..(?P<cusip>.........).$')
    cusip = pc.struct_field(cusip, [0])
    cat = cat.append_column('cusip', cusip)

    cat_in = filter_by_value_in_column(cat, 'cusip', matched_trades_matched_coupon_string, ['cusip'])
    cat_out = filter_by_value_in_column(cat, 'cusip', matched_trades_matched_coupon_string, ['cusip'], True)

    def report_cat(ct, in_or_outside):
        ct = pa.table({c: ct[c] for c in ['Description', 'ISIN', 'Ccy']})
        ct = ct.sort_by([('Description', 'ascending'), ('ISIN', 'ascending'), ('Ccy', 'ascending')])
        print(f'{category} {in_or_outside} our universe:')
        print(f"    {'{0:0.3f}'.format(100 * len(ct) / len(cat))}% ({len(ct)} / {len(cat)})")
        write_csv(ct, f'{category}-{in_or_outside}.csv')

    report_cat(cat_in, 'in')
    report_cat(cat_out, 'outside')

# James provided index files containing a set of commonly traded High Yield and Investement Grade bonds
# to analyze our coverage, name them HY.csv and IG.csv and put them next to this file
# and uncomment the following lines:
# process_common_bond_category('HY')
# process_common_bond_category('IG')
