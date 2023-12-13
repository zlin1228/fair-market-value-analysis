import sys
sys.path.insert(0, '../')

standard_settings = {
    'ratings_enabled': True,
    '$.data_path': 's3://deepmm.test.data/deepmm.parquet/v0.7.zip'}
import settings
settings.override(standard_settings)

import git
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import unittest

from load_data import get_initial_quotes, get_initial_trades, get_ordinals, get_table
import s3
from tests.helpers import CleanEnvironment

def _create_get_s3_path():
    prefix = 's3://deepmm.test.data/load_data_reference/'
    load_data_reference_files = set(s3.get_all_paths_with_prefix(prefix))
    repo = git.Repo('.', search_parent_directories=True)
    hashes = []
    for commit in repo.iter_commits():
        hashes.append(commit.hexsha)
    def _get_s3_path(path: str):
        for hash in hashes:
            s3_path = f'{prefix}{hash}/{path}'
            if s3_path in load_data_reference_files:
                return s3_path
        raise FileExistsError(f's3 path not found for "{path}"')
    return _get_s3_path
get_s3_path = _create_get_s3_path()

def assert_table_equivalent_to_s3_table(table, s3_path):
    ref_table = pq.read_table(s3.get_object(s3_path)[0])
    if table.column_names != ref_table.column_names:
        raise KeyError('columns differ')
    if len(table) != len(ref_table):
        raise ValueError('table lengths differ')
    for c in table.column_names:
        if not pc.all(pc.equal(table[c], ref_table[c]), skip_nulls=True).as_py():
            raise ValueError(f'values differ in column "{c}"')
        if not pc.all(pc.equal(pc.is_null(table[c]), pc.is_null(ref_table[c]))).as_py():
            raise ValueError(f'null values differ in column "{c}"')

class TestGetInitialQuotes(unittest.TestCase):

    def test_get_initial_quotes(self):
        with CleanEnvironment(lambda _: standard_settings):
            quotes = get_initial_quotes()
            assert_table_equivalent_to_s3_table(quotes, get_s3_path('quotes.parquet'))

class TestGetInitialTrades(unittest.TestCase):

    def test_get_initial_trades(self):
        with CleanEnvironment(lambda _: standard_settings):
            trades = get_initial_trades()
            assert_table_equivalent_to_s3_table(trades, get_s3_path('trades.parquet'))

    def test_get_initial_trades_proportion_of_trades_to_load(self):
        with CleanEnvironment(lambda _: {
                        **standard_settings,
                        '$.data.trades.proportion_of_trades_to_load': 0.1}):
            trades = get_initial_trades()
            assert_table_equivalent_to_s3_table(trades, get_s3_path('trades_proportion_of_trades_to_load.parquet'))

    def test_get_initial_trades_ratings_disabled(self):
        with CleanEnvironment(lambda _: {
                        **standard_settings,
                        'ratings_enabled': False}):
            trades = get_initial_trades()
            assert_table_equivalent_to_s3_table(trades, get_s3_path('trades_ratings_disabled.parquet'))

    def test_get_initial_trades_different_investment_grade_ratings(self):
        with CleanEnvironment(lambda _: {
                        **standard_settings,
                        '$.data.ratings.investment_grade_ratings': ['Baa2', 'Baa3', 'Caa1', 'Caa2', 'DOES_NOT_EXIST']}):
            trades = get_initial_trades()
            assert_table_equivalent_to_s3_table(trades, get_s3_path('trades_different_investment_grade_ratings.parquet'))

class TestAppliedOrdinals(unittest.TestCase):

    def test_applied_ordinals(self):
        with CleanEnvironment(lambda _: standard_settings):
            ordinals = get_ordinals()
            for table_name in ['bonds', 'quotes', 'ratings', 'trades']:
                table = get_table(table_name)
                table = pa.table({
                        **{ ordinal: ordinals[ordinal].take(table[ordinal]) for ordinal in ordinals.keys() if ordinal in table.column_names },
                        **{ non_ordinal: table[non_ordinal] for non_ordinal in table.column_names if non_ordinal not in ordinals.keys() }})
                assert_table_equivalent_to_s3_table(table, get_s3_path(f'applied_ordinals/{table_name}.parquet'))

if __name__ == '__main__':
    unittest.main()
