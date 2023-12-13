import os
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

from data_pipeline.helpers import table_re, ordinal_re
from grouped_ratings import GroupedRatings
from performance_helper import create_profiler
from pyarrow_helpers import drop_duplicates, sample
import s3
import settings
from update_model_data import UNSPECIFIED_ORDINAL_VALUE

def _create_get_table():
    '''Helper function to cache get_table results'''
    _cached_tables = {}
    def _get_table(name: Literal['bonds', 'quotes', 'ratings', 'trades']):
        '''Get data table'''
        nonlocal _cached_tables
        if name not in _cached_tables:
            file_list = s3.get_object(settings.get('$.data_path'))
            profile = create_profiler(f'loading table: "{name}"')
            with profile():
                for file in file_list:
                    if table_match := table_re.search(file):
                        table = table_match.group(1)
                        if table == name:
                            _cached_tables[table] = pq.read_table(file)
                            break
        return _cached_tables[name]
    return _get_table
get_table = _create_get_table()

def _create_get_ordinals():
    '''Helper function to cache get_ordinals results'''
    _cached_result = None
    def _get_ordinals():
        '''Get ordinals'''
        nonlocal _cached_result
        if _cached_result is None:
            file_list = s3.get_object(settings.get('$.data_path'))
            profile = create_profiler(f'loading ordinals')
            with profile():
                _cached_result = {}
                for file in file_list:
                    if ordinal_match := ordinal_re.search(file):
                        ordinal = ordinal_match.group(1)
                        _cached_result[ordinal] = pq.read_table(file)[ordinal].combine_chunks()
        return _cached_result
    return _get_ordinals
get_ordinals = _create_get_ordinals()

def parse_coupon_string(coupon_string: pa.array):
    '''Parse an array of coupon_string values and return an array of coupons.
       This function is public because it is also used when creating the bond index for the UI.'''
    # create coupon by parsing coupon_string
    # note: coupon_string is more complete and accurate than the coupon field
    # TODO: improve this to include more coupon_string formats
    parsed_coupon_string = pc.extract_regex(coupon_string, r'^(?P<coupon>\d+(?:\.\d+)?)%$')
    return pc.struct_field(parsed_coupon_string, [0]).cast(pa.float64())

def _process_bonds(bonds: pa.table):
    '''Accepts, processes, and returns a pyarrow table of bonds'''
    profile = create_profiler('_process_bonds')
    with profile():
        coupon = parse_coupon_string(get_ordinals()['coupon_string'].take(bonds['coupon_string']))
        bonds = bonds.append_column('coupon', coupon)
        bonds = bonds.select(['coupon', 'cusip', 'figi', 'industry', 'issue_date',
                                                'issuer', 'maturity', 'outstanding', 'sector'])
        if settings.get('$.tolerate_missing_reference_data'):
            # TODO: add ordinal values ('unknown') if needed
            bonds = pa.table({
                    'figi': bonds['figi'],
                    'cusip': bonds['cusip'].fill_null('unknown'),
                    'issuer': bonds['issuer'],
                    'issue_date': bonds['issue_date'].fill_null(0),
                    'outstanding': bonds['outstanding'].fill_null(-1),
                    'maturity': bonds['maturity'].fill_null(0),
                    'coupon': bonds['coupon'].fill_null(-100),
                    'sector': bonds['sector'].fill_null('unknown'),
                    'industry': bonds['industry'].fill_null('unknown')
                })
        # only keep bonds with a maturity > 0
        mask = pc.greater(bonds['maturity'], 0)
        # filter out bonds with null in any column
        for column in (c for c in bonds.column_names):
            mask = pc.and_(mask, pc.true_unless_null(bonds[column]))
        bonds = bonds.filter(mask)
        bonds = bonds.sort_by([('figi', 'ascending')])
        bonds = drop_duplicates(bonds, ['figi'])
        return bonds

def cache_result(func):
    _func_called = False
    _cached_result = None
    def inner_func(*args, **kwargs):
        nonlocal _func_called, _cached_result
        if not _func_called:
            _cached_result = func(*args, **kwargs)
            _func_called = True
        return _cached_result
    return inner_func

def _create_process_quotes():
    '''Helper function to create process_quotes so we can cache values used for each call'''
    # TODO: determine if using trades for this is optimal
    trades = get_table('trades')
    unique_cusip_figi_pairs = trades.select(['cusip', 'figi'])
    unique_cusip_figi_pairs = unique_cusip_figi_pairs.group_by(unique_cusip_figi_pairs.column_names).aggregate([])
    unique_cusip_figi_pairs = unique_cusip_figi_pairs.drop_null()
    def _process_quotes(quotes: pa.table):
        '''Accepts, processes, and returns a pyarrow table of quotes'''
        profile = create_profiler('process_quotes')
        with profile():
            with profile('join with unique cusip figi pairs'):
                quotes = quotes.join(unique_cusip_figi_pairs, 'cusip', 'cusip', 'inner')
            with profile('select'):
                quotes = quotes.select(['figi', 'party_id', 'entry_type', 'entry_date', 'price', 'quantity'])
            with profile('sort_by'):
                quotes = quotes.sort_by([(c, 'ascending') for c in quotes.column_names])
            return quotes
    return _process_quotes
create_process_quotes = cache_result(_create_process_quotes)

def sample_trades(trades: pa.table):
    '''Accepts, samples, and returns a pyarrow table of trades'''
    fraction = settings.get('$.data.trades.proportion_of_trades_to_load')
    if fraction != 1.0:
        profile = create_profiler('sample proportion of trades')
        with profile():
            trades = sample(trades, frac=fraction)
    return trades

def _create_process_trades():
    '''Helper function to create process_trades so we can cache values used for each call'''

    ordinals = get_ordinals()
    ratings = GroupedRatings(get_table('ratings'))
    bonds = _process_bonds(get_table('bonds'))

    # TODO: add values to ordinals if needed
    def _create_get_ordinal_value():
        ordinals_cache = {}
        def _get_ordinal_value(ordinal: str, value: str):
            nonlocal ordinals_cache
            if ordinal not in ordinals_cache:
                ordinals_cache[ordinal] = {}
            if value not in ordinals_cache[ordinal]:
                ordinals_cache[ordinal][value] = ordinals[ordinal].index(value)
                if ordinals_cache[ordinal][value].as_py() < 0:
                    raise KeyError(f'Ordinal "{ordinal}" missing value "{value}"')
            return ordinals_cache[ordinal][value]
        return _get_ordinal_value
    get_ordinal_value = _create_get_ordinal_value()

    investment_grade_rating_ordinals = []
    for i, v in enumerate(ordinals['rating'].to_pylist()):
        if v in settings.get('$.data.ratings.investment_grade_ratings'):
            investment_grade_rating_ordinals.append(i)

    def _process_trades(trades: pa.table) -> pa.table:
        '''Accepts, processes, and returns a pyarrow table of trades'''
        profile = create_profiler('process_trades')
        with profile():
            with profile('filter'):
                # true_unless_null - figi, cusip
                mask = pc.true_unless_null(trades['figi'])
                mask = pc.and_(mask, pc.true_unless_null(trades['cusip']))
                # is_null - as_of_indicator
                mask = pc.and_(mask, pc.is_null(trades['as_of_indicator']))
                # when_issued_indicator = N
                mask = pc.and_(mask, pc.equal(trades['when_issued_indicator'], get_ordinal_value('when_issued_indicator', 'N')))
                # sub_product = CORP (TODO: double-check that C144A is not needed)
                mask = pc.and_(mask, pc.equal(trades['sub_product'], get_ordinal_value('sub_product', 'CORP')))
                # trade_original_format = finra_historical and trade_status = 'T'
                mask = pc.and_(mask, pc.if_else(
                                pc.equal(trades['trade_original_format'], get_ordinal_value('trade_original_format', 'finra_historical')),
                                    pc.equal(trades['trade_status'], get_ordinal_value('trade_status', 'T')),
                                    True))
                # trade_original_format = finra_eod and message_category = 'T' and message_type = 'M'
                mask = pc.and_(mask, pc.if_else(
                                pc.equal(trades['trade_original_format'], get_ordinal_value('trade_original_format', 'finra_eod')),
                                    pc.and_(pc.equal(trades['message_category'], get_ordinal_value('message_category', 'T')),
                                            pc.equal(trades['message_type'], get_ordinal_value('message_type', 'M'))),
                                    True))
                # filter for invalid execution date
                mask = pc.and_(mask, pc.not_equal(trades['execution_date'], 0))
                trades = trades.filter(mask)
            with profile('join cusip_to_rating'):
                unspecified_ordinal_value = ordinals['rating'].index(UNSPECIFIED_ORDINAL_VALUE).as_py()
                trades = ratings.join_with_cusip_to_rating(trades, unspecified_ordinal_value)
            with profile('join bonds'):
                trades = trades.join(bonds, 'cusip', 'cusip', 'inner', '', '_b')
                # only keep trades where the cusip and figi both match our reference data
                trades = trades.filter(pc.equal(trades['figi'], trades['figi_b']))
            with profile('sort_by'):
                trades = trades.sort_by([
                        ('report_date', 'ascending'),
                        ('record_num', 'ascending'),
                        ('figi', 'ascending'),
                        ('rating_date', 'ascending')])
            with profile('drop duplicates'):
                trades = drop_duplicates(trades, [
                                            'report_date',
                                            'record_num',
                                            'figi'
                                        ], keep='last')
            trades = trades.select(settings.get('$.data.trades.columns'))
            with profile('Replace null values with constant values'):
                # For each of the trades columns specified in the 'replace_nulls' setting, replace null values with
                # the specified constant value
                for column_name, value in settings.get('$.data.trades.replace_nulls').items():
                    column = trades[column_name]
                    # Compute a mask of null entries
                    mask = pc.is_null(column)
                    # Create a scalar value from the replacement
                    replacement = pa.scalar(value, type=column.type)
                    # Use the mask to replace null values
                    column = pc.if_else(mask, replacement, column)
                    # Replace the column in the table
                    trades = trades.set_column(trades.schema.get_field_index(column_name), column_name, column)
            with profile('filter out maturity > report_date, null values, and invalid labels'):
                mask = pc.greater(trades['maturity'], trades['report_date'])
                for column in trades.column_names:
                    mask = pc.and_(mask, pc.true_unless_null(trades[column]))
                for column_name in settings.get('$.data.trades.rfq_labels'):
                    column = trades[column_name]
                    mask = pc.and_(mask, pc.not_equal(column, pa.scalar(0.0, type=column.type)))
                trades = trades.filter(mask)
            with profile('Fix cases where report_date is before execution_date:'):
                if not trades.num_rows:
                    return trades
                min_date = pc.min_element_wise(trades['report_date'], trades['execution_date'])
                trades = trades.set_column(trades.schema.get_field_index('execution_date'), 'execution_date', min_date)
                # Assert that the execution date is always before or equal to the report date
                assert(pc.all(pc.less_equal(trades['execution_date'], trades['report_date'])).as_py())
            with profile('update quantity'):
                trades = trades.set_column(trades.column_names.index('quantity'), 'quantity', pc.cast(
                            pc.floor(
                                pc.if_else(
                                    pc.is_in(trades['rating'], pa.array(investment_grade_rating_ordinals)),
                                    pc.if_else(
                                        pc.greater(trades['quantity'], 5_000_000.0),
                                        5_000_000.0,
                                        trades['quantity']
                                    ),
                                    pc.if_else(
                                        pc.greater(trades['quantity'], 1_000_000.0),
                                        1_000_000.0,
                                        trades['quantity']
                                    )
                                )
                            ), pa.uint32()))
            if not settings.get('$.ratings_enabled'):
                with profile('Set ratings to zeros'):
                    # The index of the column you want to replace
                    column_index_to_replace = trades.column_names.index('rating')
                    # Create a new column with same length as original one filled with zeros
                    new_column = pa.nulls(trades.num_rows, trades['rating'].type).fill_null(0)
                    # Replace the required column with new column
                    trades = trades.set_column(column_index_to_replace, 'rating', new_column)
            return trades
    return _process_trades
create_process_trades = cache_result(_create_process_trades)

def get_initial_trades():
    '''Convenience function to get the sampled and processed trades from the data file'''
    return sample_trades(create_process_trades()(get_table('trades')))

def get_initial_quotes():
    '''Convenience function to get the processed quotes from the data file'''
    return create_process_quotes()(get_table('quotes'))
