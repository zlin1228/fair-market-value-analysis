from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from pyarrow import compute as pc

def drop_duplicates(table: pa.table, columns: list[str], keep: Literal['first', 'last']='first') -> pa.table:
    """Drops duplicate rows from a table and returns the new table

    Groups table by the column(s) provided and returns a new table consisting of only the first or last row of each group

    example:

    drop_duplicates(t, ['a','b'], keep='last')

    this will group table 't' by columns 'a' and 'b' and include only the last row of each group in the returned table

    Keyword arguments:
    table -- pyarrow table
    keep -- determines which row to keep when duplicate rows are found: 'first' keeps the first row, 'last' keeps the last row
    """
    # select the columns to group by
    indexed_table = table.select(columns)
    # add a numerical index to column selection
    indexed_table = indexed_table.add_column(
                    indexed_table.num_columns,
                    '_index',
                    pc.indices_nonzero(pc.if_else(True, True, pa.nulls(len(indexed_table), pa.bool_()))))
    # group by the selected columns and apply aggregation function to get the selected index from each group
    aggregation_function: Literal['max', 'min'] = 'max' if keep == 'last' else 'min'
    indexed_table = indexed_table.combine_chunks()
    selector = indexed_table.group_by(columns).aggregate([('_index', aggregation_function)])
    # filter table by whether the index for the row exists in the set of selected indexes
    return table.filter(pc.is_in(indexed_table['_index'], value_set=selector[f'_index_{aggregation_function}']))

def filter_by_value_in_column(table: pa.table, column: str, filter_table: pa.table, filter_columns: list[str], invert=False) -> pa.table:
    """Filters 'table' by keeping rows that have a value in 'column' that is found in any of the 'filter_columns' of 'filter_table'.

    Passing 'invert'=True will invert the selection, ie: keep rows that have a value in 'column' that is NOT found in any of the 'filter_columns' of 'filter_table'.

    examples:

    result = filter_by_value_found_in_column(trades, 'figi', cbonds_data, ['figi'])
    this will return only the rows in 'trades' that have a 'figi' value that is also found in the 'figi' column of cbonds_data

    result = filter_by_value_found_in_column(trades, 'cusip', cbonds_data, ['cusip_regs', 'cusip_144a'], invert=True)
    this will return only the rows in 'trades' that have a 'cusip' value that is NOT found in either the 'cusip_regs' or 'cusip_144a' columns of cbonds_data

    Keyword arguments:
    table -- pyarrow table
    column -- str
    filter_table -- pyarrow table
    filter_column -- str
    found -- bool
    """
    # retrieve all unique values from the filter_columns of the filter_table
    unique_value_arrays = []
    for filter_column in filter_columns:
        unique_value_arrays.append(filter_table[filter_column].unique())
    unique_values = pa.concat_arrays(unique_value_arrays).unique()
    # remove null
    unique_values = unique_values.filter(pc.true_unless_null(unique_values))
    # remove empty strings
    if unique_values.type == pa.string():
        unique_values = unique_values.filter(pc.invert(pc.match_substring_regex(unique_values, r'^\s*$')))
    # if there are still unique values left
    if len(unique_values):
        # create a table where each row has a unique value in '_unique_value_from_t2' and True in '_found_in_t2'
        right_table = pa.table({ '_unique_value_from_t2': unique_values, '_found_in_t2': pa.array([True] * len(unique_values), pa.bool_())})
        # now left outer join 'right_table' onto 'table' using 'column' == '_unique_value_from_t2'
        # this adds a '_found_in_t2' column to 'table'
        # for each row in 'table', the value of '_found_in_t2' will either be
        #   True if the value of 'column' was found in one of the 'filter_columns' of 'filter_table'
        #   otherwise it will be null
        result = table.join(right_table, column, right_keys='_unique_value_from_t2', join_type='left outer')
        # filter the result, keep only '_found_in_t2' == null if 'invert', otherwise keep only '_found_in_t2' == True
        result = result.filter(pc.is_null(result['_found_in_t2']) if invert else pc.equal(result['_found_in_t2'], True))
        # remove '_found_in_t2'
        result = result.remove_column(result.column_names.index('_found_in_t2'))
        # return the result
        return result
    # if there are no unique values in the 'filter_columns' of 'filter_table' then filter all rows from 'table'
    return table.slice(0,0)


def sample(table: pa.table, frac: float=1):
    """Filters 'table' by keeping a fraction ('frac') of evenly spaced rows.  Sampling is deterministic.

    examples:

    result = sample(trades, frac=0.1)
    this will return 10% of the rows in trades

    result = sample(trades, frac=0.95)
    this will return 95% of the rows in trades

    Keyword arguments:
    table -- pyarrow table
    frac -- float, fraction of table to keep
    """
    if frac < 0 or frac > 1:
        raise ValueError('frac must be a value between 0 and 1')
    # create an array the same length as table that contains contains incrementing values [0,1,2,...]
    filter = pc.indices_nonzero(pc.if_else(True, True, pa.nulls(len(table), pa.bool_())))
    # create a second array shifted +1 [1,2,3,...]
    shifted = pc.add(filter, 1)
    # create a filter by multiplying each value by the fraction and rounding up
    # and then comparing to find where the result is not the same between the first array
    # and the shifted array
    #
    # example: frac = 0.4
    #
    #   filter             [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,... ]
    #   shifted            [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,... ]
    #
    #   filter   x 0.4 rounded up    [  0,  0,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,... ]
    #   shifted  x 0.4 rounded up    [  0,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  6,... ]
    #
    #   filter                       [  F,  T,  F,  T,  F,  F,  T,  F,  T,  F,  F,  T,  F,  T,... ]
    #
    filter = pc.not_equal(pc.ceil(pc.multiply(filter, frac)), pc.ceil(pc.multiply(shifted, frac)))
    # filter the table and return the result
    return table.filter(filter)
