cimport cython
from cython.parallel  import prange
from libc.stdio       cimport *
from libcpp.vector    cimport vector

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

from helpers import get_table_from_array
from performance_helper import create_profiler

cdef extern from "<sys/syscall.h>" nogil:
    int __NR_gettid
    long syscall(long number, ...)


def combine_trace(trace: pa.Table) -> pa.Table:
    # Convert PyArrow table to pandas DataFrame
    df = trace.to_pandas()

    # Prepare the list of column names excluding 'report_date' and 'quantity'
    trace_columns_quantity_removed = [c for c in df.columns if c not in ('report_date', 'quantity')]

    # Group by the columns and aggregate 'quantity' and 'report_date'
    df_combined = df.groupby(trace_columns_quantity_removed).agg({'quantity': 'sum', 'report_date': 'max'}).reset_index()

    # Cast 'quantity' to uint32 as required
    df_combined['quantity'] = df_combined['quantity'].astype('uint32')

    # Restore the original order of columns
    df_combined = df_combined[trace.schema.names]

    # Convert back to PyArrow Table and return
    return_value = pa.Table.from_pandas(df_combined, preserve_index=False)
    return return_value.select(trace.column_names)


'''

                                                                                    -------------------------------------
                                                                                    |            GroupedHistory         |
                                                                                    |                                   |
                                                                                    |  store trades                     |
                                                                                    |  share with GroupedEntry classes  |
                                                                                    -------------------------------------

                                                                                                       |
                                                                                                       |
                                                                                                       |
                   -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                   |                                         |                                         |                                         |                                         |
                   |                                         |                                         |                                         |                                         |

--------------------------------------    --------------------------------------    --------------------------------------    --------------------------------------    --------------------------------------
|        GroupedEntryByFigi          |    |          GroupedEntryByIssuer      |    |        GroupedEntryByIndustry      |    |         GroupedEntryBySector       |    |         GroupedEntryByRating       |
|                                    |    |                                    |    |                                    |    |                                    |    |                                    |
|  only save entries                 |    |  only save entries                 |    |  only save entries                 |    |  only save entries                 |    |  only save entries                 |
|  group by figi value               |    |  group by issuer value             |    |  group by industry value           |    |  group by sector value             |    |  group by rating value             |
|  look up trades in GroupedHistory  |    |  look up trades in GroupedHistory  |    |  look up trades in GroupedHistory  |    |  look up trades in GroupedHistory  |    |  look up trades in GroupedHistory  |
--------------------------------------    --------------------------------------    --------------------------------------    --------------------------------------    --------------------------------------

entry: the row number in the 'trades' table

'''


cdef class GroupedEntry:
    '''
    We have feature values (figi, price, report_date, ...) in GroupedHistory.
    We don't need to save them again in GroupedEntry.
    We only save entries of the trades.
    We can look up feature values of trades by using these entries.

    entry_groups          : entry_group array
                            an entry_group is a list of entries
                            entry_groups is a 2D array (it has group-value layer)
    group_value_of_entry  : map entries to group values (figi, issuer, ...)
                            sometimes we remove entries from entry_groups
                            we need to make sure which entry_group contains the entry we want to remove
                            to look up entry_group, we use group_value_of_entry
    '''
    cdef vector[vector[long]] entry_groups
    cdef vector[long] group_value_of_entry

    def __init__(self):
        self.entry_groups.clear()
        self.group_value_of_entry.clear()

    '''
    This function appends entries of trades to the end of an entry_group.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __append_entries_to_end_of_group(self, vector[long] &entry_group, long[:] entries) nogil:
        cdef long group_size = <long>entry_group.size(), i
        entry_group.resize(group_size + entries.shape[0])
        for i in prange(entries.shape[0], schedule='static', nogil=True):
            entry_group[group_size + i] = entries[i]

    '''
    This function compares two double values.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef long __compare_values(self, double x, double y) nogil:
        if x < y:
            return -1
        elif x > y:
            return 1
        else:
            return 0

    '''
    This function compares two trades.
    (report_date first, then other columns)
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef long __compare_trades(self, vector[double] &trade_1, vector[double] &trade_2, long feature_of_report_date) nogil:
        cdef long sign = self.__compare_values(trade_1[feature_of_report_date], trade_2[feature_of_report_date]), i
        if sign == 0:
            for i in range(<long>trade_1.size()):
                sign = self.__compare_values(trade_1[i], trade_2[i])
                if sign != 0:
                    break
        return sign

    '''
    This function merges 'entries' into 'entry_group'.
    We share 'feature_values' with GroupedHistory class.
    So, we can compare two entries (because we use two entries to look up two trades).
    We can compare two trades by the above '__compare_trades' function.

    'entry_group' and 'entries' are both sorted by report_date and other columns.
    But, we must reorder some entries to merge 'entries' into 'entry_group'.

    'entry_group[first_index_in_group]' is the first entry in 'entry_group'
    which is later the first entry in 'entries'.
    It means that we don't need to reorder 'entry_group[:first_index_in_group]'.
    We can merge 'entry_group[first_index_in_group:]' and 'entries',
    then append the merged entries to the end of 'entry_group[:first_index_in_group]'.

    We use merge-sort to merge 'entry_group[first_index_in_group:]' and 'entries'.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __merge_entries_into_group(self, vector[long] &entry_group, long first_index_in_group, long[:] entries, vector[vector[double]] &feature_values, long feature_of_report_date) nogil:
        cdef long i, group_size = <long>entry_group.size()
        cdef long index_in_group = first_index_in_group                                     # only 'entry_group[first_index_in_group:]'
        cdef long index_in_entries = 0                                                      # whole 'entries'
        cdef vector[long] merged_entries
        merged_entries.resize(group_size - first_index_in_group + entries.shape[0])         # the size of 'merged_entries'
        for i in range(<long>merged_entries.size()):
            if index_in_group < group_size and index_in_entries < entries.shape[0]:         # compare 'entry_group[index_in_group]' and 'entries[index_in_entries]'
                if self.__compare_trades(feature_values[entry_group[index_in_group]],
                                         feature_values[entries[index_in_entries]],
                                         feature_of_report_date) < 0:                       # append index_in_group
                    merged_entries[i] = entry_group[index_in_group]
                    index_in_group += 1
                else:                                                                       # append index_in_entries
                    merged_entries[i] = entries[index_in_entries]
                    index_in_entries += 1
            elif index_in_group < group_size:                                               # append index_in_group
                merged_entries[i] = entry_group[index_in_group]
                index_in_group += 1
            else:                                                                           # append index_in_entries
                merged_entries[i] = entries[index_in_entries]
                index_in_entries += 1
        entry_group.resize(group_size + entries.shape[0])                                   # append 'merged_trades' to 'entry_group[:first_entry_in_group]'
        for i in range(<long>merged_entries.size()):
            entry_group[first_index_in_group + i] = merged_entries[i]

    '''
    This function appends 'entries' to 'entry_group'.
    We use 'feature_values' and 'feature_of_report_date' to compare entries. (look at the above function)

    Some entries in 'entry_group' are later the first entry in 'entries'. (define as 'entries_1')
    Some entries in 'entries' are before the last entry in 'entry_group'. (define as 'entries_2')
    'entries_1' should be 'entry_group[index:]'. (Because 'entry_group' is sorted.)
    'entries_2' should be 'entries[:index]'. (Because 'entries' is sorted.)

    First of all, we make sure 'entries_1' and 'entries_2'.
    To do this, we find 'first_index_in_group' and 'last_index_in_entries'.
    'entries_1': entry_group[first_index_in_group:]
    'entries_2': entries[:last_index_in_group + 1]

    Then, we use merge-sort to reorder.
    Finally, we append the merged entries to 'entry_group'.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __append_entries_to_group(self, vector[long] &entry_group, long[:] entries, vector[vector[double]] &feature_values, long feature_of_report_date) nogil:
        cdef long first_index_in_group = <long>entry_group.size()
        cdef long last_index_in_entries = -1
        if entry_group.size() > 0:                                                                    # we don't reorder if 'entry_group' is empty
            while first_index_in_group > 0:                                                           # iterate 'first_index_in_group' from the end
                if self.__compare_trades(feature_values[entry_group[first_index_in_group - 1]],
                                         feature_values[entries[0]],
                                         feature_of_report_date) > 0:
                    first_index_in_group -= 1
                else:
                    break
            while last_index_in_entries + 1 < entries.shape[0]:                                       # iterate 'last_index_in_entries' from the beginning
                if self.__compare_trades(feature_values[entries[last_index_in_entries + 1]],
                                         feature_values[entry_group[<long>entry_group.size() - 1]],
                                         feature_of_report_date) < 0:
                    last_index_in_entries += 1
                else:
                    break
        if first_index_in_group < <long>entry_group.size() and last_index_in_entries >= 0:            # if there are some entries we should reorder
            self.__merge_entries_into_group(entry_group, first_index_in_group, entries[:last_index_in_entries + 1], feature_values, feature_of_report_date)
            self.__append_entries_to_end_of_group(entry_group, entries[last_index_in_entries + 1:])
        else:
            self.__append_entries_to_end_of_group(entry_group, entries)                               # else, we can append to the end

    '''
    'group_values' is sorted in ascending order.
    So, each group consists of contiguous indices.
    We can find the end of the group which starts from 'start' using binary-search.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef long __get_the_end_of_group(self, long start, long[:] group_values) nogil:
        cdef long group_value = group_values[start]               # group-value of the group
        cdef long st = start, en = group_values.shape[0] - 1, md  # use binary-search
        while en - st > 1:
            md = (st + en) // 2
            if group_values[md] == group_value:
                st = md
            else:
                en = md
        if group_values[en] == group_value:                       # edge case
            st = en
        return st

    '''
    This function updates 'group_value_of_entry'.
    We map entries to the corresponding group values.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __update_group_value_of_entry(self, long[:] entries, long[:] group_values) nogil:
        cdef long i, max_entry = -1
        for i in range(entries.shape[0]):
            if max_entry < entries[i]:
                max_entry = entries[i]
        if max_entry >= <long>self.group_value_of_entry.size():     # make sure the maximum size
            self.group_value_of_entry.resize(max_entry + 1)
        for i in prange(group_values.shape[0], schedule='static', nogil=True):
            self.group_value_of_entry[entries[i]] = group_values[i]

    '''
    This function appends entries by group-value.
    We use 'feature_values' and 'feature_of_report_date' to compare trades.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __append_entries(self, long[:] group_values, long[:] entries, vector[vector[double]] &feature_values, long feature_of_report_date) nogil:
        cdef int current_position = 0, last_position_of_group, group_value                         # start from the beginning
        while current_position < group_values.shape[0]:
            last_position_of_group = self.__get_the_end_of_group(current_position, group_values)   # get the end of the group
            group_value = group_values[current_position]                                           # group-value of the current group
            if group_value >= <long>self.entry_groups.size():
                self.entry_groups.resize(group_value + 1)
            self.__append_entries_to_group(self.entry_groups[group_value],                         # append entries to the group
                                           entries[current_position:last_position_of_group+1],
                                           feature_values,
                                           feature_of_report_date)
            current_position = last_position_of_group + 1                                          # move to the next group
        self.__update_group_value_of_entry(entries, group_values)                                  # update group_value_of_entry

    def append_entries(self, long[:] group_values, long[:] entries, TradeFeatureArray feature_array, long feature_of_report_date):
        self.__append_entries(group_values, entries, feature_array.feature_values, feature_of_report_date)

    '''
    This function removes an entry from 'entry_groups'.
    First, we get the group value of the entry using 'group_value_of_entry'.
    Then, we remove an entry from 'entry_groups[group_value]'.
    The entry should be near the end of the entry-group.
    So we iterate from the end.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __remove_entry_from_group(self, long entry) nogil:
        cdef long group_value = self.group_value_of_entry[entry]
        cdef long index = self.entry_groups[group_value].size() - 1
        while self.entry_groups[group_value][index] != entry:
            index -= 1
        self.entry_groups[group_value].erase(self.entry_groups[group_value].begin() + index)

    '''
    This function removes entries in [first_entry:last_entry+1].
    We can use the above function.
    '''
    def remove_entries(self, long first_entry, long last_entry):
        for entry in range(last_entry - 1, first_entry - 1, -1):
            self.__remove_entry_from_group(entry)

    '''
    This function returns the index of the most recent entry in a group right before the execution date.
    Entries in a group are sorted. So we can use binary-search.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef long __get_index_of_most_recent_entry_in_group(self, vector[long] &entry_group, double execution_date, vector[vector[double]] &feature_values, long feature_of_report_date) nogil:
        if entry_group.empty():
            return -1                                                                     # return -1 for an empty group
        cdef long st = 0
        cdef long en = <long>entry_group.size() - 1
        while en - st > 1:                                                                # use binary search
            md = (st + en) // 2
            if feature_values[entry_group[md]][feature_of_report_date] < execution_date:
                st = md
            else:
                en = md
        if feature_values[entry_group[en]][feature_of_report_date] < execution_date:      # edge case
            st = en
        if feature_values[entry_group[st]][feature_of_report_date] >= execution_date:     # edge case
            return -1
        return st

    '''
    We set rows of batches using this function.
    We use most recent feature values to set 'X_b'.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __set_rows(self, double[:, :] X_b, double[:] group_values, double[:] execution_dates, long sequence_length, vector[vector[double]] &feature_values, vector[long] &trade_feature_indices, long feature_of_report_date) nogil:
        cdef long batch_size = X_b.shape[0]
        cdef long i, j, k, l, index, group_value, previous_index
        cdef long trade_feature_count = trade_feature_indices.size()
        for i in prange(batch_size, schedule='static', nogil=True):
            group_value = <int>group_values[i]
            # if entry_groups doesn't contain group_value, continue
            if group_value >= <int>self.entry_groups.size():
                continue
            # get the previous index before the execution date
            previous_index = self.__get_index_of_most_recent_entry_in_group(self.entry_groups[group_value], execution_dates[i], feature_values, feature_of_report_date)
            # set rows
            for j in prange(sequence_length):
                index = previous_index - j
                if index < 0:
                    continue
                l = (sequence_length - j - 1) * trade_feature_count
                for k in prange(trade_feature_count):
                    X_b[i, l + k] = feature_values[self.entry_groups[group_value][index]][trade_feature_indices[k]]

    def set_rows(self, double[:, :] X_b, double[:] group_values, double[:] execution_dates, long sequence_length, TradeFeatureArray feature_array, long feature_of_report_date):
        self.__set_rows(X_b, group_values, execution_dates, sequence_length, feature_array.feature_values, feature_array.trade_feature_indices, feature_of_report_date)

    '''
    All entries are sorted by report_date.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __get_last_entries(self, vector[long] &entry_group, long entry_count, long[:] last_entries) nogil:
        cdef long i
        cdef long starting_index = <long>entry_group.size() - entry_count
        for i in prange(entry_count, schedule='static', nogil=True):
            last_entries[i] = entry_group[starting_index + i]

    '''
    This function returns a number of last entries for the given figi value.
    We only use this function in terms of figi - GroupedEntry.
    '''
    def get_last_entries(self, long figi_value, long trade_count):
        if figi_value >= <long>self.entry_groups.size():
            return np.array([])
        entry_count = min(trade_count, <long>self.entry_groups[figi_value].size())
        last_entries = np.zeros(entry_count).astype(int)
        self.__get_last_entries(self.entry_groups[figi_value], entry_count, last_entries)
        return last_entries


'''
Instead of numpy arrays, we use TradeFeatureArray data structure to save all feature values of trades.
We use a 2D C++ vector to save all feature values.
It's comfortable for appending and resizing.
'''
cdef class TradeFeatureArray:
    cdef long __feature_count
    cdef vector[long] trade_feature_indices
    cdef vector[vector[double]] feature_values

    def __init__(self, feature_count, trade_feature_indices):
        self.__feature_count = feature_count
        self.cleanup()
        for trade_feature_index in trade_feature_indices:
            self.trade_feature_indices.push_back(trade_feature_index)

    def cleanup(self):
        self.trade_feature_indices.clear()
        self.feature_values.clear()

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __append(self, double[:, :] feature_array):
        cdef long i, j
        cdef long previous_size = self.feature_values.size()
        self.feature_values.resize(previous_size + feature_array.shape[0])
        for i in prange(feature_array.shape[0], schedule='static', nogil=True):
            self.feature_values[previous_size + i].resize(self.__feature_count)
            for j in prange(self.__feature_count):
                self.feature_values[previous_size + i][j] = feature_array[i, j]

    def append(self, double[:, :] feature_array):
        self.__append(feature_array)

    def resize(self, size):
        self.feature_values.resize(size)

    '''
    This function is for unit testing.
    It returns feature values as a 2D numpy array.
    '''
    def get_feature_array(self):
        rows, cols = self.feature_values.size(), self.feature_values[0].size()
        feature_array = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                feature_array[i, j] = self.feature_values[i][j]
        return feature_array

    '''
    Get trade feature values.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __get_trade_features(self, long[:] entries, double[:, :] trade_features_array) nogil:
        cdef long i, j
        for i in prange(entries.shape[0], schedule='static', nogil=True):
            for j in prange(self.__feature_count):
                trade_features_array[i, j] = self.feature_values[entries[i]][j]

    '''
    Select rows in the whole feature table using entry numbers.
    This function returns a numpy array.
    '''
    def get_trade_features(self, long[:] entries):
        trade_features_array = np.zeros((entries.shape[0], self.__feature_count))
        self.__get_trade_features(entries, trade_features_array)
        return trade_features_array


class GroupedHistory:
    '''
    This class saves distinct trades. (no duplicate)
    It has 'GroupedEntry's by each group in '__groups'.
    And, it generates batches for model inputs.

    '__trades'       : a pyarrow table that contains all distinct trades (sorted by execution_date column)
    '__feature_array': TradeFeatureArray -- Data Structure to save all features in trades
    '__grouped_entry': a GroupedEntry class for a group ('figi', 'issuer', 'industry', 'sector', and 'rating')
    '''
    def __init__(self, groups, sequence_length, rfq_features, trade_features, rfq_labels):
        self.__trades = None
        self.__trade_feature_array = None
        self.__sequence_length = sequence_length
        self.__rfq_features = rfq_features
        self.__trade_features = trade_features
        self.__rfq_labels = rfq_labels
        self.__init_grouped_entry(groups)

    def __init_grouped_entry(self, groups):
        self.__groups = groups
        self.__grouped_entry = {}
        for group in self.__groups:
            self.__grouped_entry[group] = GroupedEntry()

    def __init_column_indices(self, columns):
        self.__group_indices = {group: columns.index(group) for group in self.__groups}
        self.__rfq_feature_indices = [columns.index(col) for col in self.__rfq_features]
        self.__trade_feature_indices = [columns.index(col) for col in self.__trade_features]
        self.__rfq_label_indices = [columns.index(col) for col in self.__rfq_labels]
        self.__report_date_index = columns.index('report_date')
        self.__trade_feature_array = TradeFeatureArray(len(columns), self.__trade_feature_indices)

    def cleanup(self):
        del self.__trades
        self.__trade_feature_array.cleanup()
        del self.__trade_feature_array
        for group in self.__groups:
            self.__grouped_entry[group].__init__()
            del self.__grouped_entry[group]
        del self.__grouped_entry
        del self.__groups
        del self.__sequence_length
        del self.__rfq_features
        del self.__trade_features
        del self.__rfq_labels
        del self.__group_indices
        del self.__rfq_feature_indices
        del self.__trade_feature_indices
        del self.__rfq_label_indices

    def get_trace(self):
        return self.__trades

    '''
    This function returns the most recent trades which has the given figi_value.
    '''
    def get_last_figi_trades(self, figi_value, trade_count):
        assert('figi' in self.__groups)
        entries = self.__grouped_entry['figi'].get_last_entries(figi_value, trade_count)
        figi_trace_array = self.__trade_feature_array.get_trade_features(entries)
        figi_trace_table = get_table_from_array(figi_trace_array, self.__trades.schema)
        return figi_trace_table

    '''
    '__trades' is sorted by execution_date.
    We find the entry of the first trade which is after the given 'execution_date'
    by using binary search. If there is no trade, returns None.
    '''
    def __get_entry_of_next_trade_by_execution_date(self, execution_date):
        st, en = 0, self.__trades.num_rows - 1
        while en - st > 1:
            md = (st + en) // 2
            if self.__trades['execution_date'][md].as_py() >= execution_date:
                en = md
            else:
                st = md
        if self.__trades['execution_date'][st].as_py() >= execution_date:  # edge case
            en = st
        if self.__trades['execution_date'][en].as_py() >= execution_date:  # there is the first trade
            return en
        else:                                                              # no trade after 'execution_date'
            return None

    '''
    This function sorts a pyarrow table by column names.
    ('group', 'report_date' column first, then other columns)
    '''
    def __sort_by_report_date_and_group_value(self, trades: pa.Table, group: str) -> pa.Table:
        column_sort_list = trades.column_names
        column_sort_list.remove('report_date')
        column_sort_list.insert(0, 'report_date')
        column_sort_list.remove(group)
        column_sort_list.insert(0, group)
        return trades.sort_by([(column, 'ascending') for column in column_sort_list])

    '''
    This function appends new trades.
    It updates '__trades' and '__feature_array'.
    Then, appends sorted pyarrow tables by report_date with 'entry' column to GroupedEntries.
    '''
    def __append_new_trades_to_groups(self, new_trades: pa.Table):
        profile = create_profiler('GroupedHistory.__append_new_trades_to_groups')
        # update '__trades'
        if self.__trades is None:
            self.__trades = new_trades
            with profile('init column indices'):
                self.__init_column_indices(new_trades.column_names)
        else:
            with profile('concat new trades'):
                self.__trades = pa.concat_tables([self.__trades, new_trades])
        # append new trades to '__trade_feature_array'
        with profile('add new trades to feature array'):
            self.__trade_feature_array.append(new_trades.to_pandas().to_numpy())
        # append 'entry' column
        with profile('append the entry column'):
            new_trades = new_trades.append_column('entry', pa.array(np.arange(self.__trades.num_rows - new_trades.num_rows, self.__trades.num_rows)))
        for group in self.__groups:
            with profile('sort trades by report_date and group_value'):
                new_trades = self.__sort_by_report_date_and_group_value(new_trades, group)  # sort table by 'report_date'
            with profile(f'append new trades to {group}'):
                self.__grouped_entry[group].append_entries(new_trades.column(group).to_numpy().astype(int),
                                                           new_trades.column('entry').to_numpy().astype(int),
                                                           self.__trade_feature_array,
                                                           new_trades.column_names.index('report_date'))

    '''
    This function removes all entries which are greater than or equal to 'previous_entry'.
    '''
    def __remove_entries_from_groups(self, first_entry, last_entry):
        profile = create_profiler('GroupedHistory.__remove_entries_from_groups')
        for group in self.__groups:
            with profile(f'remove entries from {group}'):
                self.__grouped_entry[group].remove_entries(first_entry, last_entry)

    '''
    This function appends new trades.

    we get the finra data on a UDP multicast
    finra broadcasts messages out using the UDP protocol
    all routers and switches between finra and us are responsible for forwarding the packets
    there is no guarantee of order or even delivery with UDP
    (there actually isn't a guarantee of delivery)
    so finra sends out the packets and hopefully they get to us

    if one packet gets sent along a faster route
    and the next gets delayed along a slower route
    they won't arrive in order
    (one or the other might not even arrive)

    Sometimes, we need to reorder trades to keep a sorted table by 'execution_date'.
    We only change a few trades at the end of '__trades'.
    We keep most trades, concat trades in '__trades' which should be reordered with new trades,
    then append sorted new trades by execution_date.
    '''
    def append_trades(self, new_trades: pa.Table):
        profile = create_profiler('GroupedHistory.append_trades')
        if self.__trades is not None:
            with profile('reorder trades'):
                # get the first execution_date in new trades
                first_execution_date = pc.min(new_trades.column('execution_date')).as_py()
                # we can keep existing trades which are before the 'first_execution_date'
                next_entry = self.__get_entry_of_next_trade_by_execution_date(first_execution_date)
                # if there are some trades we should reorder ...
                if next_entry is not None:
                    # remove all trades we should reorder
                    self.__remove_entries_from_groups(next_entry, len(self.__trades))
                    # update new_trades
                    new_trades = pa.concat_tables([new_trades, self.__trades.slice(next_entry)])
                    # update '__trades', '__trade_feature_array'
                    self.__trades = self.__trades.slice(0, next_entry)
                    self.__trade_feature_array.resize(next_entry)
        # remove duplicates from new_trades
        with profile('combine new trades'):
            new_trades = combine_trace(new_trades)
        # sort by 'execution_date'
        with profile('sort new trades by execution_date'):
            new_trades = new_trades.sort_by([('execution_date', 'ascending')])
        # append new trades
        with profile('append new trades to groups'):
            self.__append_new_trades_to_groups(new_trades)

    '''
    This function generates batch from Z_b.
    '''
    def generate_batch(self, Z_b: np.ndarray) -> np.ndarray:
        Y_b = Z_b[:, self.__rfq_label_indices]
        batch_size = Z_b.shape[0]
        sequences_size = self.__sequence_length * len(self.__trade_features)
        X_b = np.zeros((batch_size, len(self.__rfq_features) + len(self.__groups) * sequences_size))
        # Set this rfq's features at the beginning:
        X_b[:, :len(self.__rfq_features)] = Z_b[:, self.__rfq_feature_indices]
        for i, group in enumerate(self.__groups):
            group_values = Z_b[:, self.__group_indices[group]]
            execution_dates = Z_b[:, self.__trades.column_names.index('execution_date')]
            l = len(self.__rfq_features) + i * sequences_size
            r = l + sequences_size
            self.__grouped_entry[group].set_rows(X_b[:, l:r], group_values, execution_dates, self.__sequence_length, self.__trade_feature_array, self.__report_date_index)
        return X_b, Y_b


cdef class GroupedEntryTest(GroupedEntry):
    '''
    This class is for unit testing.
    '''
    def __init__(self):
        super().__init__()

    '''
    This function appends entries_2 to the end of entries_1,
    and returns the merged numpy array.
    '''
    def append_entries_to_end_of_group(self, long[:] entries_1, long[:] entries_2):
        cdef vector[long] group
        cdef long i
        group.resize(entries_1.shape[0])
        for i in range(entries_1.shape[0]):
            group[i] = entries_1[i]
        self.__append_entries_to_end_of_group(group, entries_2)
        entries = np.zeros(entries_1.shape[0] + entries_2.shape[0]).astype(int)
        for i in range(entries.shape[0]):
            entries[i] = group[i]
        return entries

    def compare_values(self, double x, double y):
        return self.__compare_values(x, y)

    def compare_trades(self, double[:] trade_1, double[:] trade_2, long feature_of_report_date):
        cdef vector[double] array_1, array_2
        array_1.resize(trade_1.shape[0])
        array_2.resize(trade_2.shape[0])
        for i in range(trade_1.shape[0]):
            array_1[i] = trade_1[i]
            array_2[i] = trade_2[i]
        return self.__compare_trades(array_1, array_2, feature_of_report_date)

    '''
    This function uses merge-sort to combine trades_1 and trades_2.
    It returns the sorted trades after merging.
    '''
    def merge_entries_into_group(self, double[:, :] trades_1, double[:, :] trades_2):
        cdef vector[long] group
        group.resize(trades_1.shape[0])
        for i in range(<long>group.size()):
            group[i] = i
        first_index_in_group = 0
        entries = np.zeros(trades_2.shape[0]).astype(int)
        for i in range(entries.shape[0]):
            entries[i] = trades_1.shape[0] + i
        cdef vector[vector[double]] feature_array
        feature_array.resize(trades_1.shape[0] + trades_2.shape[0])
        for i in range(trades_1.shape[0]):
            feature_array[i].resize(trades_1.shape[1])
            for j in range(trades_1.shape[1]):
                feature_array[i][j] = trades_1[i, j]
        for i in range(trades_2.shape[0]):
            feature_array[trades_1.shape[0] + i].resize(trades_2.shape[1])
            for j in range(trades_2.shape[1]):
                feature_array[trades_1.shape[0] + i][j] = trades_2[i, j]
        feature_of_report_date = 0
        self.__merge_entries_into_group(group, first_index_in_group, entries, feature_array, feature_of_report_date)
        feature_values = np.concatenate((trades_1, trades_2), axis=0)
        indices = np.zeros(group.size()).astype(int)
        for i in range(group.size()):
            indices[i] = group[i]
        return feature_values[indices, :]

    '''
    This function appends trades_2 to trades_1.
    It returns the merged trades.
    '''
    def append_entries_to_group(self, double[:, :] trades_1, double[:, :] trades_2):
        cdef vector[long] group
        group.resize(trades_1.shape[0])
        for i in range(<long>group.size()):
            group[i] = i
        entries = np.zeros(trades_2.shape[0]).astype(int)
        for i in range(entries.shape[0]):
            entries[i] = trades_1.shape[0] + i
        cdef vector[vector[double]] feature_array
        feature_array.resize(trades_1.shape[0] + trades_2.shape[0])
        for i in range(trades_1.shape[0]):
            feature_array[i].resize(trades_1.shape[1])
            for j in range(trades_1.shape[1]):
                feature_array[i][j] = trades_1[i, j]
        for i in range(trades_2.shape[0]):
            feature_array[trades_1.shape[0] + i].resize(trades_2.shape[1])
            for j in range(trades_2.shape[1]):
                feature_array[trades_1.shape[0] + i][j] = trades_2[i, j]
        feature_of_report_date = 0
        self.__append_entries_to_group(group, entries, feature_array, feature_of_report_date)
        feature_values = np.concatenate((trades_1, trades_2), axis=0)
        indices = np.zeros(group.size()).astype(int)
        for i in range(<long>group.size()):
            indices[i] = group[i]
        return feature_values[indices, :]

    def get_the_end_of_group(self, long start, long[:] entries):
        return self.__get_the_end_of_group(start, entries)

    '''
    This function returns 'group_value_of_entry' as a numpy array.
    '''
    def update_group_value_of_entry(self, long[:] entries, long[:] group_values):
        self.__update_group_value_of_entry(entries, group_values)
        group_value_of_entry = np.zeros(self.group_value_of_entry.size()).astype(int)
        for i in range(group_value_of_entry.shape[0]):
            group_value_of_entry[i] = self.group_value_of_entry[i]
        return group_value_of_entry

    '''
    This function returns the index of most recent entry before 'execution_date'.
    '''
    def get_index_of_most_recent_entry_in_group(self, double[:, :] trades, double execution_date):
        feature_of_report_date = 0
        cdef vector[long] entry_group
        entry_group.resize(trades.shape[0])
        for i in range(trades.shape[0]):
            entry_group[i] = i
        cdef vector[vector[double]] feature_values
        feature_values.resize(trades.shape[0])
        for i in range(trades.shape[0]):
            feature_values[i].resize(trades.shape[1])
            for j in range(trades.shape[1]):
                feature_values[i][j] = trades[i, j]
        return self.__get_index_of_most_recent_entry_in_group(entry_group, execution_date, feature_values, feature_of_report_date)

    def get_entry_groups(self):
        entry_groups = np.array([]).astype(int)
        for i in range(self.entry_groups.size()):
            if self.entry_groups[i].empty():
                continue
            temp = np.zeros(self.entry_groups[i].size()).astype(int)
            for j in range(self.entry_groups[i].size()):
                temp[j] = self.entry_groups[i][j]
            entry_groups = np.append(entry_groups, temp)
        return entry_groups


class GroupedHistoryTest(GroupedHistory):
    def __init__(self, groups, sequence_length, rfq_features, trade_features, rfq_labels):
        GroupedHistory.__init__(self, groups, sequence_length, rfq_features, trade_features, rfq_labels)
        self.__update_grouped_entry()

    def __update_grouped_entry(self):
        for group in self._GroupedHistory__groups:
            self._GroupedHistory__grouped_entry[group] = GroupedEntryTest()
