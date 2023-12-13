cimport cython
from cython.parallel  import prange
from libc.stdio       cimport *
from libcpp.vector    cimport vector

import numpy as np
import pyarrow as pa

cdef extern from "<sys/syscall.h>" nogil:
    int __NR_gettid
    long syscall(long number, ...)


cdef class GroupedRatings:
    '''
        In 'cusip_to_rating', there are three columns - 'cusip', 'rating', and 'rating_date'.
        We group the data by 'cusip', and save the ratings (sorted by 'rating_date').

        __rating_groups: array of rating groups (each rating group is a sorted array by 'rating_date')
    '''
    cdef vector[vector[vector[long]]] __rating_groups

    def __init__(self, table):
        self.__rating_groups.clear()
        # Sort the table
        table = table.sort_by([('cusip', 'ascending'),
                               ('rating_date', 'ascending'),
                               ('rating', 'ascending')])
        # Make sure the column order
        table = table.select(['cusip', 'rating_date', 'rating'])
        # Set rating groups
        self.__set_rating_groups(table.to_pandas().to_numpy(),
                                 table.column('cusip').to_pandas().max())

    '''
        Set a rating group.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __set_rating_group(self, long cusip, long[:, :] rating_array) nogil:
        cdef long rows = rating_array.shape[0]
        cdef long cols = rating_array.shape[1]
        cdef long i, j
        self.__rating_groups[cusip].resize(rows)
        for i in prange(rows, schedule='static', nogil=True):
            self.__rating_groups[cusip][i].resize(cols)
            for j in prange(cols):
                self.__rating_groups[cusip][i][j] = rating_array[i, j]

    '''
        'cusip_to_rating' is sorted by 'cusip'.
        So each rating group is a set of contiguous rows.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __set_rating_groups(self, long[:, :] cusip_to_rating, long maximum_cusip) nogil:
        cdef long start_row_number = 0
        cdef long end_row_number
        cdef long size = cusip_to_rating.shape[0]
        cdef long current_cusip, index_of_cusip = 0
        # Allocate memory
        self.__rating_groups.resize(maximum_cusip + 1)
        # Group by 'cusip'
        while start_row_number < size:
            # cusip value of the current rating group
            current_cusip = cusip_to_rating[start_row_number, index_of_cusip]
            # Find the end of the current rating group
            end_row_number = start_row_number + 1
            while end_row_number < size and cusip_to_rating[end_row_number, index_of_cusip] == current_cusip:
                end_row_number += 1
            # Set the rating group
            self.__set_rating_group(current_cusip, cusip_to_rating[start_row_number:end_row_number, index_of_cusip+1:])
            # Move start_row_number
            start_row_number = end_row_number

    '''
    We have the 'cusip_to_rating' table that contains cusips, rating_dates, and ratings.
    When a cusip and an execution_date are given, we need to find the most recent rating in the table.
    This function returns the index of the row -- the most recent rating.
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef long __get_index_of_most_recent_rating(self, long cusip, long execution_date) nogil:
        # Check for impossible cases and return -1
        if cusip >= <long>self.__rating_groups.size() or <long>self.__rating_groups[cusip].size() == 0:
            return -1
        cdef long index_of_rating_date = 0  # rating_date: first column
        cdef long st, en, md, res = -1
        st, en = 0, <long>self.__rating_groups[cusip].size() - 1
        # Use binary search to find the most recent rating not later than execution_date
        while st <= en:
            md = (st + en) // 2
            if self.__rating_groups[cusip][md][index_of_rating_date] <= execution_date:
                res = md
                st = md + 1
            else:
                en = md - 1
        return res

    '''
    For each pair (cusip, execution_date), get the most recent rating to set 'rating_dates' and 'ratings'.
    If there is no rating, set 'rating_dates' to 'execution_date' + 1 (so we can filter)
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef void __get_most_recent_ratings(self, long[:] cusips, long[:] execution_dates, long[:] rating_dates, long[:] ratings, long unspecified_ordinal_value) nogil:
        cdef long i, j, cusip
        cdef long index_of_rating_date = 0
        cdef long index_of_rating = 1
        cdef long table_size = cusips.shape[0]
        for i in prange(table_size, schedule='static', nogil=True):
            cusip = cusips[i]
            # Get the most recent index
            j = self.__get_index_of_most_recent_rating(cusip, execution_dates[i])
            if j >= 0:
                rating_dates[i] = self.__rating_groups[cusip][j][index_of_rating_date]
                ratings[i] = self.__rating_groups[cusip][j][index_of_rating]
            else:
                rating_dates[i] = execution_dates[i]  # We can filter
                ratings[i] = unspecified_ordinal_value

    '''
    This function returns a new pyarrow table.
    We append 'rating_date' and 'rating' columns to the original table to get the new table.
    '''
    def join_with_cusip_to_rating(self, table: pa.table, unspecified_ordinal_value: int) -> pa.table:
        table_size = len(table)
        cusips = table.column('cusip').to_numpy().astype(int).copy()
        execution_dates = table.column('execution_date').to_numpy().astype(int).copy()
        rating_dates = np.zeros(table_size, dtype=int)
        ratings = np.zeros(table_size, dtype=int)
        # Get rating_dates and ratings
        self.get_most_recent_ratings(cusips, execution_dates, rating_dates, ratings, unspecified_ordinal_value)
        # Append columns
        table = table.append_column('rating_date', pa.array(rating_dates))
        table = table.append_column('rating', pa.array(ratings).cast(pa.int32()))  # rating dtype: int32
        return table

    def get_index_of_most_recent_rating(self, long cusip, long execution_date):
        return self.__get_index_of_most_recent_rating(cusip, execution_date)

    def get_most_recent_ratings(self, long[:] cusips, long[:] execution_dates, long[:] rating_dates, long[:] ratings, long unspecified_ordinal_value):
        self.__get_most_recent_ratings(cusips, execution_dates, rating_dates, ratings, unspecified_ordinal_value)