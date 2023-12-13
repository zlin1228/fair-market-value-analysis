cimport cython
from cython.parallel  import prange
from libc.stdio       cimport *
from libcpp.vector    cimport vector
from libcpp.algorithm cimport sort
import numpy as np

cdef extern from "<sys/syscall.h>" nogil:
    int __NR_gettid
    long syscall(long number, ...)


# When we generate X_b, we're gonna add quote features according to figi.
# This data structure is five-layer dictionary for saving the streaming quote features
# for both training and inference time.
cdef class GroupedQuotes:

    # five-layer dictionary for saving streaming quotes.
    # 1st layer is figi
    # 2nd layer is party_id
    # 3rd layer is entry_type
    # 4th layer is entry (multiple entries sorted by entry_time
    # for training time, whereas for inference time it just holds the latest entry)
    # 5th layer is feature vector
    cdef vector[vector[vector[vector[vector[double]]]]] __grouped_data
    # number of saved_features: len([entry_date, price, quantity])
    cdef long int __number_of_saved_features
    # number of used_features: len([party_id, entry_type, entry_date, price, quantity])
    cdef long int __number_of_used_features
    # index of entry_date in saved_features
    cdef long unsigned int __feature_of_entry_date
    # index of figi, party_id, entry_type in quote_features
    cdef long unsigned int __feature_of_figi
    cdef long unsigned int __feature_of_party_id
    cdef long unsigned int __feature_of_entry_type
    cdef long int __dealer_window
    cdef long int __enable_inference


    # initialize dictionary
    def __init__(self, dealer_window, enable_inference, number_of_saved_features, number_of_used_features,
                 index_of_entry_date, index_of_figi, index_of_party_id, index_of_entry_type):
        self.__grouped_data.clear()
        self.__dealer_window = dealer_window
        self.__enable_inference = enable_inference
        self.__number_of_saved_features = number_of_saved_features
        self.__number_of_used_features = number_of_used_features
        self.__feature_of_entry_date = index_of_entry_date
        self.__feature_of_figi = index_of_figi
        self.__feature_of_party_id = index_of_party_id
        self.__feature_of_entry_type = index_of_entry_type


    # check if 'quote_1', 'quote_2' has the same 'figi', 'party_id', and 'entry_type'
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef int __same_group(self, double[:] quote_1, double[:] quote_2) nogil:
        return quote_1[self.__feature_of_figi] == quote_2[self.__feature_of_figi] and \
               quote_1[self.__feature_of_party_id] == quote_2[self.__feature_of_party_id] and \
               quote_1[self.__feature_of_entry_type] == quote_2[self.__feature_of_entry_type]

    # append quotes to group by figi, party_id, and entry_type
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef void __append_quotes_to_group(self, vector[vector[double]] &group, double[:, :] quotes) nogil:
        cdef long unsigned int rows = quotes.shape[0]
        cdef long unsigned int cols = quotes.shape[1]
        cdef long unsigned int i, j, zero_index = 0
        cdef long unsigned int previous_size = group.size()

        # for the inference time, we update the existing one
        if self.__enable_inference:
            # allocate memory
            if group.size() == 0:
                group.push_back(vector[double](cols))
            # update with the latest quote
            for i in prange(cols):
                group[zero_index][i] = quotes[rows - 1, i]
        # for the training time, we append the whole data
        else:
            # allocate memory
            group.resize(previous_size + rows)
            for i in prange(rows, schedule='static', nogil=True):
                group[previous_size + i].resize(cols)
            # update quote features
            for i in prange(rows, schedule='static', nogil=True):
                for j in prange(cols):
                    group[previous_size + i][j] = quotes[i][j]

    # add streaming quotes to the dictionary for both training and inference time
    # when training time, we save all quote feature vectors
    # when inference time, we just save the current ones
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef void __append(self, double[:, :] quotes):
        cdef long unsigned int rows = quotes.shape[0]
        # 'current_entry' indicates the entry that we are appending
        cdef long unsigned int current_entry = 0
        cdef long unsigned int st, en, md
        cdef long unsigned int figi, party_id, entry_type

        while current_entry < rows:
            # we group quotes by 'figi', 'party_id', and 'entry_type'
            # 'quotes' has sorted by 'figi', 'party_id', and 'entry_type'
            # we can find the group using binary search
            st = current_entry
            en = rows - 1
            while en - st > 1:
                md = (st + en) // 2
                if self.__same_group(quotes[current_entry, :], quotes[md, :]):
                    st = md
                else:
                    en = md
            # edge case
            if self.__same_group(quotes[current_entry, :], quotes[en, :]):
                st = en
            # current group -> quotes[current_entry:st+1, :]
            figi = int(quotes[current_entry, self.__feature_of_figi])
            party_id = int(quotes[current_entry, self.__feature_of_party_id])
            entry_type = int(quotes[current_entry, self.__feature_of_entry_type])
            # allocate memory
            if figi >= self.__grouped_data.size():
                self.__grouped_data.resize(figi + 1)
            if party_id >= self.__grouped_data[figi].size():
                self.__grouped_data[figi].resize(party_id + 1)
            if entry_type >= self.__grouped_data[figi][party_id].size():
                self.__grouped_data[figi][party_id].resize(entry_type + 1)
            # append quotes to group
            self.__append_quotes_to_group(self.__grouped_data[figi][party_id][entry_type], quotes[current_entry:st+1, self.__feature_of_entry_type + 1:])
            # move 'current_entry' to 'st' + 1
            current_entry = st + 1

    # add streaming quotes
    def append(self, double[:, :] quotes):
        self.__append(quotes)


    # find the previous bondcliq feature vector for the given execution_date
    # using binary search in entry_date column
    cdef long int __find_previous(self, long unsigned int figi, long unsigned int party_id, long unsigned int entry_type, double execution_date) nogil:
        # if there is no feature vector, return -1
        if figi >= self.__grouped_data.size():
            return -1
        if party_id >= self.__grouped_data[figi].size():
            return -1
        if entry_type >= self.__grouped_data[figi][party_id].size():
            return -1
        if self.__grouped_data[figi][party_id][entry_type].size() == 0:
            return -1
        # initialize st, en
        cdef long int st = 0
        cdef long int en = self.__grouped_data[figi][party_id][entry_type].size() - 1
        cdef long int md
        # binary search
        while en - st > 1:
            md = (st + en) // 2
            if self.__grouped_data[figi][party_id][entry_type][md][self.__feature_of_entry_date] < execution_date:
                st = md
            else:
                en = md
        # clarify edge cases
        if self.__grouped_data[figi][party_id][entry_type][en][self.__feature_of_entry_date] < execution_date:
            st = en
        if self.__grouped_data[figi][party_id][entry_type][st][self.__feature_of_entry_date] >= execution_date:
            return -1
        # return position
        return st

    # This function is for unit testing.
    def find_previous(self, long int figi, long int party_id, long int entry_type, double execution_date):
        return self.__find_previous(figi, party_id, entry_type, execution_date)


    # set one batch
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef void __set_batch(self, double[:] X_b, long unsigned int figi, double execution_date) nogil:
        cdef long unsigned int party_id, entry_type, zero_index = 0
        cdef long int entry
        cdef long unsigned int i, k, p, q
        cdef vector[double] feature_vector
        cdef vector[vector[double]] feature_vectors

        for party_id in range(self.__grouped_data[figi].size()):
            for entry_type in range(self.__grouped_data[figi][party_id].size()):
                if self.__grouped_data[figi][party_id][entry_type].size() == 0:
                    continue
                if self.__enable_inference:
                    entry = zero_index
                    if self.__grouped_data[figi][party_id][entry_type][entry][self.__feature_of_entry_date] >= execution_date:
                        entry = -1
                else:
                    entry = self.__find_previous(figi, party_id, entry_type, execution_date)
                if entry >= 0 and self.__grouped_data[figi][party_id][entry_type][entry].size() > 0:
                    # get feature vector -- 'entry_date', 'price', 'quantity', 'party_id', 'entry_type'
                    feature_vector.clear()
                    for i in range(self.__grouped_data[figi][party_id][entry_type][entry].size()):
                        feature_vector.push_back(self.__grouped_data[figi][party_id][entry_type][entry][i])
                    feature_vector.push_back(party_id)
                    feature_vector.push_back(entry_type)
                    # add 'feature_vector' to 'feature_vectors'
                    feature_vectors.push_back(feature_vector)

        # if feature_vectors is empty, continue
        if feature_vectors.size() == 0:
            return
        # sort feature vector by first value, second value, ...
        sort(feature_vectors.begin(), feature_vectors.end())
        # set X_b with recent feature vectors
        for i in prange(self.__dealer_window, schedule='static', nogil=True):
            # if the size of feature_vectors is less than number of vectors, break
            if i >= feature_vectors.size():
                continue
            # set batch with last ith vector
            p = feature_vectors.size() - i - 1
            q = (self.__dealer_window - i - 1) * self.__number_of_used_features
            for k in prange(self.__number_of_used_features):
                X_b[q + k] = feature_vectors[p][k]

    # generate X_b adding bondcliq features
    # using most __dealer_window recent feature vectors of the given figi
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef void __set_grouped_rows(self, double[:, :] X_b, double[:] figis, double[:] execution_dates):
        cdef long unsigned int batch_size = X_b.shape[0]
        cdef long unsigned int i, figi

        for i in range(batch_size):
        # for i in prange(batch_size, schedule='static', nogil=True):
            figi = int(figis[i])
            # if 'figi' doesn't exist in the grouped data, continue
            if figi >= self.__grouped_data.size():
                continue
            self.__set_batch(X_b[i, :], figi, execution_dates[i])

    def set_grouped_rows(self, double[:, :] X_b, double[:] figis, double[:] execution_dates):
        self.__set_grouped_rows(X_b, figis, execution_dates)


    # return feature matrix of the given figi, party_id, and entry_type
    def get_grouped_data(self, figi, party_id, entry_type):
        rlt = np.zeros((self.__grouped_data[figi][party_id][entry_type].size(), self.__number_of_saved_features))
        for i in range(self.__grouped_data[figi][party_id][entry_type].size()):
            for j in range(self.__number_of_saved_features):
                rlt[i, j] = self.__grouped_data[figi][party_id][entry_type][i][j]
        return rlt

    def get_all_data(self):
        result = np.zeros((0, self.__number_of_saved_features))
        for figi in range(self.__grouped_data.size()):
            for party_id in range(self.__grouped_data[figi].size()):
                for entry_type in range(self.__grouped_data[figi][party_id].size()):
                    result = np.concatenate((result, self.get_grouped_data(figi, party_id, entry_type)))
        return result
