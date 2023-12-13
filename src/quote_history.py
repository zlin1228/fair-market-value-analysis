import numpy as np
import pyarrow as pa

import settings
from performance_helper import create_profiler
import pyximport
pyximport.install(language_level=3)

from grouped_quotes import GroupedQuotes


class QuoteHistory:

    # trace is the pyarrow table where each row are the BondCliQ quote features
    def __init__(self):
        # load dealer_window and quote_features from settings
        self.__quote_features = settings.get('$.data.quotes.features')
        self.__dealer_window = settings.get('$.data.quotes.dealer_window')
        # currently, quote features are figi, party_id, entry_type, entry_date, price and quantity.
        # we are saving entry_date, price and quantity in the feature vector
        self.__saved_features = [feature for feature in self.__quote_features
                                 if feature not in ['figi', 'party_id', 'entry_type']]
        # when we generate batches, we use features except figi
        self.__used_features = [feature for feature in self.__quote_features
                                if feature not in ['figi']]
        # set __number_of_saved_features
        self.__number_of_saved_features = len(self.__saved_features)
        # set __number_of_used_features
        self.__number_of_used_features = len(self.__used_features)
        # create GroupedQuotes
        self.group = GroupedQuotes(self.__dealer_window,                      # dealer_window - 10
                                   settings.get('$.enable_inference'),        # enable_inference
                                   self.__number_of_saved_features,           # number of saved features - 3
                                   self.__number_of_used_features,            # number of used features - 5
                                   self.__saved_features.index('entry_date'), # index of entry_date in saved_features
                                   self.__quote_features.index('figi'),       # index of figi       in quote_features
                                   self.__quote_features.index('party_id'),   # index of party_id   in quote_features
                                   self.__quote_features.index('entry_type')) # index of entry_type in quote_features

    def append(self, quotes: pa.Table):
        profile = create_profiler('QuoteHistory.append')
        quotes = quotes.select(self.__quote_features)
        # sort the quotes by entry_date
        with profile('sort_streaming_quotes'):
            column_sort_list = quotes.column_names
            column_sort_list.remove('figi')
            column_sort_list.remove('party_id')
            column_sort_list.remove('entry_type')
            column_sort_list.remove('entry_date')
            column_sort_list = ['figi', 'party_id', 'entry_type', 'entry_date'] + column_sort_list
            # Create list of tuples of column name and 'ascending'
            quotes = quotes.sort_by([(column, 'ascending') for column in column_sort_list])
        # add streaming quotes
        with profile('add_streaming_quotes'):
            self.group.append(quotes.to_pandas().to_numpy())

    def generate_batch(self, figis, execution_dates):
        batch_size = figis.shape[0]
        sequences_size = len(self.__used_features) * self.__dealer_window
        X_b = np.zeros((batch_size, sequences_size))
        self.group.set_grouped_rows(X_b, figis, execution_dates)
        return X_b
