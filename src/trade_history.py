import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import settings

from performance_helper import create_profiler
from grouped_history import GroupedHistory


def replace(l: list[str], old: str, new: str)-> list[str]:
    return list(map(lambda x: x.replace(old, new), l))

class TradeHistory:
    def __init__(self):
        self.rfq_features = None
        self.trade_features = None
        self.trade_feature_to_index = None
        self.rfq_feature_to_index = None
        self.grouped = None
        self.rfq_label_dict = None
        self.rfq_label_indices = None
        self.trades_columns = None
        self.trades_loc_dict = None
        self.sequence_length = settings.get('$.data.trades.sequence_length')
        self.groups = settings.get('$.data.trades.groups')
        self.rfq_labels = settings.get('$.data.trades.rfq_labels')
        self.rfq_label_set = set(self.rfq_labels)
        # Set the column order to be alphabetical to ensure a consistent order of the columns
        self.trades_columns = sorted(settings.get('$.data.trades.columns'))
        self.trades_loc_dict = dict(zip(self.trades_columns, range(len(self.trades_columns))))
        # create_mapping
        self.create_mapping()
        # create GroupedHistory
        self.grouped = GroupedHistory(self.groups, self.sequence_length, self.rfq_features, self.trade_features, self.rfq_labels)

    def cleanup(self):
        self.grouped.cleanup()
        del self.grouped

    def create_mapping(self):
        # Remove RFQ-level features
        remove_features = set(settings.get('$.data.trades.remove_features'))

        self.rfq_label_indices = list(map(self.get_feature_Z_index, self.rfq_labels))
        self.rfq_label_dict = dict(zip(self.rfq_labels, range(len(self.rfq_labels))))

        self.rfq_features = list(filter(lambda x: x not in self.rfq_label_set and x not in remove_features,
                                        self.trades_columns))
        self.trade_features = list(filter(lambda x: x not in remove_features, self.trades_columns))
        self.trade_feature_to_index = dict(zip(self.trade_features, range(len(self.trade_features))))
        self.rfq_feature_to_index = dict(zip(self.rfq_features, range(len(self.rfq_features))))

    def get_feature_Z_index(self, column_name):
        return self.trades_loc_dict[column_name]

    def get_figi_trace(self, figi, trade_count):
        return self.grouped.get_last_figi_trades(figi, trade_count)

    def append(self, trades: pa.table):
        profile = create_profiler('TradeHistory.append_trades')
        with profile():
            trades = trades.select(self.trades_columns)
            assert(trades.column_names == self.trades_columns)
            with profile('append trades'):
                self.grouped.append_trades(trades)

    def get_rfq_feature(self, X_b, feature_name):
        return X_b[:, self.rfq_feature_to_index[feature_name]]

    def get_rfq_feature_index(self, feature_name):
        return self.rfq_feature_to_index[feature_name]

    def set_rfq_feature(self, X_b: np.ndarray, feature_name: str, values: np.ndarray):
        if isinstance(feature_name, list):
            feature_indices = list(map(self.get_rfq_feature_index, feature_name))
        else:
            feature_indices = self.get_rfq_feature_index(feature_name)
        X_b[:, feature_indices] = values

    def get_trade_features(self):
        return self.trade_features

    def get_trade_feature_index(self, feature_name):
        if isinstance(feature_name, list):
            return list(map(self.trade_feature_to_index.get, feature_name))
        else:
            return self.trade_feature_to_index[feature_name]

    def get_trade_features_count(self):
        return len(self.get_trade_features())

    def get_trade_features_total_size(self):
        return self.get_group_count() * \
               self.get_sequence_length() * \
               self.get_trade_features_count()

    def get_features_total_size(self):
        return self.get_rfq_features_count() + self.get_trade_features_total_size()

    def get_trade_X_b(self, X_b: np.ndarray):
        '''

        :param raw_X_b: The raw X_b before the "remove_features" have been removed.
        :return:
        '''
        X_b = X_b[:, self.get_rfq_features_count():]
        X_b = X_b.reshape((-1, self.get_group_count(), self.get_sequence_length(), self.get_trade_features_count()))
        return X_b

    def get_trade_feature_values(self, X_b: np.ndarray, feature_name: str | list[str]):
        '''

        :param X_b: X_b, where the rfq and trade "remove_features" have already been filtered out.
        :param feature_name: the trade feature you want to retrieve.
        :return:
        '''
        X_b = self.get_trade_X_b(X_b)
        return X_b[:, :, :, self.get_trade_feature_index(feature_name)]

    def set_trade_feature_values(self, X_b, feature_name, feature_values):
        X_b = X_b[:, self.get_rfq_features_count():]
        X_b = X_b.reshape((-1, self.get_group_count(), self.get_sequence_length(), self.get_trade_features_count()))
        X_b[:, :, :, self.get_trade_feature_index(feature_name)] = feature_values

    def get_most_recent_trade_feature_value(self, X_b: np.ndarray, group_index: int, feature_name: str | list[str]):
        X_b = self.get_trade_feature_values(X_b, feature_name)
        return X_b[:, group_index, -1]

    def get_most_recent_trade_feature_value_from_closest_group(self, X_b: np.ndarray, feature_name: str | list[str]):
        '''
        The groups are in an order, defined when the generator was created. This order normally specifies an
        order of precedence that the group has to the target RFQ. For example, the trades from the same figi
        are the most relevant to an RFQ with that same FIGI, followed by trades from the same issuer, then
        the industry, and then the sector. It is often the case that we want to get the most recent trade
        for what is considered the most relevant group; we do this for example when we're normalizing the data,
        such as in the "NormalizedTraceDataGenerator." Sometimes though, the most relevant group will not have any
        recent trades before the target RFQ. For that reason, we may want to go to the next group, etc.
        :param X_b: The batch's features from which you want to extract the most recent trade feature value from
        the closest group.
        :param feature_name: The name of the feature for which you want to get the value for the closest group
        :return: the feature value for the feature_name in the closest group, or None otherwise.
        '''
        feature_values = np.empty((X_b.shape[0],))
        feature_values[:] = np.NaN
        for i in range(self.get_group_count()):
            sizes = self.get_most_recent_trade_feature_value(X_b, i, 'quantity')
            group_feature_values = self.get_most_recent_trade_feature_value(X_b, i, feature_name)
            feature_values = self.filter_feature_value(feature_values, group_feature_values, sizes)
            '''
            try:
                assert ((group_feature_values[sizes > 0] != 0.0).all())
            except AssertionError as e:
                print('Assertion failed.')
                print(f'Group: {self.groups[i]}')
                print(f'i: {i}')
                X_b = X_b[:, self.get_rfq_features_count():]
                X_b = np.reshape(X_b, (X_b.shape[0], self.get_group_count(),
                                       self.get_sequence_length(), self.get_trade_features_count()))
                X_b = X_b[:, i, :, :]
                print('Trade feature names:')
                print(self.get_trade_features())
                print('Trade feature values:')
                print(X_b)
                raise e
            '''
        return feature_values


    @staticmethod
    def filter_feature_value(feature_values:np.ndarray, group_feature_values:np.ndarray, sizes:np.ndarray):
        return np.where(
            np.isnan(feature_values),
            # Where the size is greater than 0 and at least on the labels is valid (not zero),
            # because the price really should be non-zero for this to be considered a trade we
            # can price
            np.where(sizes > 0, group_feature_values, np.NaN),
            feature_values
        )

    def get_figi_trade_features_and_count(self, X_b: np.ndarray, feature_names: list[str]):
        # We assume that the figi group is the first group.
        X_b = self.get_trade_X_b(X_b)
        quantity_index = self.get_trade_feature_index('quantity')
        quantities = X_b[:, 0, :, quantity_index]
        features_counts = np.sum(quantities > 0, axis=1)
        feature_indices = list(map(self.get_trade_feature_index, feature_names))
        features = X_b[:, 0, :, :][:, :, feature_indices]
        return features, features_counts, quantities

    def get_sequence_length(self):
        return self.sequence_length

    def get_rfq_label_indices(self):
        return self.rfq_label_indices

    def get_rfq_label_index(self, label_name):
        return self.rfq_label_dict[label_name]

    def get_rfq_labels(self):
        return self.rfq_labels

    def get_rfq_label_count(self):
        return len(self.get_rfq_labels())

    def get_rfq_features(self):
        return self.rfq_features

    def get_rfq_features_count(self):
        return len(self.get_rfq_features())

    def get_group_count(self):
        return len(self.groups)

    def generate_batch(self, Z_b: pa.Table):
        assert(Z_b.column_names == self.trades_columns)
        return self.__generate_batch(Z_b.to_pandas().to_numpy())

    def generate_batch_np(self, Z_b: np.ndarray):
        assert(Z_b.shape[1] == len(self.trades_columns))
        return self.__generate_batch(Z_b)

    def __generate_batch(self, Z_b: np.ndarray):
        return self.grouped.generate_batch(Z_b)