import time

import numpy as np
import pyarrow as pa
import tensorflow as tf

from ema_model import exponential_moving_average
from generator import TraceDataGenerator
from load_data import get_initial_trades
import settings
from trade_history import TradeHistory

class NormalizedTraceDataGenerator(tf.keras.utils.Sequence):
    '''
    Wrapper for TraceDataGenerator which takes care of the normalization used for the NN / Deep learning models
    '''
    generator: TraceDataGenerator

    def __init__(self, generator: TraceDataGenerator):
        self.generator = generator

    def cleanup(self, clean_data_cube):
        self.generator.cleanup(clean_data_cube)
        del self.generator

    @staticmethod
    def protected_log(x):
        """
        A log which can handle 0.0 values, but not values less than -1.0.
        :param x: a numpy array
        :return: the log of x + 1.0, element wise
        """
        assert(np.all(x >= 0.0))
        return np.where(x <= 0.0, 0.0, np.log(x + 1.0))


    def normalize_x(self, X_b):
        """

        :param X_b:
        :return: previous_labels
        """
        previous_labels = None

        if settings.get('$.ema_normalization'):
            previous_labels = exponential_moving_average(self.generator, X_b)
        rfq_labels = self.generator.get_rfq_labels()
        if not settings.get('$.ema_normalization'):
            previous_labels = np.empty((X_b.shape[0], self.generator.get_rfq_label_count()))
            for j in range(self.generator.get_rfq_label_count()):
                previous_labels[:, j] = self.generator.get_most_recent_trade_feature_value_from_closest_group(X_b,
                                                                                                              rfq_labels[j])
        na_filter = ~(np.isnan(previous_labels).any(axis=1))
        # Assert that we don't have an empty batch
        assert (np.any(na_filter))
        #X_b = X_b[na_filter]
        #previous_labels = previous_labels[na_filter]

        # Useful watch for debugging this assert:
        # X_b[:, self.generator.get_rfq_features_count():].reshape((X_b.shape[0], self.generator.get_group_count(), self.generator.get_sequence_length(), self.generator.get_trade_features_count()))[:, :, :, self.generator.get_trade_feature_index('price')]
        assert not (previous_labels == 0.0).all(axis=1).any()

        # Normalize the trade features equivalent to the RFQ labels
        rfq_label_trade_feature_values = self.generator.get_trade_feature_values(X_b, rfq_labels)
        reshaped_previous_labels = np.reshape(previous_labels,
                                              (previous_labels.shape[0], 1, 1, previous_labels.shape[1]))
        rfq_label_trade_feature_values = rfq_label_trade_feature_values - reshaped_previous_labels
        self.generator.set_trade_feature_values(X_b, rfq_labels, rfq_label_trade_feature_values)

        # Get the RFQ time features
        trade_time_feature_names = ['execution_date', 'report_date']
        rfq_execution_dates = self.generator.get_rfq_feature(X_b, 'execution_date')
        rfq_report_dates = self.generator.get_rfq_feature(X_b, 'report_date')

        # Normalize the trade relative time features
        trade_execution_dates = self.generator.get_trade_feature_values(X_b, 'execution_date')
        trade_execution_dates = np.reshape(trade_execution_dates, trade_execution_dates.shape + (1,))
        relative_time_features = ['maturity', 'issue_date']
        trade_relative_time_feature_values = self.generator.get_trade_feature_values(X_b, relative_time_features)

        rfq_relative_time_feature_values = NormalizedTraceDataGenerator.protected_log(
            np.abs(trade_relative_time_feature_values - trade_execution_dates))
        self.generator.set_trade_feature_values(X_b, relative_time_features, rfq_relative_time_feature_values)

        # Normalize the trade time features
        trade_date_values = self.generator.get_trade_feature_values(X_b, trade_time_feature_names)

        relative_trade_date_values = np.reshape(rfq_execution_dates, (rfq_execution_dates.shape[0], 1, 1, 1)) \
                               - trade_date_values
        assert np.all(relative_trade_date_values > 0.0)
        trade_date_values = NormalizedTraceDataGenerator.protected_log(relative_trade_date_values)
        self.generator.set_trade_feature_values(X_b, trade_time_feature_names, trade_date_values)

        # we set the RFQ date features to 0.0 as they are no longer needed because
        # all of the trade feature dates have been made relative to them,
        # and also to respect causality (for example, we don't know at the time of the
        # arrival of the RFQ when the report date is going to be relative to the execution
        # date, for example.
        for relative_time_feature in relative_time_features:
            rfq_relative_time_feature_values = self.generator.get_rfq_feature(X_b, relative_time_feature)
            rfq_relative_time_feature_values = NormalizedTraceDataGenerator.protected_log(
                                                np.abs(rfq_relative_time_feature_values - rfq_execution_dates))
            self.generator.set_rfq_feature(X_b, relative_time_feature, rfq_relative_time_feature_values)
        self.generator.set_rfq_feature(X_b, trade_time_feature_names, 0.0)

        rfq_quantity_values = self.generator.get_rfq_feature(X_b, 'quantity')
        rfq_quantity_values = np.log(1.0 + rfq_quantity_values)
        self.generator.set_rfq_feature(X_b, 'quantity', rfq_quantity_values)

        trade_quantity_values = self.generator.get_trade_feature_values(X_b, 'quantity')
        trade_quantity_values = NormalizedTraceDataGenerator.protected_log(trade_quantity_values)
        self.generator.set_trade_feature_values(X_b, 'quantity', trade_quantity_values)

        return na_filter, X_b, previous_labels

    def get_rfq_label_index(self, label_name):
        return self.generator.get_rfq_label_index(label_name)

    def get_rfq_feature_Z_index(self, rfq_feature):
        return self.generator.get_rfq_feature_Z_index(rfq_feature)

    def get_rfq_feature(self, X_b, feature_name):
        return self.generator.get_rfq_feature(X_b, feature_name)

    def normalize(self, X_b, Y_b):
        na_filter, X_b, previous_labels = self.normalize_x(X_b)
        X_b = X_b[na_filter]
        Y_b = Y_b[na_filter]
        previous_labels = previous_labels[na_filter]
        # Y_b = (Y_b - previous_labels) / previous_labels
        Y_b = Y_b - previous_labels
        return previous_labels, X_b, Y_b

    @staticmethod
    def unnormalize_y(previous_labels, Y_b_hat):
        return Y_b_hat + previous_labels
        # return Y_b_hat * previous_labels + previous_labels

    def on_epoch_end(self):
        self.generator.on_epoch_end()

    # generates the batch of X_b without normalizing
    # it. This is useful for out-of sample test time
    # or production use of the model.
    def generate_batch(self, Z_b: pa.Table):
        X_b, Y_b = self.generator.generate_batch(Z_b)
        assert (X_b.shape[0] > 0)
        return X_b, Y_b

    def generate_batch_np(self, Z_b: np.ndarray)->(np.ndarray, np.ndarray):
        X_b, Y_b = self.generator.generate_batch_np(Z_b)
        assert (X_b.shape[0] > 0)
        return X_b, Y_b

    def get_Z(self):
        return self.generator.get_Z()

    def get_Z_b(self, index):
        return  self.generator.get_Z_b(index)

    # For use during out of sample test time
    # or production inference time
    def getitem(self, index):
        return self.generator.__getitem__(index)

    def set_batch_size(self, batch_size):
        self.generator.set_batch_size(batch_size)

    def __getitem__(self, index):
        X_b, Y_b = self.generator.__getitem__(index)
        _, X_b, Y_b = self.normalize(X_b, Y_b)
        assert (X_b.shape[0] > 0)
        return X_b, Y_b

    def __len__(self):
        return self.generator.__len__()

    def get_rfq_features(self):
        return self.generator.get_rfq_features()

    def get_trade_features(self):
        return self.generator.get_trade_features()

def main():
    trade_history = TradeHistory()
    trade_history.append(get_initial_trades())

    generator = TraceDataGenerator(trade_history, batch_size=100)
    generator = NormalizedTraceDataGenerator(generator)

    def print_generator_info(generator):
        print("About to get_item")
        get_item_start = time.time()
        for i in range(2000):
            (X_b, Y_b) = generator.__getitem__(i)
        get_item_end = time.time()
        print("Done getting item")
        print(f'It took this much time to get item: {get_item_end - get_item_start}')
        print(X_b)
        print(Y_b)

    print_generator_info(generator)
    print('RFQ features:')
    print(generator.get_rfq_features())
    print('Trade feautures:')
    print(generator.get_trade_features())


if __name__ == "__main__":
    main()
