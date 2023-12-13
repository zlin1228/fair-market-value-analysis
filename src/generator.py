import settings
settings.override_if_main(__name__, 1)

from datetime import datetime
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytz
import pyximport
import tensorflow as tf

from load_data import get_initial_quotes, get_initial_trades
from grouped_history import combine_trace
from performance_helper import create_profiler
from quote_history import QuoteHistory
from trade_history import TradeHistory

pyximport.install(language_level=3)


def pa_to_numpy(table: pa.table)-> np.ndarray:
    column_names = table.column_names
    return table.to_pandas()[column_names].to_numpy()


class TraceDataGenerator(tf.keras.utils.Sequence):
    # trace is the data frame where each row are the
    # RFQ trade features
    def __init__(self,
                 data_cube: TradeHistory,
                 quote_cube=None,
                 batch_size=None,
                 should_shuffle=None,
                 Z: pa.Table = None,  # Z must me a subset of trace with the same columns, and the same ordinals.
                 ):
        self.data_cube = data_cube
        profile = create_profiler('TraceDataGenerator.__init__')

        if batch_size is None:
            batch_size = settings.get('$.batch_size')
        self.batch_size = batch_size
        if should_shuffle is None:
            should_shuffle = settings.get('$.should_shuffle')
        self.should_shuffle = should_shuffle
        # seed using random to enable cascading deterministic behavior
        self.np_rng = np.random.default_rng(random.randint(1, 1_000_000))

        if settings.get('$.enable_bondcliq'):
            self.quote_cube = quote_cube

        if settings.get('$.enable_bondcliq'):
            self.quote_cube = quote_cube

        with profile('_finish_init'):
            self._finish_init(Z=Z)

    def cleanup(self, clean_data_cube):
        del self.Z
        if clean_data_cube:
            self.data_cube.cleanup()
            del self.data_cube

    def _finish_init(self, Z:pa.Table=None):
        self.Z = Z
        if self.Z is None:
            self.Z = self.data_cube.grouped.get_trace()
        else:
            self.Z = combine_trace(self.Z)
        self.Z = self.Z.select(self.data_cube.trades_columns)
        assert pc.all(pc.less_equal(self.Z['execution_date'], self.Z['report_date'])).as_py()
        self.Z = pa_to_numpy(self.Z)
        assert (self.Z[:, self.get_rfq_feature_Z_index('execution_date')] <= self.Z[:, self.get_rfq_feature_Z_index('report_date')]).all()
        self.shuffle()

    def get_figi_trace(self, figi: int, trade_count: int) -> pa.Table:
        return self.data_cube.get_figi_trace(figi, trade_count)

    def get_rfq_feature(self, X_b, feature_name):
        return self.data_cube.get_rfq_feature(X_b, feature_name)

    def get_rfq_feature_index(self, feature_name):
        return self.data_cube.get_rfq_feature_index(feature_name)

    def set_rfq_feature(self, X_b, feature_name, values):
        self.data_cube.set_rfq_feature(X_b, feature_name, values)

    def get_trade_features(self):
        return self.data_cube.get_trade_features()

    def get_trade_feature_index(self, feature_name):
        return self.data_cube.get_trade_feature_index(feature_name)

    def get_trade_features_count(self):
        return self.data_cube.get_trade_features_count()

    def get_trade_features_total_size(self):
        return self.data_cube.get_trade_features_total_size()

    def get_features_total_size(self, enable_bondcliq=False):
        if enable_bondcliq:
            return self.data_cube.get_features_total_size() + self.quote_cube.get_features_total_size()
        else:
            return self.data_cube.get_features_total_size()

    def get_trade_X_b(self, X_b):
        return self.data_cube.get_trade_X_b(X_b)

    def get_trade_feature_values(self, X_b, feature_name):
        return self.data_cube.get_trade_feature_values(X_b, feature_name)

    def set_trade_feature_values(self, X_b, feature_name, feature_values):
        self.data_cube.set_trade_feature_values(X_b, feature_name, feature_values)

    def get_most_recent_trade_feature_value(self, X_b, group_index, feature_name):
        return self.data_cube.get_most_recent_trade_feature_value(X_b, group_index, feature_name)

    def get_most_recent_trade_feature_value_from_closest_group(self, X_b, feature_name):
        return self.data_cube.get_most_recent_trade_feature_value_from_closest_group(X_b, feature_name)

    def get_figi_trade_features_and_count(self, X_b, feature_names):
        return self.data_cube.get_figi_trade_features_and_count(X_b, feature_names)

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_sequence_length(self):
        return self.data_cube.get_sequence_length()

    def get_rfq_label_indices(self):
        return self.data_cube.get_rfq_label_indices()

    def get_rfq_label_index(self, label_name):
        return self.data_cube.get_rfq_label_index(label_name)

    def get_rfq_labels(self):
        return self.data_cube.get_rfq_labels()

    def get_rfq_label_count(self):
        return self.data_cube.get_rfq_label_count()

    def get_rfq_features(self):
        return self.data_cube.get_rfq_features()

    def get_rfq_features_count(self):
        return self.data_cube.get_rfq_features_count()

    def get_rfq_feature_Z_index(self, rfq_feature):
        return self.data_cube.get_feature_Z_index(rfq_feature)

    def get_groups_column_count(self):
        return self.data_cube.get_groups_column_count()

    def get_group_count(self):
        return self.data_cube.get_group_count()

    def shuffle(self):
        if self.should_shuffle:
            self.np_rng.shuffle(self.Z)

    def on_epoch_end(self):
        self.shuffle()

    def generate_batch(self, Z_b: pa.Table):
        assert(Z_b.column_names == self.data_cube.trades_columns)
        return self.__generate_batch(Z_b.to_pandas().to_numpy())

    def generate_batch_np(self, Z_b: np.ndarray):
        assert(Z_b.shape[1] == len(self.data_cube.trades_columns))
        return self.__generate_batch(Z_b)

    def __generate_batch(self, Z_b: np.ndarray):
        X_b, Y_b = self.data_cube.generate_batch_np(Z_b)
        if settings.get('$.enable_bondcliq'):
            X_b = np.concatenate((X_b, self.quote_cube.generate_batch(Z_b[:, self.get_rfq_feature_Z_index('figi')],
                                                                      Z_b[:, self.get_rfq_feature_Z_index('execution_date')])), axis=1)
        return X_b, Y_b

    def get_Z(self):
        return self.Z

    def get_Z_b(self, index: int)->np.ndarray:
        return self.Z[(index * self.batch_size):((index + 1) * self.batch_size), :]

    # Returns the Z_b representing all of the trades in a report timestamp range (exclusive)
    def get_date_Z_b(self, start_report_date, end_report_date):
        report_dates = self.Z[self.get_rfq_feature_Z_index('report_date'), :]
        return self.Z[(report_dates > start_report_date) & (report_dates < end_report_date)]

    # For use during out of sample test time
    # or production inference time
    def getitem(self, index):
        return self.__getitem__(index)

    # @profile
    def __getitem__(self, index):
        # S stands for sequences
        # _b stands for batch

        __get_item__start_time = time.time()
        Z_b = self.get_Z_b(index)
        return self.__generate_batch(Z_b)

    def __len__(self):
        return int(self.Z.shape[0] / self.batch_size)

    '''
    When we evaluate, we want to set a quantity range to filter trades.
    This function filters all trades in self.Z_b by a quantity range in settings.py.
    '''
    def apply_quantity_filter(self):
        index_of_quantity = self.get_rfq_feature_Z_index('quantity')
        min_quantity = settings.get('$.filter_for_evaluation.minimum_quantity')
        max_quantity = settings.get('$.filter_for_evaluation.maximum_quantity')
        condition = (self.Z[:, index_of_quantity] >= min_quantity) & (self.Z[:, index_of_quantity] <= max_quantity)
        self.Z = self.Z[condition]

    '''
    This function returns a date from a report_date.
    Here, the report_date is a timestamp in nanosecond.
    And, the date is a string in format 'YYYY-MM-DD'.
    '''
    def get_date_from_timestamp(self, report_date):
        return datetime.fromtimestamp(report_date / 1e9, tz=pytz.timezone(settings.get('$.finra_timezone'))).date().strftime('%Y-%m-%d')

    '''
    This function returns the number of distinct days in trades.
    We convert every report_date to a date, then use set to get the number of distinct days.
    '''
    def get_number_of_days(self):
        report_dates = self.Z[:, self.get_rfq_feature_Z_index('report_date')]
        return len(set([self.get_date_from_timestamp(report_date) for report_date in report_dates]))

    '''
    This function filters trades by a liquidity range.
    First of all, we calculate the frequency of each trade (by figi) in the whole date range.
    The liquidity range in settings.py is the average value per day.
    So we multiply the average value by number of days.
    Then, we filter trades by a frequency range.
    '''
    def apply_liquidity_filter(self):
        # Get the number of days
        number_of_days = self.get_number_of_days()

        # Get the index of 'figi' in the Z array
        index_of_figi = self.get_rfq_feature_Z_index('figi')

        # Extract the 'figi' column from the Z array and convert it to integer
        figis = self.Z[:, index_of_figi].astype(int)

        # Count the occurrences of each unique 'figi' value
        count_of_figis = np.bincount(figis)

        # Get the minimum and maximum liquidity settings, scale by the number of days
        min_liquidity = settings.get('$.filter_for_evaluation.minimum_liquidity') * number_of_days
        max_liquidity = settings.get('$.filter_for_evaluation.maximum_liquidity') * number_of_days
        
        # Count the occurrences of each 'figi' in the Z array
        figi_counts = count_of_figis[self.Z[:, index_of_figi].astype(int)]

        # Create a condition that is True for rows where the 'figi' count is within the desired liquidity range
        condition = (figi_counts >= min_liquidity) & (figi_counts <= max_liquidity)
        
        # Apply the condition to the Z array, keeping only the rows where the condition is True
        self.Z = self.Z[condition]

    def filter_for_evaluation(self):
        self.apply_quantity_filter()
        self.apply_liquidity_filter()


def main():
    trade_history = TradeHistory()
    trade_history.append(get_initial_trades())

    if settings.get('$.enable_bondcliq'):
        quote_history = QuoteHistory()
        quote_history.append(get_initial_quotes())
        generator = TraceDataGenerator(trade_history, quote_history, 100)
    else:
        generator = TraceDataGenerator(trade_history, batch_size=100)

    def print_generator_info(generator):
        print("About to get_item")
        get_item_start = time.time()
        for i in range(1):
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
