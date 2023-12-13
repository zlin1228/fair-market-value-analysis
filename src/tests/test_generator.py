import sys
sys.path.insert(0, '../')

standard_settings = {
    '$.enable_inference': True,
    '$.enable_bondcliq': True,
    '$.data.trades.proportion_of_trades_to_load': 0.0001,
    '$.data.quotes.dealer_window': 2,
    '$.data_path': 's3://deepmm.test.data/deepmm.parquet/v0.7.zip',
    '$.filter_for_evaluation.apply_filter': True,
    '$.filter_for_evaluation.minimum_quantity': 5,
    '$.filter_for_evaluation.maximum_quantity': 9,
    '$.filter_for_evaluation.minimum_liquidity': 2,
    '$.filter_for_evaluation.maximum_liquidity': 3,}
import settings
settings.override(standard_settings)

import unittest
from unittest.mock import patch

import datetime
import numpy as np
import pyarrow as pa
from pytz import timezone

from generator import TraceDataGenerator
from load_data import get_initial_quotes, get_initial_trades
from tests.helpers import CleanEnvironment
from normalized_trace_data_generator import NormalizedTraceDataGenerator
from quote_history import QuoteHistory
from trade_history import TradeHistory


class GeneratorTestCase(unittest.TestCase):
    def test_set_rows(self):
        with CleanEnvironment(lambda _: {**standard_settings,
                            '$.data.trades.columns': ['figi', 'yield', 'price', 'report_date', 'execution_date', 'maturity', 'quantity'],
                            '$.data.trades.sequence_length': 3,
                            '$.data.trades.groups': ['figi']}):
            trace = pa.table({'figi':           [    1,    2,    1,    1,    2,    2,    1,    1,    2,    2],
                              'yield':          [    1,    1,    2,    1,    1,    2,    1,    2,    1,    1],
                              'price':          [  7.0,  8.0,  5.0,  4.0,  9.0,  6.0,  7.0,  5.0,  7.0,  8.0],
                              'report_date':    [  1.1,  1.1,  2.1,  3.1,  3.1,  4.1,  5.1,  6.1,  6.1,  6.1],
                              'execution_date': [  1.0,  1.0,  2.0,  3.0,  3.0,  4.0,  5.0,  6.0,  6.0,  6.0],
                              'maturity':       [  2.0,  2.0,  3.0,  4.0,  4.0,  5.0,  6.0,  7.0,  7.0,  7.0],
                              'quantity':       [  1.0,  2.0,  1.0,  2.0,  1.0,  1.0,  2.0,  2.0,  1.0,  1.0]})
            data_cube = TradeHistory()
            data_cube.append(trace)
            quotes = pa.table({'figi':       [  1.0,   1.0,   2.0,   2.0,   1.0,   1.0,   1.0,   1.0,   1.0,   2.0],
                               'party_id':   [  1.0,   2.0,   3.0,   1.0,   2.0,   3.0,   1.0,   2.0,   3.0,   1.0],
                               'entry_type': [  0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0],
                               'entry_date': [  0.8,   1.0,   2.0,   4.0,   5.0,   6.0,   5.0,   3.0,   8.0,   9.0],
                               'price':      [  5.0,   5.0,   6.0,   6.0,   4.0,   4.0,   7.0,   7.0,   3.0,   3.0],
                               'quantity':   [  1.0,   2.0,   1.0,   2.0,   1.0,   2.0,   1.0,   2.0,   1.0,   2.0]})
            quote_cube = QuoteHistory()
            quote_cube.append(quotes)
            generator = TraceDataGenerator(data_cube, quote_cube, 10, should_shuffle=False)
            batch_X, batch_Y = generator.__getitem__(0)
            self.assertTrue(np.array_equal(np.array([[ 1.0, 2.0, 1.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                     [ 1.0, 2.0, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                     [ 2.0, 3.0, 1.0, 2.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 7.0, 1.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                     [ 3.0, 4.0, 2.0, 3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 7.0, 1.0, 1.1, 2.0, 3.0, 5.0, 1.0, 2.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                     [ 3.0, 4.0, 1.0, 3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 8.0, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 1.0, 3.0, 0.0],
                                                     [ 4.0, 5.0, 1.0, 4.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 8.0, 2.0, 1.1, 3.0, 4.0, 9.0, 1.0, 3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 1.0, 3.0, 0.0],
                                                     [ 5.0, 6.0, 2.0, 5.1, 1.0, 2.0, 7.0, 1.0, 1.1, 2.0, 3.0, 5.0, 1.0, 2.1, 3.0, 4.0, 4.0, 2.0, 3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 7.0, 2.0, 2.0, 1.0],
                                                     [ 6.0, 7.0, 2.0, 6.1, 2.0, 3.0, 5.0, 1.0, 2.1, 3.0, 4.0, 4.0, 2.0, 3.1, 5.0, 6.0, 7.0, 2.0, 5.1, 5.0, 4.0, 1.0, 2.0, 0.0, 5.0, 7.0, 1.0, 1.0, 0.0],
                                                     [ 6.0, 7.0, 1.0, 6.1, 1.0, 2.0, 8.0, 2.0, 1.1, 3.0, 4.0, 9.0, 1.0, 3.1, 4.0, 5.0, 6.0, 1.0, 4.1, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 1.0, 3.0, 0.0],
                                                     [ 6.0, 7.0, 1.0, 6.1, 1.0, 2.0, 8.0, 2.0, 1.1, 3.0, 4.0, 9.0, 1.0, 3.1, 4.0, 5.0, 6.0, 1.0, 4.1, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 1.0, 3.0, 0.0]]), batch_X))
            self.assertTrue(np.alltrue(np.array([[7.0], [8.0], [5.0], [4.0], [9.0], [6.0], [7.0], [5.0], [7.0], [8.0]]) == batch_Y))

    def test_get_trade_features(self):
        with CleanEnvironment(lambda _: {**standard_settings,
                            '$.data.trades.columns': ['figi', 'yield', 'price', 'report_date', 'execution_date', 'maturity', 'quantity'],
                            '$.data.trades.sequence_length': 3,
                            '$.data.trades.groups': ['figi']}):
            trace = pa.table({'figi':           [  1.0,  2.0,  1.0,  2.0,  1.0,  2.0,  1.0,  2.0,  1.0,  2.0],
                            'yield':          [  1.0,  1.0,  2.0,  1.0,  1.0,  2.0,  1.0,  1.0,  2.0,  1.0],
                            'price':          [  7.0,  8.0,  5.0,  9.0,  4.0,  6.0,  7.0,  8.0,  5.0,  7.0],
                            'report_date':    [  1.1,  1.1,  2.1,  3.1,  3.1,  4.1,  5.1,  6.1,  6.1,  6.1],
                            'execution_date': [  1.0,  1.0,  2.0,  3.0,  3.0,  4.0,  5.0,  6.0,  6.0,  6.0],
                            'maturity':       [  2.0,  2.0,  3.0,  4.0,  4.0,  5.0,  6.0,  7.0,  7.0,  7.0],
                            'quantity':       [  1.0,  2.0,  1.0,  1.0,  2.0,  1.0,  2.0,  1.0,  2.0,  1.0]})
            data_cube = TradeHistory()
            data_cube.append(trace)
            quotes = pa.table({'figi':      [  1.0,   1.0,   2.0,   2.0,   1.0,   1.0,   1.0,   1.0,   1.0,   2.0],
                            'party_id':   [  1.0,   2.0,   3.0,   1.0,   2.0,   3.0,   1.0,   2.0,   3.0,   1.0],
                            'entry_type': [  0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0],
                            'entry_date': [  0.8,   1.0,   2.0,   4.0,   5.0,   6.0,   5.0,   3.0,   8.0,   9.0],
                            'price':      [  5.0,   5.0,   6.0,   6.0,   4.0,   4.0,   7.0,   7.0,   3.0,   3.0],
                            'quantity':   [  1.0,   2.0,   1.0,   2.0,   1.0,   2.0,   1.0,   2.0,   1.0,   2.0]})
            quote_cube = QuoteHistory()
            quote_cube.append(quotes)
            generator = TraceDataGenerator(data_cube, quote_cube, 20, should_shuffle=False)
            batch_X, batch_Y = generator.__getitem__(0)

            rfq_report_date = generator.get_rfq_feature(batch_X, 'report_date')
            self.assertTrue((rfq_report_date == trace['report_date']).all())

    def test_random_shuffling(self):
        with CleanEnvironment(lambda _: standard_settings):
            trade_history = TradeHistory()
            trade_history.append(get_initial_trades())

            quote_history = QuoteHistory()
            quote_history.append(get_initial_quotes())

            generator = TraceDataGenerator(trade_history, quote_history, 100, should_shuffle=False)
            length = generator.__len__()
            z_index = int(length / 2)
            Z_b = generator.get_Z_b(z_index)

            X_b, Y_b = generator.generate_batch_np(Z_b)

            shuffle = np.random.choice(Z_b.shape[0], Z_b.shape[0], replace=False)
            self.assertTrue(shuffle.shape[0] == Z_b.shape[0])

            Z_b_s = Z_b[shuffle, :]
            X_b_s = X_b[shuffle, :]
            Y_b_s = Y_b[shuffle, :]

            X_b_s_g, Y_b_s_g = generator.generate_batch_np(Z_b_s)

            self.assertTrue(np.array_equal(X_b_s, X_b_s_g))
            self.assertTrue(np.array_equal(Y_b_s, Y_b_s_g))

            normalized_generator = NormalizedTraceDataGenerator(generator)

            feature_size = generator.get_features_total_size(enable_bondcliq=False)
            na_filter, X_b_n, previous_labels = normalized_generator.normalize_x(X_b[:, :feature_size].copy())
            X_b_n = X_b_n[na_filter, :]
            X_b = X_b[na_filter, :]
            shuffle = np.random.choice(X_b_n.shape[0], X_b_n.shape[0], replace=False)
            X_b_s_n = X_b_n[shuffle, :]

            X_b_s = X_b[shuffle, :feature_size]
            na_filter, X_b_s_g_n, _ = normalized_generator.normalize_x(X_b_s.copy())
            X_b_s_g_n = X_b_s_g_n[na_filter]
            self.assertTrue(np.array_equal(X_b_s_n, X_b_s_g_n))

    def test_apply_quantity_filter(self):
        with CleanEnvironment(lambda _: standard_settings):
            with patch('generator.TraceDataGenerator.__init__') as mock_generator_init, \
                 patch('trade_history.TradeHistory.__init__') as mock_trade_history_init, \
                 patch('generator.TraceDataGenerator.get_rfq_feature_Z_index') as mock_get_rfq_feature_Z_index:
                mock_generator_init.return_value = None
                mock_trade_history_init.return_value = None
                mock_get_rfq_feature_Z_index.return_value = 0
                generator = TraceDataGenerator(TradeHistory(), 0)
                generator.Z = Z = np.random.randint(0, 15, size=(100, 100))
                generator.apply_quantity_filter()
                self.assertTrue(np.array_equal(generator.Z, Z[(Z[:, 0] >= 5) & (Z[:, 0] <= 9)]))

    def test_get_date_from_timestamp(self):
        with CleanEnvironment(lambda _: standard_settings):
            with patch('generator.TraceDataGenerator.__init__') as mock_generator_init:
                mock_generator_init.return_value = None
                self.assertEqual(TraceDataGenerator.get_date_from_timestamp(TraceDataGenerator(), 1615986348000000000), '2021-03-17')
                self.assertEqual(TraceDataGenerator.get_date_from_timestamp(TraceDataGenerator(), 1681761629000000000), '2023-04-17')
                self.assertEqual(TraceDataGenerator.get_date_from_timestamp(TraceDataGenerator(), 1679432332000000000), '2023-03-21')
                self.assertEqual(TraceDataGenerator.get_date_from_timestamp(TraceDataGenerator(), 1621610612000000000), '2021-05-21')
                self.assertEqual(TraceDataGenerator.get_date_from_timestamp(TraceDataGenerator(), 1661451678000000000), '2022-08-25')

    def get_timestamp(self, date_time):
        with CleanEnvironment(lambda _: standard_settings):
            dt = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
            tz = timezone(settings.get('$.finra_timezone'))
            dt = tz.localize(dt)
            return dt.timestamp() * 1e9

    def test_get_number_of_days(self):
        report_dates = np.array([self.get_timestamp('2023-05-24 00:00:00'),
                                 self.get_timestamp('2023-05-24 00:00:01'),
                                 self.get_timestamp('2023-05-24 12:00:00'),
                                 self.get_timestamp('2023-05-24 23:59:59'),
                                 self.get_timestamp('2023-05-23 00:00:00'),
                                 self.get_timestamp('2023-05-23 00:00:01'),
                                 self.get_timestamp('2023-05-23 12:00:00'),
                                 self.get_timestamp('2023-05-23 23:59:59'),
                                 self.get_timestamp('2019-01-01 00:00:00'),
                                 self.get_timestamp('2022-12-31 23:59:59')])
        with CleanEnvironment(lambda _: standard_settings):
            with patch('generator.TraceDataGenerator.__init__') as mock_generator_init, \
                 patch('trade_history.TradeHistory.__init__') as mock_trade_history_init, \
                 patch('generator.TraceDataGenerator.get_rfq_feature_Z_index') as mock_get_rfq_feature_Z_index:
                mock_generator_init.return_value = None
                mock_trade_history_init.return_value = None
                mock_get_rfq_feature_Z_index.return_value = 0
                generator = TraceDataGenerator(TradeHistory(), 0)
                generator.Z = report_dates.reshape((10, 1))
                self.assertEqual(generator.get_number_of_days(), 4)

    def test_apply_liquidity_filter(self):
        with CleanEnvironment(lambda _: standard_settings):
            with patch('generator.TraceDataGenerator.__init__') as mock_generator_init, \
                 patch('trade_history.TradeHistory.__init__') as mock_trade_history_init, \
                 patch('generator.TraceDataGenerator.get_number_of_days') as mock_get_number_of_days, \
                 patch('generator.TraceDataGenerator.get_rfq_feature_Z_index') as mock_get_rfq_feature_Z_index:
                mock_generator_init.return_value = None
                mock_trade_history_init.return_value = None
                mock_get_number_of_days.return_value = 1
                mock_get_rfq_feature_Z_index.return_value = 0
                generator = TraceDataGenerator(TradeHistory(), 0)
                generator.Z = np.array([0, 1, 2, 1, 0, 3, 3, 1, 4, 5, 6, 5, 7, 5, 8, 5, 7, 9, 0, 2]).reshape((20, 1))
                generator.apply_liquidity_filter()
                # 0: 3 times
                # 1: 3 times
                # 2: 2 times
                # 3: 2 times
                # 4: 1 time
                # 5: 4 times
                # 6: 1 time
                # 7: 2 times
                # 8: 1 time
                # 9: 1 time
                # range -- [2, 3]
                # We only keep 0, 1, 2, 3, 5, 7.
                self.assertTrue(np.array_equal(generator.Z, np.array([0, 1, 2, 1, 0, 3, 3, 1, 7, 7, 0, 2]).reshape((12, 1))))


if __name__ == '__main__':
    unittest.main()
