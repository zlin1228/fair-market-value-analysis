import sys
sys.path.insert(0, '../')

import numpy as np
import pyarrow as pa
import unittest
from unittest.mock import MagicMock, call, patch

import settings
from generator import TraceDataGenerator
from model import TraceModel, calculate_metrics


class ConcreteTraceModel(TraceModel):
    # Concrete implementation of abstract methods for testing purposes
    def fit(self):
        pass

    def evaluate_batch(self, X_b):
        return self.Y_b_hat.copy()

    def create(self):
        pass

class TestTraceModel(unittest.TestCase):

    def setUp(self):
        patcher = patch('model.get_ordinals')
        self.addCleanup(patcher.stop)
        get_ordinals_mock = patcher.start()
        get_ordinals_mock.return_value = {'buy_sell': pa.array(['S', 'B']), 'side': pa.array(['A', 'D', 'C', 'T'])}
        #                     buy_sell  side
        self.Z_b = np.array([[       1,    2],                  # buy
                             [       0,    2],                  # sell
                             [       1,    0],                  # not buy or sell (same 'buy_sell' as buy)
                             [       0,    0],                  # not buy or sell (same 'buy_sell' as sell)
                             [       2,    2],                  # not buy or sell (same 'side')
                             [       3,    3]]).astype(float)   # not buy or sell (different 'buy_sell' and 'side')
        self.X_b = np.random.rand(self.Z_b.shape[0], self.Z_b.shape[1])
        self.Y_b = np.random.rand(self.Z_b.shape[0], 1)
        self.quantity = np.random.rand(self.Z_b.shape[0])
        self.na_filter = np.array([True] * 12)

        # Mock the generator dependencies
        self.mock_train_generator = MagicMock(spec=TraceDataGenerator)
        self.mock_validation_generator = MagicMock(spec=TraceDataGenerator)
        self.mock_test_generator = MagicMock(spec=TraceDataGenerator)
        self.mock_test_generator.get_Z_b = MagicMock(return_value=self.Z_b.copy())
        self.mock_test_generator.generate_batch_np = MagicMock(return_value=(self.X_b.copy(), self.Y_b.copy()))
        self.mock_test_generator.get_rfq_feature = MagicMock(return_value=self.quantity)
        self.mock_test_generator.get_rfq_feature_Z_index = MagicMock(side_effect = lambda x: ['buy_sell', 'side'].index(x))
        self.mock_test_generator.__len__ = MagicMock(return_value = 10)
        #                                                                    buy_sell  side
        self.mock_test_generator.get_Z = MagicMock(return_value = np.array([[       1,    2],                 # buy
                                                                            [       0,    2],                 # sell
                                                                            [       0,    1],                 # dealer
                                                                            [       0,    0],                 # not
                                                                            [       0,    3],                 # not
                                                                            [       1,    0],                 # not
                                                                            [       1,    1],                 # not
                                                                            [       1,    2],                 # buy
                                                                            [       0,    2],                 # sell
                                                                            [       1,    3],                 # not
                                                                            [       0,    1],                 # dealer
                                                                            [       1,    2]]).astype(float)) # buy

        self.mock_model_settings = {'setting1': 'value1', 'setting2': 'value2'}

        self.trace_model = ConcreteTraceModel(
            'model1',
            self.mock_model_settings,
            self.mock_train_generator,
            self.mock_validation_generator,
            self.mock_test_generator,
        )
        self.trace_model.Y_b_hat = np.random.rand(self.Z_b.shape[0], 1)
        self.trace_model.Y_b_hat[2, 0] = np.nan    # set the third value to nan

    def test_initialization(self):
        self.assertEqual(self.trace_model.model_name, 'model1')
        self.assertEqual(self.trace_model.model_settings, self.mock_model_settings)
        self.assertEqual(self.trace_model.train_generator, self.mock_train_generator)
        self.assertEqual(self.trace_model.validation_generator, self.mock_validation_generator)
        self.assertEqual(self.trace_model.test_generator, self.mock_test_generator)

    def test_get_model_setting(self):
        self.assertEqual(self.trace_model.get_model_setting('setting1'), 'value1')
        self.assertEqual(self.trace_model.get_model_setting('setting2'), 'value2')

    def test_evaluate_batch_index(self):
        na_filter, error, quantity, spread = self.trace_model.evaluate_batch_index('index')
        self.mock_test_generator.get_Z_b.assert_called_once_with('index')
        self.mock_test_generator.generate_batch_np.assert_called()
        self.assertTrue(np.array_equal(self.mock_test_generator.generate_batch_np.call_args[0][0], \
            np.array([0, 2] * 6).reshape((6, 2)).astype(float)))
        self.mock_test_generator.get_rfq_feature.assert_called_once()
        self.assertTrue(np.array_equal(self.mock_test_generator.get_rfq_feature.call_args[0][0], self.X_b))
        self.assertEqual(self.mock_test_generator.get_rfq_feature.call_args[0][1], 'quantity')
        self.mock_test_generator.get_rfq_feature_Z_index.assert_called()
        self.assertEqual(self.mock_test_generator.get_rfq_feature_Z_index.call_args_list[0][0][0], 'buy_sell')
        self.assertEqual(self.mock_test_generator.get_rfq_feature_Z_index.call_args_list[1][0][0], 'side')
        # We set the third value to np.nan.
        self.assertTrue(np.array_equal(na_filter, [True, True, False, True, True, True]))
        self.assertTrue(np.array_equal(error[na_filter], (self.trace_model.Y_b_hat - self.Y_b)[na_filter]))
        self.assertTrue(np.isnan(error[2]))
        self.assertTrue(np.array_equal(quantity, self.quantity))
        self.assertTrue(np.array_equal(spread[na_filter], np.zeros((5, 1))))

    def test_evaluate_batches(self):
        na_filter, error, quantity, spread = np.random.rand(5), np.random.rand(5), np.random.rand(5), np.random.rand(5)
        with patch('model.TraceModel.evaluate_batch_index') as mock_evaluate_batch_index:
            mock_evaluate_batch_index.return_value = (na_filter, error, quantity, spread)
            na_filters, errors, quantities, spreads = self.trace_model.evaluate_batches()
            self.assertTrue(np.array_equal(na_filters, np.tile(na_filter, 10)))
            self.assertTrue(np.array_equal(errors, np.tile(error, 10)))
            self.assertTrue(np.array_equal(quantities, np.tile(quantity, 10)))
            self.assertTrue(np.array_equal(spreads, np.tile(spread, 10)))

    def test_calculate_metrics(self):
        na_filter = np.array([True, True, False, True, True, True, True, True, True, True])
        error     = np.array([   2,   -2,     1,   -4,   14,    1,    8,    4,  -14,   -6]).astype(float)
        quantity  = np.array([   1,    2,     1,    2,    3,    1,    2,    1,    2,    3]).astype(float)
        spread    = np.array([   5,   20,     1,    2,    5,    5,    0,    2,   11,    4]).astype(float)
        metrics = calculate_metrics(na_filter, error, quantity, spread)
        self.assertEqual(metrics['count'], 9)
        self.assertEqual(metrics['mse'], 533 / 9)
        self.assertEqual(metrics['rmse'], np.sqrt(533 / 9))
        self.assertEqual(metrics['mae'], 55 / 9)
        self.assertEqual(metrics['avg_spread'], 54 / 9)
        self.assertEqual(metrics['vw_mse'], 1277 / 17)
        self.assertEqual(metrics['vw_rmse'], np.sqrt(1277 / 17))
        self.assertEqual(metrics['vw_mae'], 123 / 17)
        self.assertEqual(metrics['vw_avg_spread'], 105 / 17)
        tiles = list(range(1, settings.get('$.tiles') + 1))
        self.assertEqual(metrics['error_tiles']['tiles'], tiles)
        self.assertEqual(metrics['error_tiles']['values'], [np.percentile(error[na_filter], percentile) for percentile in tiles])

    def test_get_buy_filter(self):
        self.assertTrue(np.array_equal(self.trace_model.get_buy_filter(self.na_filter), \
            np.array([True, False, False, False, False, False, False, True, False, False, False, True])))

    def test_get_sell_filter(self):
        self.assertTrue(np.array_equal(self.trace_model.get_sell_filter(self.na_filter), \
            np.array([False, True, False, False, False, False, False, False, True, False, False, False])))

    def test_get_dealer_filter(self):
        self.assertTrue(np.array_equal(self.trace_model.get_dealer_filter(self.na_filter), \
            np.array([False, False, True, False, False, False, False, False, False, False, True, False])))

    def test_get_metrics(self):
        with patch('model.calculate_metrics') as mock_calculate_metrics, \
             patch('model.TraceModel.get_buy_filter') as mock_get_buy_filter, \
             patch('model.TraceModel.get_sell_filter') as mock_get_sell_filter, \
             patch('model.TraceModel.get_dealer_filter') as mock_get_dealer_filter:
            mock_get_buy_filter.return_value = 'buy_filter'
            mock_get_sell_filter.return_value = 'sell_filter'
            mock_get_dealer_filter.return_value = 'dealer_filter'
            mock_calculate_metrics.return_value = 'metrics'
            self.assertEqual(self.trace_model.get_metrics('na_filter', 'error', 'quantity', 'spread'), \
                {'overall': 'metrics', 'buy': 'metrics', 'sell': 'metrics', 'dealer': 'metrics'})
            mock_get_buy_filter.assert_called_once_with('na_filter')
            mock_get_sell_filter.assert_called_once_with('na_filter')
            mock_get_dealer_filter.assert_called_once_with('na_filter')
            self.assertEqual(mock_calculate_metrics.call_args_list, [call('na_filter', 'error', 'quantity', 'spread'),
                                                                     call('buy_filter', 'error', 'quantity', 'spread'),
                                                                     call('sell_filter', 'error', 'quantity', 'spread'),
                                                                     call('dealer_filter', 'error', 'quantity', 'spread')])

    def test_evaluate(self):
        with patch('model.TraceModel.evaluate_batches') as mock_evaluate_batches, \
             patch('model.TraceModel.get_metrics') as mock_get_metrics:
            mock_evaluate_batches.return_value = ('na_filter', 'error', 'quantity', 'spread')
            mock_get_metrics.return_value = 'metrics'
            self.assertEqual(self.trace_model.evaluate(), 'metrics')
            mock_evaluate_batches.assert_called_once()
            mock_get_metrics.assert_called_once_with('na_filter', 'error', 'quantity', 'spread')


if __name__ == '__main__':
    unittest.main()
