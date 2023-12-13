import sys
sys.path.insert(0, '../')

standard_settings = {}
import settings
settings.override(standard_settings)

import unittest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pyarrow as pa

from tests.helpers import CleanEnvironment
from grouped_history import combine_trace
from trade_history import TradeHistory

class TestTradeHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.column_list = ['report_date',
                           'ats_indicator',
                           'buy_sell',
                           'figi',
                           'side',
                           'execution_date',
                           'quantity',
                           'price',
                           'yield',
                           'rating',
                           'issuer',
                           'issue_date',
                           'outstanding',
                           'maturity',
                           'coupon',
                           'industry',
                           'sector']
        cls.sorting_tuples = list(zip(cls.column_list, ["ascending"] * len(cls.column_list)))

    def test_combine_trace(self):
        trace = pa.table({'col1': [1, 1, 1, 1],
                          'col2': [10, 10, 10, 10],
                          'quantity': [100, 200, 300, 400],
                          'report_date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']})

        combined = combine_trace(trace)

        self.assertEqual(combined.num_columns, 4)
        self.assertListEqual(combined.column_names, ['col1', 'col2', 'quantity', 'report_date'])
        self.assertListEqual(combined.column(2).to_pylist(), [1000])
        self.assertListEqual(combined.column(3).to_pylist(), ['2022-01-04'])

class TestCreateMapping(unittest.TestCase):
    def setUp(self):
        self.trade_history = type('test', (object,), {})()
        self.trade_history.rfq_labels = ['label1', 'label2']
        self.trade_history.rfq_label_set = set(['label1', 'label2'])
        self.trade_history.remove_features = ['remove1', 'remove2']
        self.trade_history.trades_columns = ['trace1', 'trace2', 'label1', 'remove1']
        self.trade_history.get_feature_Z_index = lambda x: x + "_index"
        self.ordinals = {'figi': pa.array(['figi1', 'figi2'])}

    def test_create_mapping(self):
        with CleanEnvironment(lambda _: {**standard_settings,
                            '$.data.trades.remove_features': ['remove1', 'remove2']}):
            TradeHistory.create_mapping(self.trade_history)
            self.assertEqual(self.trade_history.rfq_label_indices, ['label1_index', 'label2_index'])
            self.assertEqual(self.trade_history.rfq_label_dict, {'label1': 0, 'label2': 1})
            self.assertEqual(self.trade_history.rfq_features, ['trace1', 'trace2'])
            self.assertEqual(self.trade_history.trade_features, ['trace1', 'trace2', 'label1'])
            self.assertEqual(self.trade_history.trade_feature_to_index, {'trace1': 0, 'trace2': 1, 'label1': 2})
            self.assertEqual(self.trade_history.rfq_feature_to_index, {'trace1': 0, 'trace2': 1})

class TestGetFigiTrace(unittest.TestCase):
    def test(self):
        with patch('trade_history.TradeHistory.__init__') as mock_trade_history_init:
            mock_trade_history_init.return_value = None
            trade_history = TradeHistory(None)
            trade_history.grouped = Mock()
            trade_history.grouped.get_last_figi_trades = Mock(return_value='last_figi_trades')
            self.assertEqual(trade_history.get_figi_trace('figi', 'trade_count'), 'last_figi_trades')
            trade_history.grouped.get_last_figi_trades.assert_called_once_with('figi', 'trade_count')

class TestTradeHistory_append(unittest.TestCase):
    def setUp(self):
        self.self_mock = Mock()
        self.self_mock.trades_columns = 'columns'
        self.self_mock.grouped.append_trades = Mock()
        self.selected_trades_mock = Mock()
        self.trades_mock = Mock()
        self.trades_mock.select = Mock(return_value=self.selected_trades_mock)

    def test_append(self):
        self.selected_trades_mock.column_names = self.self_mock.trades_columns
        TradeHistory.append(self.self_mock, self.trades_mock)
        self.trades_mock.select.assert_called_once_with(self.self_mock.trades_columns)
        self.self_mock.grouped.append_trades.assert_called_once_with(self.selected_trades_mock)

    def test_incorrect_columns(self):
        self.selected_trades_mock.column_names = 'something else'
        with self.assertRaises(AssertionError):
            TradeHistory.append(self.self_mock, self.trades_mock)

class TradeHistorySubclass_set_rfq_feature(TradeHistory):
    def __init__(self):
        pass

class TestTradeHistory_set_rfq_feature(unittest.TestCase):
    def setUp(self):
        self.trade_history = TradeHistorySubclass_set_rfq_feature()
        self.X_b = np.array([[1, 2, 3], [4, 5, 6]])
        self.values = np.array([[10, 20], [30, 40]])

    def test_set_rfq_feature_string(self):
        feature_name = "feature1"
        with patch.object(TradeHistory, "get_rfq_feature_index", return_value=0) as mock_index:
            values = self.values[:, 1]
            self.trade_history.set_rfq_feature(self.X_b, feature_name, values)
            mock_index.assert_called_once_with(feature_name)
            np.testing.assert_array_equal(self.X_b, np.array([[20, 2, 3], [40, 5, 6]]))

    def test_set_rfq_feature_list(self):
        feature_name = ["feature1", "feature2"]
        with patch.object(TradeHistory, "get_rfq_feature_index", side_effect=[0, 1]) as mock_index:
            self.trade_history.set_rfq_feature(self.X_b, feature_name, self.values)
            mock_index.assert_has_calls([mock.call(feature_name[0]), mock.call(feature_name[1])])
            np.testing.assert_array_equal(self.X_b, np.array([[10, 20, 3], [30, 40, 6]]))

class TradeHistorySubclass_get_trade_feature_index(TradeHistory):
    def __init__(self):
        self.trade_feature_to_index = MagicMock()

class TestGetTradeFeatureIndex(unittest.TestCase):
    def setUp(self):
        self.trade_history = TradeHistorySubclass_get_trade_feature_index()

    def test_get_index_for_list_of_feature_names(self):
        self.trade_history.trade_feature_to_index.get.return_value = 1
        feature_names = ['feature1', 'feature2']
        expected_output = [1, 1]

        result = self.trade_history.get_trade_feature_index(feature_names)

        self.assertEqual(result, expected_output)
        self.trade_history.trade_feature_to_index.get.assert_has_calls([
            unittest.mock.call('feature1'),
            unittest.mock.call('feature2'),
        ])

    def test_get_index_for_single_feature_name(self):
        self.trade_history.trade_feature_to_index = {'feature1': 1}
        feature_name = 'feature1'
        expected_output = 1

        result = self.trade_history.get_trade_feature_index(feature_name)

        self.assertEqual(result, expected_output)


class TradeHistorySubclass_get_trade_features_total_size(TradeHistory):
    def __init__(self):
	    pass

class TestTradeHistory_get_trade_features_total_size(unittest.TestCase):
	def setUp(self):
		self.trade_history = TradeHistorySubclass_get_trade_features_total_size()

	def test_get_trade_features_total_size(self):
		self.trade_history.get_group_count = MagicMock(return_value=10)
		self.trade_history.get_sequence_length = MagicMock(return_value=20)
		self.trade_history.get_trade_features_count = MagicMock(return_value=5)
		result = self.trade_history.get_trade_features_total_size()
		self.assertEqual(result, 10 * 20 * 5)

class TradeHistorySubclass_get_trade_X_b(TradeHistory):
    def __init__(self):
        pass

class TestTradeHistory_get_trade_X_b(unittest.TestCase):
    def setUp(self):
        self.th = TradeHistorySubclass_get_trade_X_b()
        self.th.get_rfq_features_count = MagicMock(return_value=3)
        self.th.get_group_count = MagicMock(return_value=2)
        self.th.get_sequence_length = MagicMock(return_value=4)
        self.th.get_trade_features_count = MagicMock(return_value=2)

    def test_get_trade_X_b(self):
        X_b = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        result = self.th.get_trade_X_b(X_b)
        expected_result = np.array([[[[4, 5], [6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17], [18, 19]]]])
        self.assertTrue(np.array_equal(result, expected_result))


class TradeHistorySubclass_get_trade_feature_values(TradeHistory):
    def __init__(self, *args, **kwargs):
        self.get_trade_X_b = MagicMock()
        self.get_trade_feature_index = MagicMock()


class TestTradeHistory_get_trade_feature_values(unittest.TestCase):
    def setUp(self):
        self.trade_history = TradeHistorySubclass_get_trade_feature_values()

    def test_returns_correct_value_with_single_feature(self):
        X_b = np.array([[1, 2, 3], [4, 5, 6]])
        X_b_reshaped = X_b.reshape((X_b.shape[0], 1, 1, X_b.shape[1]))
        self.trade_history.get_trade_X_b.return_value = X_b_reshaped
        self.trade_history.get_trade_feature_index.return_value = 1
        result = self.trade_history.get_trade_feature_values(X_b, 'feature_name')
        np.testing.assert_array_equal(result, X_b_reshaped[:, :, :, 1])

    def test_returns_correct_value_with_multiple_features(self):
        X_b = np.array([[1, 2, 3], [4, 5, 6]])
        X_b_reshaped = X_b.reshape((X_b.shape[0], 1, 1, X_b.shape[1]))
        self.trade_history.get_trade_X_b.return_value = X_b_reshaped
        self.trade_history.get_trade_feature_index.return_value = [0, 2]
        result = self.trade_history.get_trade_feature_values(X_b, ['feature_name_1', 'feature_name_2'])
        np.testing.assert_array_equal(result, X_b_reshaped[:, :, :, [0, 2]])


class TradeHistorySubclass_set_trade_feature_values(TradeHistory):
    def __init__(self):
        pass


class TestTradeHistory_set_trade_feature_values(unittest.TestCase):
    def setUp(self):
        self.th = TradeHistorySubclass_set_trade_feature_values()

    @patch.object(TradeHistory, 'get_rfq_features_count', return_value=3)
    @patch.object(TradeHistory, 'get_group_count', return_value=2)
    @patch.object(TradeHistory, 'get_sequence_length', return_value=4)
    @patch.object(TradeHistory, 'get_trade_features_count', return_value=5)
    @patch.object(TradeHistory, 'get_trade_feature_index', return_value=[2,4])
    def test_set_trade_feature_values(self, mock_get_rfq_features_count, mock_get_group_count, mock_get_sequence_length,
                                      mock_get_trade_features_count, mock_get_trade_feature_index):
        X_b = np.zeros((1, 3+2*4*5))
        X_b[:, [0,1,2]] = [1,2,3]
        feature_name = ['price', 'quantity']
        feature_values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).transpose()
        expected_output = np.array([[[[0, 0, 1, 0, 5], [0, 0, 2, 0, 6], [0, 0, 3, 0, 7], [0, 0, 4, 0, 8]],
                                     [[0, 0, 1, 0, 5], [0, 0, 2, 0, 6], [0, 0, 3, 0, 7], [0, 0, 4, 0, 8]]]],
                                   dtype=np.float32).reshape((1,-1))
        final_expected_output = np.zeros((1, 3+2*4*5))
        final_expected_output[:, [0,1,2]] = [1,2,3]
        final_expected_output[:, 3:] = expected_output
        expected_output = final_expected_output

        self.th.set_trade_feature_values(X_b, feature_name, feature_values)
        self.assertTrue(np.array_equal(X_b, expected_output))


class TradeHistorySubclass_get_most_recent_trade_feature_value(TradeHistory):
    def __init__(self):
        pass

class TestGetMostRecentTradeFeatureValue(unittest.TestCase):
    def setUp(self):
        self.trade_history = TradeHistorySubclass_get_most_recent_trade_feature_value()

    @patch('trade_history.TradeHistory.get_trade_feature_values')
    def test_get_most_recent_trade_feature_value(self, mock_get_trade_feature_values):
        X_b = np.array([[1, 2, 3], [4, 5, 6]])
        group_index = 0
        feature_name = 'feature_name'
        mock_get_trade_feature_values.return_value = np.array([[[[1],[4]]]])

        result = self.trade_history.get_most_recent_trade_feature_value(X_b, group_index, feature_name)
        mock_get_trade_feature_values.assert_called_once_with(X_b, feature_name)
        self.assertEqual(result, [[4]])


class TradeHistorySubclass_get_most_recent_trade_feature_value_from_closest_group(TradeHistory):
    def __init__(self):
        pass

class TestTradeHistory_get_most_recent_trade_feature_value_from_closest_group(unittest.TestCase):
    def setUp(self):
        self.trade_history = TradeHistorySubclass_get_most_recent_trade_feature_value_from_closest_group()
        self.trade_history.get_group_count = MagicMock(return_value=2)
        self.trade_history.get_most_recent_trade_feature_value = MagicMock(return_value=np.array([2, 3]))
        self.trade_history.filter_feature_value = MagicMock(return_value=np.array([2, 3]))

    def test_get_most_recent_trade_feature_value_from_closest_group(self):
        X_b = np.array([[1, 2], [3, 4]])
        feature_name = 'price'
        result = self.trade_history.get_most_recent_trade_feature_value_from_closest_group(X_b, feature_name)
        self.assertTrue(np.array_equal(result, np.array([2, 3])))
        self.trade_history.get_most_recent_trade_feature_value.assert_any_call(X_b, 0, 'quantity')
        self.trade_history.get_most_recent_trade_feature_value.assert_any_call(X_b, 1, 'quantity')
        self.trade_history.get_most_recent_trade_feature_value.assert_any_call(X_b, 0, feature_name)
        self.trade_history.get_most_recent_trade_feature_value.assert_any_call(X_b, 1, feature_name)
        self.assertEqual(self.trade_history.filter_feature_value.call_count, 2)


class FilterFeatureValueTestCase(unittest.TestCase):
    def test_filter_feature_value(self):
        # Test case 1: All values are valid
        feature_values = np.array([1, 2, 3, 4, 5])
        group_feature_values = np.array([10, 20, 30, 40, 50])
        sizes = np.array([100, 200, 300, 400, 500])
        expected = feature_values
        result = TradeHistory.filter_feature_value(feature_values, group_feature_values, sizes)
        self.assertTrue(np.array_equal(result, expected))

        # Test case 2: Some values are NaN
        feature_values = np.array([1, 2, np.NaN, 4, np.NaN])
        group_feature_values = np.array([10, 20, 30, 40, 50])
        sizes = np.array([100, 200, 300, 400, 500])
        expected = np.array([1, 2, 30, 4, 50])
        result = TradeHistory.filter_feature_value(feature_values, group_feature_values, sizes)
        self.assertTrue(np.array_equal(result, expected))

        # Test case 3: All values are NaN and size is zero
        feature_values = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        group_feature_values = np.array([10, 20, 30, 40, 50])
        sizes = np.array([0, 0, 0, 0, 0])
        expected = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        result = TradeHistory.filter_feature_value(feature_values, group_feature_values, sizes)
        self.assertTrue(np.all(np.isnan(result)))


class TradeHistorySubclass_get_figi_trade_features_and_count(TradeHistory):
    def __init__(self):
        pass

class TestTradeHistory_get_figi_trade_features_and_count(unittest.TestCase):
    def setUp(self):
        self.th = TradeHistorySubclass_get_figi_trade_features_and_count()
        self.th.get_trade_X_b = MagicMock()
        self.th.get_trade_feature_index = MagicMock()

    def test_get_figi_trade_features_and_count(self):
        X_b = np.array([[[[1, 2, 0], [4, 5, 6]]]])
        feature_names = ['foo', 'bar']
        def get_trade_feature_index(feature_name):
            if feature_name == 'foo':
                return 0
            elif feature_name == 'bar':
                return 1
            elif feature_name == 'quantity':
                return 2

        self.th.get_trade_X_b.return_value = X_b
        self.th.get_trade_feature_index.side_effect = get_trade_feature_index

        features, features_counts, quantities = self.th.get_figi_trade_features_and_count(X_b, feature_names)

        self.th.get_trade_X_b.assert_called_with(X_b)
        self.th.get_trade_feature_index.assert_any_call('quantity')
        self.th.get_trade_feature_index.assert_any_call('foo')
        self.th.get_trade_feature_index.assert_any_call('bar')
        self.assertEqual(features.tolist(), [[[1, 2], [4, 5]]])
        self.assertEqual(features_counts.tolist(), [1])
        self.assertEqual(quantities.tolist(), [[0,6]])

if __name__ == "__main__":
    unittest.main()
