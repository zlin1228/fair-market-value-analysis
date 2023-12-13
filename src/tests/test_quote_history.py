import sys
sys.path.insert(0, '../')

import importlib
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pyarrow as pa

from grouped_quotes import GroupedQuotes
from quote_history import QuoteHistory
import settings
from load_data import get_initial_quotes
from tests.helpers import CleanEnvironment


class TestQuoteHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        column_list = ['entry_date', 'figi', 'party_id', 'entry_type', 'price', 'quantity']
        cls.sorting_tuples = list(zip(column_list, ["ascending"] * len(column_list)))

    def test_append_consistency(self):
        with CleanEnvironment(lambda _: {"$.data.trades.proportion_of_trades_to_load": 0.0001, '$.enable_bondcliq': True,
                           '$.data_path': 's3://deepmm.test.data/deepmm.parquet/v0.7.zip'}):
            quotes = get_initial_quotes()
            quote_cube1 = QuoteHistory()
            quote_cube1.append(quotes)
            quotes = quotes.sort_by(self.sorting_tuples)
            quote_cube2 = QuoteHistory()
            quote_cube2.append(quotes[:quotes.shape[0]//2])
            quote_cube2.append(quotes[quotes.shape[0]//2:])
            history_1 = quote_cube1.group.get_all_data()
            history_2 = quote_cube2.group.get_all_data()
            self.assertTrue(np.array_equal(history_1, history_2))

    def test_append(self):
        importlib.reload(settings)

        grouped_quotes_mock = Mock()
        grouped_quotes_mock.append = Mock()
        with patch('quote_history.GroupedQuotes') as mock_GroupedQuotes:
            mock_GroupedQuotes.return_value = grouped_quotes_mock

            quote_cube = QuoteHistory()
            quotes = pa.table({'figi':       [  1.0,   1.0,   2.0,   2.0,   1.0,   1.0,   1.0,   1.0,   1.0,   2.0],
                               'party_id':   [  1.0,   2.0,   3.0,   1.0,   2.0,   3.0,   1.0,   2.0,   3.0,   1.0],
                               'entry_type': [  0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0],
                               'entry_date': [  3.0,   6.0,   1.0,   9.0,   5.0,   8.0,   5.0,   3.0,   8.0,   1.0],
                               'price':      [  5.0,   5.0,   6.0,   6.0,   4.0,   4.0,   7.0,   7.0,   3.0,   3.0],
                               'quantity':   [  1.0,   2.0,   1.0,   2.0,   1.0,   2.0,   1.0,   2.0,   1.0,   2.0]})
            quote_cube.append(quotes)

            grouped_quotes_mock.append.assert_called_once()
            self.assertTrue(np.array_equal(grouped_quotes_mock.append.call_args[0][0], np.array([[1., 1., 0., 3., 5., 1.],
                                                                                                 [1., 1., 0., 5., 7., 1.],
                                                                                                 [1., 2., 0., 5., 4., 1.],
                                                                                                 [1., 2., 1., 3., 7., 2.],
                                                                                                 [1., 2., 1., 6., 5., 2.],
                                                                                                 [1., 3., 0., 8., 3., 1.],
                                                                                                 [1., 3., 1., 8., 4., 2.],
                                                                                                 [2., 1., 1., 1., 3., 2.],
                                                                                                 [2., 1., 1., 9., 6., 2.],
                                                                                                 [2., 3., 0., 1., 6., 1.]])))

    def test_generate_batch(self):
        importlib.reload(settings)

        grouped_quotes_mock = Mock()
        grouped_quotes_mock.set_grouped_rows = Mock()
        with patch('quote_history.GroupedQuotes') as mock_GroupedQuotes:
            mock_GroupedQuotes.return_value = grouped_quotes_mock

            quote_cube = QuoteHistory()
            figis = np.array([1, 2, 3])
            execution_dates = np.array([4, 5, 6])
            quote_cube.generate_batch(figis, execution_dates)

            grouped_quotes_mock.set_grouped_rows.assert_called_once()
            X_b = grouped_quotes_mock.set_grouped_rows.call_args[0][0]
            self.assertEqual(X_b.shape[0], 3)
            self.assertEqual(X_b.shape[1], 50)
            self.assertTrue(np.array_equal(figis, grouped_quotes_mock.set_grouped_rows.call_args[0][1]))
            self.assertTrue(np.array_equal(execution_dates, grouped_quotes_mock.set_grouped_rows.call_args[0][2]))


if __name__ == "__main__":
    unittest.main()
