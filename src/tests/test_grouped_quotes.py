import sys
sys.path.insert(0, '../')

import pyximport
pyximport.install(language_level=3)
import unittest

import numpy as np

from grouped_quotes import GroupedQuotes


class FindPreviousTestCase(unittest.TestCase):
    # We only use __find_previous for the training time
    @classmethod
    def setUpClass(cls):
        #                   figi  party_id  entry_type  entry_date  price  quantity
        quotes = np.array([[   1,        1,          1,          1,     2,        4],
                           [   1,        1,          1,          2,     3,        1],
                           [   1,        1,          1,          3,     3,        6],
                           [   1,        1,          1,          3,     4,        5],
                           [   1,        1,          1,          4,     8,        3],
                           [   1,        1,          1,          5,     3,        5],
                           [   1,        1,          1,          5,     6,        3],
                           [   1,        1,          1,          5,     8,        7],
                           [   1,        1,          1,          6,     3,        7],
                           [   1,        1,          1,          7,     5,        1]]).astype(float)
        cls.group = GroupedQuotes(dealer_window=10, enable_inference=0, number_of_saved_features=3, number_of_used_features=5,
                                  index_of_entry_date=0, index_of_figi=0, index_of_party_id=1, index_of_entry_type=2)
        cls.group.append(quotes)

    # all values are greater than input
    def test_before_beginning(self):
        self.assertEqual(self.group.find_previous(1, 1, 1, 0.0), -1)

    # all values are greater than or equal to input
    def test_beginning(self):
        self.assertEqual(self.group.find_previous(1, 1, 1, 1.0), -1)

    # some values are less than input
    def test_middle(self):
        self.assertEqual(self.group.find_previous(1, 1, 1, 2.5), 1)

    # some values are less than or equal to input
    def test_middle_tie(self):
        self.assertEqual(self.group.find_previous(1, 1, 1, 6.0), 7)

    # input is equal to the last value in the array
    def test_ending(self):
        self.assertEqual(self.group.find_previous(1, 1, 1, 7.0), 8)

    # all values are less than input
    def test_after_ending(self):
        self.assertEqual(self.group.find_previous(1, 1, 1, 8.0), 9)

    # no feature vector with the given figi, party_id, and entry_type
    def test_empty(self):
        self.assertEqual(self.group.find_previous(1, 2, 3, 4.0), -1)


class AppendTestCase(unittest.TestCase):
    def setUp(self):
        #                          figi  party_id  entry_type  entry_date  price  quantity
        self.quotes_1 = np.array([[   1,        2,          0,          5,     3,        5],
                                  [   1,        2,          1,          1,     2,        4],
                                  [   1,        2,          1,          3,     3,        6],
                                  [   1,        2,          1,          3,     4,        5],
                                  [   1,        2,          1,          5,     8,        7],
                                  [   2,        1,          0,          2,     3,        1],
                                  [   2,        1,          0,          4,     8,        3],
                                  [   2,        1,          0,          5,     6,        3],
                                  [   2,        1,          0,          6,     3,        7],
                                  [   2,        1,          1,          7,     5,        1]]).astype(float)
        self.quotes_2 = np.array([[   1,        1,          0,         12,     8,        7],
                                  [   1,        2,          0,          9,     3,        6],
                                  [   1,        2,          0,         11,     4,        5],
                                  [   1,        2,          1,          8,     2,        4],
                                  [   1,        2,          1,          9,     4,        5],
                                  [   1,        2,          1,         11,     6,        7],
                                  [   2,        1,          0,          8,     3,        1],
                                  [   2,        1,          0,         10,     8,        3],
                                  [   2,        1,          0,         12,     2,        7],
                                  [   2,        1,          1,         11,     2,        3]]).astype(float)
        self.quotes_3 = np.array([[   1,        1,          0,         18,     3,        5],
                                  [   1,        1,          1,         18,     6,        3],
                                  [   1,        2,          0,         16,     4,        5],
                                  [   1,        2,          1,         13,     2,        4],
                                  [   1,        2,          1,         20,     8,        7],
                                  [   2,        1,          0,         17,     8,        3],
                                  [   2,        1,          1,         21,     3,        7],
                                  [   2,        2,          0,         15,     3,        1],
                                  [   2,        2,          0,         16,     3,        6],
                                  [   2,        2,          1,         22,     5,        1]]).astype(float)

    # test for inference time
    def test_for_inference(self):
        group = GroupedQuotes(dealer_window=10, enable_inference=1, number_of_saved_features=3, number_of_used_features=5,
                              index_of_entry_date=0, index_of_figi=0, index_of_party_id=1, index_of_entry_type=2)
        group.append(self.quotes_1)
        #                                                     figi  party_id  entry_type              entry_date  price  quantity
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          0), np.array([[         5,     3,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          1), np.array([[         5,     8,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          0), np.array([[         6,     3,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          1), np.array([[         7,     5,        1]]).astype(float)))
        group.append(self.quotes_2)
        #                                                     figi  party_id  entry_type              entry_date  price  quantity
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          0), np.array([[        11,     4,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          1), np.array([[        11,     6,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          0), np.array([[        12,     2,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          1), np.array([[        11,     2,        3]]).astype(float)))
        group.append(self.quotes_3)
        #                                                     figi  party_id  entry_type              entry_date  price  quantity
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        1,          0), np.array([[        18,     3,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        1,          1), np.array([[        18,     6,        3]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          0), np.array([[        16,     4,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          1), np.array([[        20,     8,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          0), np.array([[        17,     8,        3]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          1), np.array([[        21,     3,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        2,          0), np.array([[        16,     3,        6]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        2,          1), np.array([[        22,     5,        1]]).astype(float)))

    # test for training time
    def test_for_training(self):
        group = GroupedQuotes(dealer_window=10, enable_inference=0, number_of_saved_features=3, number_of_used_features=5,
                              index_of_entry_date=0, index_of_figi=0, index_of_party_id=1, index_of_entry_type=2)
        group.append(self.quotes_1)
        #                                                     figi  party_id  entry_type              entry_date  price  quantity
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          0), np.array([[         5,     3,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          1), np.array([[         1,     2,        4],
                                                                                                     [         3,     3,        6],
                                                                                                     [         3,     4,        5],
                                                                                                     [         5,     8,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          0), np.array([[         2,     3,        1],
                                                                                                     [         4,     8,        3],
                                                                                                     [         5,     6,        3],
                                                                                                     [         6,     3,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          1), np.array([[         7,     5,        1]]).astype(float)))
        group.append(self.quotes_2)
        #                                                     figi  party_id  entry_type              entry_date  price  quantity
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          0), np.array([[         5,     3,        5],
                                                                                                     [         9,     3,        6],
                                                                                                     [        11,     4,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          1), np.array([[         1,     2,        4],
                                                                                                     [         3,     3,        6],
                                                                                                     [         3,     4,        5],
                                                                                                     [         5,     8,        7],
                                                                                                     [         8,     2,        4],
                                                                                                     [         9,     4,        5],
                                                                                                     [        11,     6,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          0), np.array([[         2,     3,        1],
                                                                                                     [         4,     8,        3],
                                                                                                     [         5,     6,        3],
                                                                                                     [         6,     3,        7],
                                                                                                     [         8,     3,        1],
                                                                                                     [        10,     8,        3],
                                                                                                     [        12,     2,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          1), np.array([[         7,     5,        1],
                                                                                                     [        11,     2,        3]]).astype(float)))
        group.append(self.quotes_3)
        #                                                     figi  party_id  entry_type              entry_date  price  quantity
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        1,          0), np.array([[        12,     8,        7],
                                                                                                     [        18,     3,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        1,          1), np.array([[        18,     6,        3]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          0), np.array([[         5,     3,        5],
                                                                                                     [         9,     3,        6],
                                                                                                     [        11,     4,        5],
                                                                                                     [        16,     4,        5]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   1,        2,          1), np.array([[         1,     2,        4],
                                                                                                     [         3,     3,        6],
                                                                                                     [         3,     4,        5],
                                                                                                     [         5,     8,        7],
                                                                                                     [         8,     2,        4],
                                                                                                     [         9,     4,        5],
                                                                                                     [        11,     6,        7],
                                                                                                     [        13,     2,        4],
                                                                                                     [        20,     8,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          0), np.array([[         2,     3,        1],
                                                                                                     [         4,     8,        3],
                                                                                                     [         5,     6,        3],
                                                                                                     [         6,     3,        7],
                                                                                                     [         8,     3,        1],
                                                                                                     [        10,     8,        3],
                                                                                                     [        12,     2,        7],
                                                                                                     [        17,     8,        3]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        1,          1), np.array([[         7,     5,        1],
                                                                                                     [        11,     2,        3],
                                                                                                     [        21,     3,        7]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        2,          0), np.array([[        15,     3,        1],
                                                                                                     [        16,     3,        6]]).astype(float)))
        self.assertTrue(np.array_equal(group.get_grouped_data(   2,        2,          1), np.array([[        22,     5,        1]]).astype(float)))


class SetGroupedRowsTestCase(unittest.TestCase):
    def setUp(self):
        #                        figi  party_id  entry_type  entry_date  price  quantity
        self.quotes = np.array([[   1,        1,          0,         12,     8,        7],
                                [   1,        1,          0,         18,     3,        5],
                                [   1,        1,          1,         11,     2,        3],
                                [   1,        1,          1,         18,     6,        3],
                                [   1,        2,          0,          5,     3,        5],
                                [   1,        2,          0,          9,     3,        6],
                                [   1,        2,          0,         11,     4,        5],
                                [   1,        2,          0,         16,     4,        5],
                                [   1,        2,          1,          1,     2,        4],
                                [   1,        2,          1,          3,     3,        6],
                                [   1,        2,          1,          3,     4,        5],
                                [   1,        2,          1,          5,     8,        7],
                                [   1,        2,          1,          8,     2,        4],
                                [   1,        2,          1,          9,     4,        5],
                                [   1,        2,          1,         11,     6,        7],
                                [   1,        2,          1,         13,     2,        4],
                                [   1,        2,          1,         20,     8,        7],
                                [   2,        1,          0,          2,     3,        1],
                                [   2,        1,          0,          4,     8,        3],
                                [   2,        1,          0,          5,     6,        3],
                                [   2,        1,          0,          6,     3,        7],
                                [   2,        1,          0,          8,     3,        1],
                                [   2,        1,          0,         10,     8,        3],
                                [   2,        1,          0,         12,     2,        7],
                                [   2,        1,          0,         17,     8,        3],
                                [   2,        1,          1,          7,     5,        1],
                                [   2,        1,          1,         21,     3,        7],
                                [   2,        2,          0,         15,     3,        1],
                                [   2,        2,          0,         16,     3,        6],
                                [   2,        2,          1,         22,     5,        1],]).astype(float)
        self.figis = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0])
        self.execution_dates = np.array([2.0, 16.0, 12.0, 20.0, 25.0, 17.0, 21.0, 23.0, 24.0, 19.0])
        # shape of X_b - (len(figis), dealer_window * number_of_used_features)
        self.X_b = np.zeros((10, 15)).astype(float)

    def test_for_inference(self):
        group = GroupedQuotes(dealer_window=3, enable_inference=1, number_of_saved_features=3, number_of_used_features=5,
                              index_of_entry_date=0, index_of_figi=0, index_of_party_id=1, index_of_entry_type=2)
        group.append(self.quotes)
        group.set_grouped_rows(X_b=self.X_b, figis=self.figis, execution_dates=self.execution_dates)
        #                            entry_date  price  quantity  party_id  entry_type  entry_date  price  quantity  party_id  entry_type  entry_date  price  quantity  party_id  entry_type
        expected_result = np.array([[         0,     0,        0,        0,          0,          0,     0,        0,        0,          0,          0,     0,        0,        0,          0],
                                    [         0,     0,        0,        0,          0,          0,     0,        0,        0,          0,          0,     0,        0,        0,          0],
                                    [         0,     0,        0,        0,          0,          0,     0,        0,        0,          0,          0,     0,        0,        0,          0],
                                    [        16,     4,        5,        2,          0,         18,     3,        5,        1,          0,         18,     6,        3,        1,          1],
                                    [        17,     8,        3,        1,          0,         21,     3,        7,        1,          1,         22,     5,        1,        2,          1],
                                    [         0,     0,        0,        0,          0,          0,     0,        0,        0,          0,         16,     3,        6,        2,          0],
                                    [         0,     0,        0,        0,          0,         16,     3,        6,        2,          0,         17,     8,        3,        1,          0],
                                    [        18,     3,        5,        1,          0,         18,     6,        3,        1,          1,         20,     8,        7,        2,          1],
                                    [        17,     8,        3,        1,          0,         21,     3,        7,        1,          1,         22,     5,        1,        2,          1],
                                    [        16,     4,        5,        2,          0,         18,     3,        5,        1,          0,         18,     6,        3,        1,          1]]).astype(float)
        self.assertTrue(np.array_equal(self.X_b, expected_result))

    def test_for_training(self):
        group = GroupedQuotes(dealer_window=3, enable_inference=0, number_of_saved_features=3, number_of_used_features=5,
                              index_of_entry_date=0, index_of_figi=0, index_of_party_id=1, index_of_entry_type=2)
        group.append(self.quotes)
        group.set_grouped_rows(X_b=self.X_b, figis=self.figis, execution_dates=self.execution_dates)
        #                            entry_date  price  quantity  party_id  entry_type  entry_date  price  quantity  party_id  entry_type  entry_date  price  quantity  party_id  entry_type
        expected_result = np.array([[         0,     0,        0,        0,          0,          0,     0,        0,        0,          0,          1,     2,        4,        2,          1],
                                    [         7,     5,        1,        1,          1,         12,     2,        7,        1,          0,         15,     3,        1,        2,          0],
                                    [        11,     2,        3,        1,          1,         11,     4,        5,        2,          0,         11,     6,        7,        2,          1],
                                    [        16,     4,        5,        2,          0,         18,     3,        5,        1,          0,         18,     6,        3,        1,          1],
                                    [        17,     8,        3,        1,          0,         21,     3,        7,        1,          1,         22,     5,        1,        2,          1],
                                    [         7,     5,        1,        1,          1,         12,     2,        7,        1,          0,         16,     3,        6,        2,          0],
                                    [         7,     5,        1,        1,          1,         16,     3,        6,        2,          0,         17,     8,        3,        1,          0],
                                    [        18,     3,        5,        1,          0,         18,     6,        3,        1,          1,         20,     8,        7,        2,          1],
                                    [        17,     8,        3,        1,          0,         21,     3,        7,        1,          1,         22,     5,        1,        2,          1],
                                    [        16,     4,        5,        2,          0,         18,     3,        5,        1,          0,         18,     6,        3,        1,          1]]).astype(float)
        group.set_grouped_rows(X_b=self.X_b, figis=self.figis, execution_dates=self.execution_dates)
        self.assertTrue(np.array_equal(self.X_b, expected_result))


if __name__ == '__main__':
    unittest.main()
