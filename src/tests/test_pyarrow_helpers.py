import sys
sys.path.insert(0, '../')

import unittest

import pyarrow as pa

from pyarrow_helpers import drop_duplicates, filter_by_value_in_column, sample

class TestDropDuplicates(unittest.TestCase):

    def test_empty(self):
        input = pa.table({'a': []})
        output = pa.table({'a': []})
        self.assertTrue(drop_duplicates(input, ['a']).equals(output))

    def test_single_column_defaults(self):
        input = pa.table({
            'a': [1,1,2,2,3,3],
            'b': [1,2,3,4,5,6]})
        output = pa.table({
            'a': [1,  2,  3],
            'b': [1,  3,  5]})
        self.assertTrue(drop_duplicates(input, ['a']).equals(output))

    def test_single_column_unsorted_first(self):
        input = pa.table({
            'a': [3,1,2,1,3,2],
            'b': [1,2,3,4,5,6]})
        output = pa.table({
            'a': [3,1,2      ],
            'b': [1,2,3      ]})
        self.assertTrue(drop_duplicates(input, ['a'], 'first').equals(output))

    def test_single_column_unsorted_last(self):
        input = pa.table({
            'a': [3,1,2,1,3,2],
            'b': [1,2,3,4,5,6]})
        output = pa.table({
            'a': [      1,3,2],
            'b': [      4,5,6]})
        self.assertTrue(drop_duplicates(input, ['a'], 'last').equals(output))

    def test_multiple_column_unsorted_first(self):
        input = pa.table({
            'a': [1,2,3,1,2,3,1,2,3,1,2,3],
            'b': [1,1,2,2,3,3,1,1,2,2,3,3],
            'c': [1,1,1,2,2,2,3,3,3,1,2,3],
            'd': [1,2,3,1,1,1,2,2,2,3,3,3]})
        output = pa.table({
            'a': [1,  3,1,2,  1,  3,    3],
            'b': [1,  2,2,3,  1,  2,    3],
            'c': [1,  1,2,2,  3,  3,    3],
            'd': [1,  3,1,1,  2,  2,    3]})
        self.assertTrue(drop_duplicates(input, ['b', 'c'], 'first').equals(output))

    def test_multiple_column_unsorted_last(self):
        input = pa.table({
            'a': [1,2,3,1,2,3,1,2,3,1,2,3],
            'b': [1,1,2,2,3,3,1,1,2,2,3,3],
            'c': [1,1,1,2,2,2,3,3,3,1,2,3],
            'd': [1,2,3,1,1,1,2,2,2,3,3,3]})
        output = pa.table({
            'a': [  2,  1,      2,3,1,2,3],
            'b': [  1,  2,      1,2,2,3,3],
            'c': [  1,  2,      3,3,1,2,3],
            'd': [  2,  1,      2,2,3,3,3]})
        self.assertTrue(drop_duplicates(input, ['b', 'c'], 'last').equals(output))

class TestFilterByValueInColumn(unittest.TestCase):

    def test_empty(self):
        table = pa.table({'a': []})
        filter_table = pa.table({'z': []})
        output = pa.table({'a': []})
        self.assertTrue(filter_by_value_in_column(table, 'a', filter_table, ['z']).equals(output))

    def test_invert_empty(self):
        table = pa.table({'a': []})
        filter_table = pa.table({'z': []})
        output = pa.table({'a': []})
        self.assertTrue(filter_by_value_in_column(table, 'a', filter_table, ['z'], True).equals(output))

    def test_null_and_empty_string(self):
        table = pa.table({'a': ['', ' ', '1', None, '   ']})
        filter_table = pa.table({'z': ['', ' ', '1', None, '   ']})
        output = pa.table({'a': ['1']})
        self.assertTrue(filter_by_value_in_column(table, 'a', filter_table, ['z']).equals(output))

    def test_invert_null_and_empty_string(self):
        table = pa.table({'a': ['', ' ', '1', None, '   ']})
        filter_table = pa.table({'z': ['', ' ', '1', None, '   ']})
        output = pa.table({'a': ['', ' ', None, '   ']})
        self.assertTrue(filter_by_value_in_column(table, 'a', filter_table, ['z'], True).equals(output))

    def test_filter_by_one_column(self):
        table = pa.table({
            'a': [1,1,2,2,3,3],
            'b': [1,2,3,4,5,6]})
        filter_table = pa.table({
            'z': [2,4]})
        output = pa.table({
            'a': [  1,  2,   ],
            'b': [  2,  4,   ]})
        self.assertTrue(filter_by_value_in_column(table, 'b', filter_table, ['z']).equals(output))

    def test_invert_filter_by_one_column(self):
        table = pa.table({
            'a': [1,1,2,2,3,3],
            'b': [1,2,3,4,5,6]})
        filter_table = pa.table({
            'z': [2,4]})
        output = pa.table({
            'a': [1,  2,  3,3],
            'b': [1,  3,  5,6]})
        self.assertTrue(filter_by_value_in_column(table, 'b', filter_table, ['z'], True).equals(output))

    def test_filter_by_multiple_column(self):
        table = pa.table({
            'a': [1,1,2,2,3,3],
            'b': [1,2,3,4,5,6]})
        filter_table = pa.table({
            'y': [2,5],
            'z': [2,4]})
        output = pa.table({
            'a': [  1,  2,3, ],
            'b': [  2,  4,5, ]})
        self.assertTrue(filter_by_value_in_column(table, 'b', filter_table, ['y','z']).equals(output))

    def test_invert_filter_by_multiple_column(self):
        table = pa.table({
            'a': [1,1,2,2,3,3],
            'b': [1,2,3,4,5,6]})
        filter_table = pa.table({
            'y': [2,5],
            'z': [2,4]})
        output = pa.table({
            'a': [1,  2,    3],
            'b': [1,  3,    6]})
        self.assertTrue(filter_by_value_in_column(table, 'b', filter_table, ['y','z'], True).equals(output))

class TestSample(unittest.TestCase):

    def test_invalid(self):
        with self.assertRaises(ValueError):
            sample(pa.table({'a': [1,2,3]}), -1)
        with self.assertRaises(ValueError):
            sample(pa.table({'a': [1,2,3]}), 1.00001)

    def test_empty(self):
        input = pa.table({'a': []})
        output = pa.table({'a': []})
        self.assertTrue(sample(input).equals(output))
        self.assertTrue(sample(input, 0).equals(output))
        self.assertTrue(sample(input, 0.5).equals(output))
        self.assertTrue(sample(input, 1).equals(output))

    def test_default(self):
        input = pa.table({'a': [1,2,3]})
        output = pa.table({'a': [1,2,3]})
        self.assertTrue(sample(input).equals(output))

    def test_zero(self):
        input = pa.table({'a': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]})
        self.assertTrue(sample(input, 0).equals(input.slice(0,0)))

    def test_one(self):
        input = pa.table({'a': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]})
        self.assertTrue(sample(input, 1).equals(input))

    def test_some(self):
        input = pa.table({
            'a': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]})
        self.assertTrue(sample(input, 0.1).equals(pa.table({
            'a': [1,                   11,                          ]})))
        self.assertTrue(sample(input, 0.15).equals(pa.table({
            'a': [1,          7,                14,                 ]})))
        self.assertTrue(sample(input, 0.2).equals(pa.table({
            'a': [1,        6,         11,            16,           ]})))
        self.assertTrue(sample(input, 0.25).equals(pa.table({
            'a': [1,      5,      9,         13,         17,        ]})))
        self.assertTrue(sample(input, 0.5).equals(pa.table({
            'a': [1,  3,  5,  7,  9,   11,   13,   15,   17,   19,  ]})))
        self.assertTrue(sample(input, 0.66666).equals(pa.table({
            'a': [1,2,  4,5,  7,8,  10,11,   13,14,   16,17,   19,20]})))
        self.assertTrue(sample(input, 0.825).equals(pa.table({
            'a': [1,2,3,4,5,  7,8,9,10,11,   13,14,15,16,17,   19,20]})))
        self.assertTrue(sample(input, 0.9).equals(pa.table({
            'a': [1,2,3,4,5,6,7,8,9,   11,12,13,14,15,16,17,18,19,  ]})))

if __name__ == '__main__':
    unittest.main()
