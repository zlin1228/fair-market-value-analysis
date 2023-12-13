import sys
sys.path.insert(0, '../')

import unittest
from unittest.mock import MagicMock, patch

import pyarrow as pa

from model import TraceModel

class TestTraceModel(unittest.TestCase):
    def test_percent_split(self):
        # Prepare mock TraceModel.get_split_index that returns different values for different input parameters
        mock_split_index = MagicMock()
        mock_split_index.side_effect = lambda trace, name: 3 if name == 'training_data_split' else 4
        with patch('model.TraceModel.get_split_index', new=mock_split_index):
            # Prepare test data
            table = pa.Table.from_pydict({'x': [1, 2, 3, 4, 5], 'y': [6, 7, 8, 9, 10]})

            # Call the method under test
            training_data, validation_data, test_data = TraceModel.percent_split(table)

            # Verify the results
            self.assertEqual(training_data.num_rows, 3)
            self.assertEqual(len(validation_data), 1)
            self.assertEqual(len(test_data), 1)
            self.assertEqual(training_data.num_rows + validation_data.num_rows + test_data.num_rows, table.num_rows)
            self.assertEqual(table.slice(0, training_data.num_rows).to_pydict(), training_data.to_pydict())
            self.assertEqual(table.slice(training_data.num_rows,
                                         validation_data.num_rows).to_pydict(),
                             validation_data.to_pydict())
            self.assertEqual(table.slice(training_data.num_rows + validation_data.num_rows,
                                         test_data.num_rows).to_pydict(),
                             test_data.to_pydict())

    def test_get_date_ranges_from_trace(self):
        # Prepare test data
        trace = pa.Table.from_pydict({
            'report_date': ['2019-01-01', '2020-04-30', '2021-03-21'],
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        date_ranges = [
            {'start': '2018-12-31', 'end': '2020-03-20'},
            {'start': '2020-04-17', 'end': '2022-03-02'}
        ]

        # Call the method under test
        result = TraceModel.get_date_ranges_from_trace(date_ranges, trace)

        # Verify the results
        self.assertIsNotNone(result)
        self.assertEqual(result.num_rows, 3)
        self.assertEqual(result.num_columns, 3)
        self.assertListEqual(result.column_names, ['report_date', 'x', 'y'])
        self.assertListEqual(result['report_date'].to_pylist(), trace['report_date'].to_pylist())
        self.assertListEqual(result['x'].to_pylist(), [1, 2, 3])
        self.assertListEqual(result['y'].to_pylist(), [4, 5, 6])

        # Test getting a subset of the data.
        date_ranges = [
            {'start': '2018-12-31', 'end': '2020-03-20'},
        ]
        result = TraceModel.get_date_ranges_from_trace(date_ranges, trace)
        self.assertIsNotNone(result)
        self.assertEqual(result.num_rows, 1)
        self.assertEqual(result.num_columns, 3)
        self.assertListEqual(result.column_names, ['report_date', 'x', 'y'])
        self.assertListEqual(result['report_date'].to_pylist(), ['2019-01-01'])
        self.assertListEqual(result['x'].to_pylist(), [1])
        self.assertListEqual(result['y'].to_pylist(), [4])

    def test_date_split(self):
        mock_get_date_ranges = MagicMock(
            return_value=pa.Table.from_pydict({'x': [1, 2], 'y': [4, 5], 'report_date': ['2019-01-01', '2020-04-30']}))
        with patch('model.TraceModel.get_date_ranges_from_trace', new=mock_get_date_ranges):
            # Prepare test data
            batch_size = 2
            data_cube = pa.Table.from_pydict({'x': [1, 2, 3, 4], 'y': [4, 5, 6, 7],
                                              'report_date': ['2019-01-01', '2020-04-30', '2021-03-21', '2021-06-21']})

            # Call the method under test
            train_data, validation_data, test_data = TraceModel.date_split(data_cube)

            # Verify the results
            self.assertEqual(train_data.num_rows, 2)
            self.assertEqual(validation_data.num_rows, 2)
            self.assertEqual(test_data.num_rows, 2)
            self.assertListEqual(train_data['x'].to_pylist(), [1, 2])
            self.assertListEqual(train_data['y'].to_pylist(), [4, 5])
            self.assertListEqual(validation_data['x'].to_pylist(), [1, 2])
            self.assertListEqual(validation_data['y'].to_pylist(), [4, 5])
            self.assertListEqual(test_data['x'].to_pylist(), [1, 2])
            self.assertListEqual(test_data['y'].to_pylist(), [4, 5])


if __name__ == '__main__':
    unittest.main()
