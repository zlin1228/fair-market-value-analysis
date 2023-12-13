import sys
sys.path.insert(0, '../')

import unittest
from unittest.mock import patch, MagicMock

import importlib
import pyarrow as pa

import settings
from last_price_model import LastPriceModel
from train_evaluate import get_model_class_from_name, get_model_class, split_data, load_generators, \
                           load_data_and_create_model


class TestGetModelClassFromName(unittest.TestCase):

    def test_valid_name(self):
        # Test that a valid model name returns the correct model class
        name = 'LastPriceModel'
        expected_output = LastPriceModel
        self.assertEqual(get_model_class_from_name(name), expected_output)

    def test_invalid_name(self):
        # Test that an invalid model name raises an exception
        name = 'InvalidModel'
        with self.assertRaises(Exception):
            get_model_class_from_name(name)


class TestGetModelClass(unittest.TestCase):

    def test_valid_model_name(self):
        # Test that a valid model name returns the correct model class and settings
        model_name = 'last_price'
        model_class, model_settings = get_model_class(model_name)
        self.assertEqual(model_class, LastPriceModel)
        self.assertEqual(model_settings, {'class': 'LastPriceModel', 'assumed_spread': 0.29})

    def test_missing_model_name(self):
        # Test that a missing model name raises a ValueError
        model_name = 'invalid_model'
        with self.assertRaises(ValueError):
            get_model_class(model_name)

    @patch('settings.get')
    def test_mocked_settings(self, mock_get):
        # Test that the function retrieves settings from the correct path
        model_name = 'last_price'
        expected_settings = {'class': 'LastPriceModel', 'learning_rate': 0.01}
        mock_get.return_value = {model_name: expected_settings}
        model_class, model_settings = get_model_class(model_name)
        self.assertEqual(model_settings, expected_settings)


class TestSplitData(unittest.TestCase):

    def test_use_date_split(self):
        # Create a mock TraceModel.date_split method that returns dummy data
        mock_date_split = MagicMock(return_value=('train', 'val', 'test'))

        # Create a mock settings dictionary with the use_date_split setting set to True
        mock_settings_get = MagicMock()
        mock_settings_get.return_value = True

        # Set the mock method and mock settings as the methods to be called in the tests
        with unittest.mock.patch('model.TraceModel.date_split', new=mock_date_split):
            with unittest.mock.patch('settings.get', new=mock_settings_get):
                # Call the split_data function with a dummy trace table
                trace = pa.Table.from_pydict({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
                result = split_data(trace)

        # Assert that the mock method was called once with the correct arguments
        mock_date_split.assert_called_once_with(trace)

        # Assert that the function returns the expected result
        self.assertEqual(result, ('train', 'val', 'test'))

    def test_use_percent_split(self):
        # Create a mock TraceModel.percent_split method that returns dummy data
        mock_percent_split = MagicMock(return_value=('train', 'val', 'test'))

        # Create a mock settings dictionary with the use_date_split setting set to False
        mock_settings_get = MagicMock()
        mock_settings_get.return_value = False

        # Set the mock method and mock settings as the methods to be called in the tests
        with unittest.mock.patch('model.TraceModel.percent_split', new=mock_percent_split):
            with unittest.mock.patch('settings.get', new=mock_settings_get):
                # Call the split_data function with a dummy trace table
                trace = pa.Table.from_pydict({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
                result = split_data(trace)

        # Assert that the mock method was called once with the correct arguments
        mock_percent_split.assert_called_once_with(trace)

        # Assert that the function returns the expected result
        self.assertEqual(result, ('train', 'val', 'test'))


class TestLoadGenerators(unittest.TestCase):

    def test_load_generators(self):
        # Create a mock TraceDataGenerator object that returns dummy data
        mock_data_generator = MagicMock()

        # Create a mock split_data method that returns dummy data
        mock_split_data = MagicMock(return_value=('train', 'val', 'test'))

        # Create a mock create_profiler method that returns a dummy profiler object
        mock_profiler = MagicMock()
        mock_profiler.__enter__.return_value = mock_profiler
        mock_profiler.__exit__.return_value = False

        # Create a mock settings dictionary with the sequence_length and batch_size settings
        mock_settings_get = MagicMock()
        sequence_length = 10
        mock_settings_get.side_effect = lambda key: {
            '$.sequence_length': sequence_length,
            '$.batch_size': 32,
        }[key]

        # Set the mock methods and mock settings as the methods to be called in the test
        with unittest.mock.patch('train_evaluate.split_data', new=mock_split_data):
            with unittest.mock.patch('train_evaluate.get_initial_trades'):
                with unittest.mock.patch('train_evaluate.TradeHistory'):
                    with unittest.mock.patch('generator.TraceDataGenerator', new=mock_data_generator):
                        with unittest.mock.patch('settings.get', new=mock_settings_get):
                            result = load_generators()
        mock_split_data.assert_called_once()
        # Assert that the function returns the expected result
        self.assertEqual(result, (
        mock_data_generator.return_value, mock_data_generator.return_value, mock_data_generator.return_value))


class TestLoadDataAndCreateModel(unittest.TestCase):
    def test_load_data_and_create_model(self):
        importlib.reload(settings)
        settings.override({'$.filter_for_evaluation.apply_filter': True})

        # Create a mock TraceModel object that returns dummy data
        mock_model_class = MagicMock()
        mock_model_class.create = MagicMock()
        mock_model_class.create.return_value = None

        # Create a mock TraceDataGenerator object that returns dummy data
        mock_data_generator = MagicMock()
        mock_data_generator.filter_for_evaluation = MagicMock()

        # Create a mock load_generators method that returns dummy data
        mock_load_generators = MagicMock(return_value=(mock_data_generator, mock_data_generator, mock_data_generator))

        # Create a mock get_model_class method that returns the mock model and dummy settings
        mock_get_model_class = MagicMock(return_value=(mock_model_class, {'setting1': 'value1', 'setting2': 'value2'}))

        # Set the mock methods as the methods to be called in the test
        with unittest.mock.patch('train_evaluate.get_model_class', new=mock_get_model_class):
            with unittest.mock.patch('train_evaluate.load_generators', new=mock_load_generators):
                with unittest.mock.patch.object(mock_model_class, 'create'):
                    # Call the load_data_and_create_model function with a dummy model name
                    model_name = 'dummy_model'
                    result = load_data_and_create_model(model_name)

        # Assert that the mock methods were called with the correct arguments
        mock_get_model_class.assert_called_once_with(model_name)
        mock_load_generators.assert_called_once()
        mock_data_generator.filter_for_evaluation.assert_called_once()

        mock_model = mock_model_class()
        mock_model.create.assert_called_once()

        # Assert that the function returns the expected result
        self.assertEqual(result, (mock_model, mock_data_generator))


if __name__ == '__main__':
    unittest.main()
