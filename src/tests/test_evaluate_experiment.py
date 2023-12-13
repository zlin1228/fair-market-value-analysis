import sys
sys.path.insert(0, '../')

import gc
import numpy as np
import unittest
from unittest.mock import patch
from types import ModuleType, FunctionType

from evaluate_experiment import get_intersection, getsize, evaluate_experiment


class TestGetIntersection(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(np.array_equal(get_intersection([np.zeros(1), np.zeros(1)]), np.zeros(1)))
        self.assertTrue(np.array_equal(get_intersection([np.zeros(1), np.ones(1)]), np.zeros(1)))
        self.assertTrue(np.array_equal(get_intersection([np.ones(1), np.zeros(1)]), np.zeros(1)))
        self.assertTrue(np.array_equal(get_intersection([np.ones(1), np.ones(1)]), np.ones(1)))

    def test_complex(self):
        self.assertTrue(np.array_equal(get_intersection([np.array([1, 0, 0, 1, 1]), np.array([1, 1, 0, 0, 1])]), np.array([1, 0, 0, 0, 1])))

class TestGetSize(unittest.TestCase):
    def test_simple_object(self):
        obj = 5  # integers are simple objects with known size
        self.assertEqual(getsize(obj), sys.getsizeof(5))

    def test_blacklist_object(self):
        with self.assertRaises(TypeError):
            getsize(type)  # 'type' is in our blacklist
        with self.assertRaises(TypeError):
            getsize(ModuleType)  # 'ModuleType' is in our blacklist
        with self.assertRaises(TypeError):
            getsize(FunctionType)  # 'FunctionType' is in our blacklist

    def test_complex_object(self):
        obj = [1, 2, 3, {"a": 1, "b": 2}, "Hello, World!"]
        # the size of a complex object is not easy to predict, but should be at least the sum of sizes of its elements
        expected_size = sum(getsize(elem) for elem in obj)
        self.assertTrue(getsize(obj) >= expected_size)

class TestEvaluateExperiment(unittest.TestCase):

    @patch('settings.get')
    @patch('evaluate_experiment.load_data_and_create_model')
    @patch('evaluate_experiment.get_intersection')
    @patch('builtins.open')
    @patch('json.dump')
    @patch('evaluate_experiment.getsize')
    def test_evaluate_experiment(self, mock_getsize, mock_dump, mock_open, mock_get_intersection,
                                 mock_load_data_and_create_model,
                                 mock_settings_get):
        # Mock Model
        class MockModel:
            def cleanup(self):
                pass

            def fit(self):
                pass

            def evaluate_batches(self):
                return np.array([True]), 'error', 'quantity', 'spread'

            def get_metrics(self, na_filter, error, quantity, spread):
                return 'metrics'

        # Set up mocks
        mock_settings_get.return_value = True
        mock_load_data_and_create_model.return_value = (MockModel(), None)
        mock_get_intersection.return_value = 'intersection'
        mock_getsize.return_value = 'size'

        # Call the function
        losses = evaluate_experiment(['model1', 'model2'], '../experiments/evaluate.json')

        # Check if the loss data passed to to_csv is correct
        expected_losses = {'model1': 'metrics', 'model2': 'metrics'}
        self.assertEqual(expected_losses, losses)

        # Asserts
        mock_load_data_and_create_model.assert_called_with('model2')
        mock_get_intersection.assert_called_once_with([np.array([True]), np.array([True])])

        # Assert that open has been called with the correct arguments
        mock_open.assert_called_with('../experiments/evaluate.json', 'w')

        # Assert that json.dump has been called with the correct arguments
        mock_dump.assert_called_with(losses, mock_open.return_value.__enter__(), indent=4)

        # Clear memory
        gc.collect()


if __name__ == '__main__':
    unittest.main()
