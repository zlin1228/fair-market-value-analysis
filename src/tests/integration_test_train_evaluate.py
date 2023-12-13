import sys
sys.path.insert(0, '../')

standard_settings = {
    '$.data_path': 's3://deepmm.test.data/deepmm.parquet/v0.7.zip',
    '$.keras_models.load': False,
    '$.keras_models.use_gpu': False, # not all of our test environments have gpus
    '$.batch_size': 1024,
    '$.keras_models.max_batch_size': 1024}
import settings
settings.override(standard_settings)

import os
import re
import time
import unittest

import git

from tests.helpers import CleanEnvironment, compare_s3_files
import s3
import train_evaluate

zip_re = re.compile(r'^.*/(.*?).zip$')

def get_reference_model(reference_model_name, model_name):
    """Starts at the current git commit and traverses up the git commit history looking for a reference model
        matching the hash of the git commit and returns the first one it finds, or None if it doesn't find one."""
    s3_prefix = f's3://deepmm.test.data/integration_test_models/{reference_model_name}/{model_name}/'
    paths = s3.get_all_paths_with_prefix(s3_prefix)
    hashes = {zip_re.match(m).group(1) for m in (p for p in paths if zip_re.match(p))}
    repo = git.Repo('.', search_parent_directories=True)
    for commit in repo.iter_commits():
        if commit.hexsha in hashes:
            return f'{s3_prefix}{commit}.zip'
    return None

class TestTrainEvaluate(unittest.TestCase):

    def setUp(self):
        # Training automatically saves to S3 so keep track of files we need to clean up after each test
        self.s3_urls_to_delete = set()

    def tearDown(self):
        # Clean up any S3 files created during the test
        for s3url in self.s3_urls_to_delete:
            s3.delete_object(s3url)

    def test_residual_nn_percentage(self):
        temp_id = time.time_ns() # just use time in ns since we are seeding rng during tests
        model_dir = f'deepmm.temp/integration_test_models/test_residual_nn_percentage/{temp_id}'
        with CleanEnvironment(lambda temp_dir: {
                        **standard_settings,
                        '$.environment_settings.development.model_bucket': model_dir,
                        '$.use_date_split': False,
                        '$.training_data_split': 0.5,
                        '$.validation_data_split': 0.75,
                        '$.data.trades.proportion_of_trades_to_load': 0.05,
                        '$.models.residual_nn_integration_test.history_filepath': os.path.join(temp_dir, 'history')}):
            train_evaluate.train_and_evaluate('residual_nn_integration_test')
            for s3url in s3.get_all_with_prefix(f's3://{model_dir}').keys():
                reference_s3url = get_reference_model('test_residual_nn_percentage', 'residual_nn_integration_test')
                self.assertTrue(compare_s3_files(s3url, reference_s3url))
                self.s3_urls_to_delete.add(s3url)

    def test_residual_nn_date_range(self):
        temp_id = time.time_ns() # just use time in ns since we are seeding rng during tests
        model_dir = f'deepmm.temp/integration_test_models/test_residual_nn_date_range/{temp_id}'
        with CleanEnvironment(lambda temp_dir: {
                        **standard_settings,
                        '$.environment_settings.development.model_bucket': model_dir,
                        '$.use_date_split': True,
                        '$.training_date_ranges[0].start': '2019-01-01',
                        '$.training_date_ranges[0].end': '2019-09-01',
                        '$.training_date_ranges[1].start': '2020-08-01',
                        '$.training_date_ranges[1].end': '2022-03-02',
                        '$.validation_date_ranges[0].start': '2020-01-01',
                        '$.validation_date_ranges[0].end': '2020-08-01',
                        '$.validation_date_ranges[1].start': '2022-03-02',
                        '$.validation_date_ranges[1].end': '2022-08-01',
                        '$.test_date_ranges[0].start': '2019-09-01',
                        '$.test_date_ranges[0].end': '2020-01-01',
                        '$.test_date_ranges[1].start': '2022-08-01',
                        '$.test_date_ranges[1].end': '2023-03-01',
                        '$.data.trades.proportion_of_trades_to_load': 0.05,
                        '$.models.residual_nn_integration_test.history_filepath': os.path.join(temp_dir, 'history')}):
            train_evaluate.train_and_evaluate('residual_nn_integration_test')
            for s3url in s3.get_all_with_prefix(f's3://{model_dir}').keys():
                reference_s3url = get_reference_model('test_residual_nn_date_range', 'residual_nn_integration_test')
                self.assertTrue(compare_s3_files(s3url, reference_s3url))
                self.s3_urls_to_delete.add(s3url)

    def test_residual_nn_date_range_resume(self):
        common_settings = {
                        '$.use_date_split': True,
                        '$.training_date_ranges[0].start': '2019-01-01',
                        '$.training_date_ranges[0].end': '2019-09-01',
                        '$.training_date_ranges[1].start': '2020-08-01',
                        '$.training_date_ranges[1].end': '2022-03-02',
                        '$.validation_date_ranges[0].start': '2020-01-01',
                        '$.validation_date_ranges[0].end': '2020-08-01',
                        '$.validation_date_ranges[1].start': '2022-03-02',
                        '$.validation_date_ranges[1].end': '2022-08-01',
                        '$.test_date_ranges[0].start': '2019-09-01',
                        '$.test_date_ranges[0].end': '2020-01-01',
                        '$.test_date_ranges[1].start': '2022-08-01',
                        '$.test_date_ranges[1].end': '2023-03-01',
                        '$.data.trades.proportion_of_trades_to_load': 0.05}
        # train the first epochs
        temp_id = time.time_ns() # just use time in ns since we are seeding rng during tests
        model_dir = f'deepmm.temp/integration_test_models/test_residual_nn_date_range_partial/{temp_id}'
        with CleanEnvironment(lambda temp_dir: {
                        **standard_settings,
                        **common_settings,
                        '$.environment_settings.development.model_bucket': model_dir,
                        '$.models.residual_nn_integration_test.epochs': 3,
                        '$.models.residual_nn_integration_test.history_filepath': os.path.join(temp_dir, 'history')}):
            train_evaluate.train_and_evaluate('residual_nn_integration_test')
            for s3url in s3.get_all_with_prefix(f's3://{model_dir}').keys():
                reference_s3url = get_reference_model('test_residual_nn_date_range_partial', 'residual_nn_integration_test')
                self.assertTrue(compare_s3_files(s3url, reference_s3url))
                self.s3_urls_to_delete.add(s3url)
        # finish training
        with CleanEnvironment(lambda temp_dir: {
                        **standard_settings,
                        **common_settings,
                        '$.environment_settings.development.model_bucket': model_dir,
                        '$.models.residual_nn_integration_test.epochs': 2,
                        '$.models.residual_nn_integration_test.history_filepath': os.path.join(temp_dir, 'history'),
                        '$.keras_models.load': True}):
            train_evaluate.train_and_evaluate('residual_nn_integration_test')
            for s3url in s3.get_all_with_prefix(f's3://{model_dir}').keys():
                reference_s3url = get_reference_model('test_residual_nn_date_range_resume', 'residual_nn_integration_test')
                self.assertTrue(compare_s3_files(s3url, reference_s3url))
                self.s3_urls_to_delete.add(s3url)


if __name__ == '__main__':
    unittest.main()
