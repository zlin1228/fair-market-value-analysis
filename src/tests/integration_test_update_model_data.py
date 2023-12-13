import sys
sys.path.insert(0, '../')

standard_settings = {
    '$.raw_data_path': 's3://deepmm.test.data/deepmm.data/',
    '$.data_path': 's3://deepmm.test.data/deepmm.parquet/v0.7.zip'}
import settings
settings.override(standard_settings)

import unittest
from unittest.mock import patch

from update_model_data import _update_model_data
from tests.helpers import CleanEnvironment, compare_local_to_s3

def create_side_effect(bucket: str, path: str, remote: str):
    def side_effect(local, upload_bucket, upload_path, extra_args):
        assert bucket == upload_bucket
        assert path == upload_path
        assert len(extra_args) == 0
        assert compare_local_to_s3(local, remote)
    return side_effect

class TestCreateModelData(unittest.TestCase):

    def test_update_model_data_new(self):
        with CleanEnvironment(lambda _: {
                        **standard_settings,
                        # point to a location that doesn't exist so we create instead of update
                        '$.data_path': 's3://deepmm.test.data/deepmm.parquet/DOES_NOT_EXIST.zip'}):
            # patch upload_file so we don't actually upload the resulting files
            with patch('s3.s3_client.upload_file') as upload_file_mock:
                upload_file_mock.side_effect = create_side_effect('deepmm.test.data', 'deepmm.parquet/DOES_NOT_EXIST.zip', 's3://deepmm.test.data/deepmm.parquet/v0.7.zip')
                _update_model_data() # run the update

    def test_update_model_data_existing(self):
        with CleanEnvironment(lambda _: {
                        **standard_settings,
                        # point to a baseline
                        '$.data_path': 's3://deepmm.test.data/deepmm.parquet/v0.7_baseline.zip'}):
            # patch upload_file so we don't actually upload the resulting files
            with patch('s3.s3_client.upload_file') as upload_file_mock:
                upload_file_mock.side_effect = create_side_effect('deepmm.test.data', 'deepmm.parquet/v0.7_baseline.zip', 's3://deepmm.test.data/deepmm.parquet/v0.7_updated.zip')
                _update_model_data() # run the update


if __name__ == '__main__':
    unittest.main()
