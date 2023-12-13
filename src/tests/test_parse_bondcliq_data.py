import sys
sys.path.insert(0, '../')

import unittest

import numpy as np
import pyarrow as pa

from data_pipeline.parse_bondcliq import _parse_bondcliq_entry_datetime


class TestParseBondcliqData(unittest.TestCase):
    def test_parse_bondcliq_entry_datetime(self):
        table = pa.Table.from_arrays([
            pa.array(['2019-01-01', '2021-03-04', '2022-12-31', '2023-02-20']),
            pa.array(['00:00:00', '12:15:30', '21:35:15', '15:55:29'])
        ], ['date', 'time'])
        parsed_date = _parse_bondcliq_entry_datetime(table, ['date', 'time'])
        self.assertTrue(np.array_equal(parsed_date.to_numpy(), np.array([1546318800000000000,
                                                                         1614878130000000000,
                                                                         1672540515000000000,
                                                                         1676926529000000000]).astype('float64')))


if __name__ == "__main__":
    unittest.main()
