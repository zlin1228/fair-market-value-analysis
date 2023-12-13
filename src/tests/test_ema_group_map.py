import sys
sys.path.insert(0, '../')

import pyximport
pyximport.install(language_level=3)
import unittest

from ema_group_map import EmaGroupMap

class TestEmaGroupMap(unittest.TestCase):
    def test_update_ema(self):
        ema_map = EmaGroupMap(alpha=0.1)

        # Test adding a new group and group_id
        ema_map.update_ema(group=1, group_id=2, value=10.0)
        self.assertTrue(ema_map.has_group(1))
        self.assertTrue(ema_map.has_group_id(1, 2))
        self.assertAlmostEqual(ema_map.get_ema(1, 2), 10.0, places=3)

        # Test updating the EMA value of an existing group and group_id
        ema_map.update_ema(group=1, group_id=2, value=20.0)
        self.assertAlmostEqual(ema_map.get_ema(1, 2), 11.0, places=3)

        # Test adding a new group_id to an existing group
        ema_map.update_ema(group=1, group_id=3, value=30.0)
        self.assertTrue(ema_map.has_group_id(1, 3))
        self.assertAlmostEqual(ema_map.get_ema(1, 3), 30.0, places=3)

        # Test adding a new group
        ema_map.update_ema(group=2, group_id=4, value=40.0)
        self.assertTrue(ema_map.has_group(2))
        self.assertTrue(ema_map.has_group_id(2, 4))
        self.assertAlmostEqual(ema_map.get_ema(2, 4), 40.0, places=3)

    def test_get_ema_errors(self):
        ema_map = EmaGroupMap(0.5)
        with self.assertRaises(ValueError) as context:
            r_value = ema_map.get_ema(group=1, group_id=2)
            print(r_value)
        self.assertEqual(str(context.exception), "Group 1 not found.")
        ema_map.update_ema(group=1, group_id=2, value=10.0)
        with self.assertRaises(ValueError) as context:
            r_value = ema_map.get_ema(group=2, group_id=2)
            print(r_value)
        self.assertEqual(str(context.exception), "Group 2 not found.")
        with self.assertRaises(ValueError) as context:
            r_value = ema_map.get_ema(group=1, group_id=3)
            print(r_value)
        self.assertEqual(str(context.exception), "Group ID 3 not found in group 1.")

    def test_has_group(self):
        ema = EmaGroupMap(alpha=0.1)
        self.assertFalse(ema.has_group(1))

        ema.update_ema(1, 1, 10)
        self.assertTrue(ema.has_group(1))

if __name__ == '__main__':
    unittest.main()