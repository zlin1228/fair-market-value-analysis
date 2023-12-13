import sys
sys.path.insert(0, '../')

import numpy as np
import pyarrow as pa
import unittest

from grouped_ratings import GroupedRatings


class TestGetIndexOfMostRecentRating(unittest.TestCase):
    def setUp(self):
        self.cusip_to_rating = GroupedRatings(pa.table({'cusip'      : pa.array([1, 1, 1, 1, 1, 2, 2, 2, 4, 4]),
                                                        'rating_date': pa.array([3, 3, 4, 5, 5, 1, 2, 4, 4, 4]),
                                                        'rating'     : pa.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}))

    def test_no_cusip_in_table(self):
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=0, execution_date=7), -1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=3, execution_date=4), -1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=5, execution_date=1), -1)

    def test_several_cusips(self):
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=1, execution_date=2), -1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=1, execution_date=4),  2)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=1, execution_date=6),  4)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=2, execution_date=0), -1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=2, execution_date=1),  0)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=2, execution_date=2),  1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=2, execution_date=3),  1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=2, execution_date=4),  2)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=2, execution_date=5),  2)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=4, execution_date=3), -1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=4, execution_date=4),  1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=4, execution_date=5),  1)

    def test_same_ratings(self):
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=1, execution_date=3), 1)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=1, execution_date=5), 4)
        self.assertEqual(self.cusip_to_rating.get_index_of_most_recent_rating(cusip=4, execution_date=4), 1)


class TestGetMostRecentRatings(unittest.TestCase):
    def test(self):
        cusip_to_rating = GroupedRatings(pa.table({'cusip'      : pa.array([1, 1, 1, 1, 1, 2, 2, 2, 4, 4]),
                                                   'rating_date': pa.array([3, 3, 4, 5, 5, 1, 2, 4, 4, 4]),
                                                   'rating'     : pa.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}))
        cusips = np.array([0, 3, 5, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 1, 1, 4])
        execution_dates = np.array([7, 4, 1, 2, 4, 6, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 5, 4])
        rating_dates = np.zeros(18, dtype=int)
        ratings = np.zeros(18, dtype=int)
        cusip_to_rating.get_most_recent_ratings(cusips, execution_dates, rating_dates, ratings, -1)
        self.assertTrue(np.array_equal(rating_dates, np.array([7, 4, 1, 2, 4, 5, 0, 1, 2, 2, 4, 4, 3, 4, 4, 3, 5, 4])))
        self.assertTrue(np.array_equal(ratings, np.array([-1, -1, -1, -1, 2, 4, -1, 5, 6, 6, 7, 7, -1, 9, 9, 1, 4, 9])))


class TestJoinWithCusipToRating(unittest.TestCase):
    def test(self):
        cusip_to_rating = GroupedRatings(pa.table({'cusip'      : pa.array([1, 1, 1, 1, 1, 2, 2, 2, 4, 4]),
                                                   'rating_date': pa.array([3, 3, 4, 5, 5, 1, 2, 4, 4, 4]),
                                                   'rating'     : pa.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}))
        table = pa.table({'cusip': [0, 3, 5, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 1, 1, 4],
                          'execution_date': [7, 4, 1, 2, 4, 6, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 5, 4]})
        table = cusip_to_rating.join_with_cusip_to_rating(table, -1)
        self.assertTrue(table.equals(pa.table({'cusip': pa.array([0, 3, 5, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 1, 1, 4]),
                                               'execution_date': pa.array([7, 4, 1, 2, 4, 6, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 5, 4]),
                                               'rating_date': pa.array([7, 4, 1, 2, 4, 5, 0, 1, 2, 2, 4, 4, 3, 4, 4, 3, 5, 4]),
                                               'rating': pa.array([-1, -1, -1, -1, 2, 4, -1, 5, 6, 6, 7, 7, -1, 9, 9, 1, 4, 9]).cast(pa.int32())})))


if __name__ == '__main__':
    unittest.main()