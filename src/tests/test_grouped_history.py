import sys
sys.path.insert(0, '../')

import numpy as np
import pyarrow as pa
import unittest
from unittest.mock import patch, Mock

import pyximport
pyximport.install(language_level=3)
from grouped_history import GroupedEntry, GroupedEntryTest, GroupedHistory, TradeFeatureArray


class GroupedEntryTestCase(unittest.TestCase):
    def setUp(self):
        self.grouped = GroupedEntryTest()
        #                                report_date  figi  price  execution_date  quantity
        self.first_array_1  = np.array([[          1,    1,    10,              0,        5],
                                        [          2,    1,     8,              1,        5],
                                        [          3,    1,     6,              3,       10],
                                        [          5,    1,     7,              4,        5],
                                        [          5,    1,     9,              6,       10],
                                        [          6,    1,     5,              5,        5]]).astype(float)
        self.second_array_1 = np.array([[          0,    1,    14,              1,        5],                 # smaller report_date
                                        [          1,    1,     6,              4,       10],                 # same report_date, smaller price
                                        [          1,    1,    11,              1,        5],                 # same report_date, bigger price
                                        [          2,    1,     3,              3,        5],                 # bigger report_date
                                        [          3,    1,    11,              5,       10],                 # same report_date, bigger price
                                        [          5,    1,     8,              4,       10],                 # same report_date, between two prices
                                        [          6,    1,     3,              4,        5],                 # same report_date, smaller price
                                        [          8,    1,     9,              4,       10]]).astype(float)  # bigger report_date
        self.expected_1     = np.array([[          0,    1,    14,              1,        5],
                                        [          1,    1,     6,              4,       10],
                                        [          1,    1,    10,              0,        5],
                                        [          1,    1,    11,              1,        5],
                                        [          2,    1,     3,              3,        5],
                                        [          2,    1,     8,              1,        5],
                                        [          3,    1,     6,              3,       10],
                                        [          3,    1,    11,              5,       10],
                                        [          5,    1,     7,              4,        5],
                                        [          5,    1,     8,              4,       10],
                                        [          5,    1,     9,              6,       10],
                                        [          6,    1,     3,              4,        5],
                                        [          6,    1,     5,              5,        5],
                                        [          8,    1,     9,              4,       10]]).astype(float)
        self.first_array_2  = np.array([[          1,    2,     9,              2,       10],
                                        [          3,    2,    11,              4,       10],
                                        [          4,    2,     9,              7,        5],
                                        [          7,    2,    13,              8,       10]]).astype(float)
        self.second_array_2 = np.array([[          0,    2,     4,              0,        5],                 # smaller report_date
                                        [          1,    2,     8,              3,       10],                 # same report_date, smaller price
                                        [          1,    2,    11,              5,       10],                 # same report_date, bigger price
                                        [          4,    2,    13,              7,        5],                 # same report_date, bigger price
                                        [          6,    2,    11,              7,        5],                 # between two report_dates
                                        [          7,    2,    12,              4,       10],                 # same report_date, smaller price
                                        [          7,    2,    15,              7,        5]]).astype(float)  # same report_date, bigger price
        self.expected_2     = np.array([[          0,    2,     4,              0,        5],
                                        [          1,    2,     8,              3,       10],
                                        [          1,    2,     9,              2,       10],
                                        [          1,    2,    11,              5,       10],
                                        [          3,    2,    11,              4,       10],
                                        [          4,    2,     9,              7,        5],
                                        [          4,    2,    13,              7,        5],
                                        [          6,    2,    11,              7,        5],
                                        [          7,    2,    12,              4,       10],
                                        [          7,    2,    13,              8,       10],
                                        [          7,    2,    15,              7,        5]]).astype(float)

    def test_append_entries_to_end_of_group(self):
        entries_1 = np.random.randint(0, 10, 10)
        entries_2 = np.random.randint(0, 10, 10)
        self.assertTrue(np.array_equal(self.grouped.append_entries_to_end_of_group(entries_1, entries_2), \
                                       np.concatenate([entries_1, entries_2])))

    def test_compare_values(self):
        self.assertEqual(self.grouped.compare_values(1, 0),  1)  # 1 > 0
        self.assertEqual(self.grouped.compare_values(1, 1),  0)  # 1 = 1
        self.assertEqual(self.grouped.compare_values(1, 2), -1)  # 1 < 2

    def test_compare_trades(self):
        self.assertEqual(self.grouped.compare_trades(np.array([1., 1.]), np.array([1., 1.]), 0),  0)  # equal trades
        self.assertEqual(self.grouped.compare_trades(np.array([1., 1.]), np.array([1., 2.]), 0), -1)  # same report_dates
        self.assertEqual(self.grouped.compare_trades(np.array([1., 1.]), np.array([1., 0.]), 0),  1)  # same report_dates
        self.assertEqual(self.grouped.compare_trades(np.array([2., 1.]), np.array([1., 2.]), 0),  1)  # different report_dates
        self.assertEqual(self.grouped.compare_trades(np.array([1., 2.]), np.array([2., 1.]), 0), -1)  # different report_dates

    def test_merge_entries_into_group(self):
        self.assertTrue(np.array_equal(self.grouped.merge_entries_into_group(self.first_array_1, self.second_array_1), self.expected_1))
        self.assertTrue(np.array_equal(self.grouped.merge_entries_into_group(self.first_array_2, self.second_array_2), self.expected_2))

    def test_append_entries_to_group(self):
        self.assertTrue(np.array_equal(self.grouped.append_entries_to_group(self.first_array_1, self.second_array_1), self.expected_1))
        self.assertTrue(np.array_equal(self.grouped.append_entries_to_group(self.first_array_2, self.second_array_2), self.expected_2))

    def test_remove_entries(self):
        grouped = GroupedEntryTest()
        grouped.append_entries(np.zeros(10).astype(int),
                               np.array([5, 3, 0, 2, 1, 4, 9, 6, 8, 7]),
                               TradeFeatureArray(0, []), 0)
        # remove entries [7, 9) -> 7, 8
        grouped.remove_entries(7, 9)
        self.assertTrue(np.array_equal(grouped.get_entry_groups(), np.array([5, 3, 0, 2, 1, 4, 9, 6])))
        # remove entries 9
        grouped.remove_entries(9, 10)
        self.assertTrue(np.array_equal(grouped.get_entry_groups(), np.array([5, 3, 0, 2, 1, 4, 6])))
        # remove entries 2, 3, 4, 5, 6
        grouped.remove_entries(2, 7)
        self.assertTrue(np.array_equal(grouped.get_entry_groups(), np.array([0, 1])))
        # remove all entries
        grouped.remove_entries(0, 2)
        self.assertTrue(np.array_equal(grouped.get_entry_groups(), np.array([]).astype(int)))

    def test_get_the_end_of_group(self):
        entries = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 5])      # 'group' should be sorted
        # groups -- [0, 0], [1], [2, 2, 2], [3], [4, 4], [5]
        # there are six group ids    -- 0, 1, 2, 3, 4, 5
        # last entries of each group -- 1, 2, 5, 6, 8, 9
        self.assertEqual(self.grouped.get_the_end_of_group(start=0, entries=entries), 1)
        self.assertEqual(self.grouped.get_the_end_of_group(start=1, entries=entries), 1)
        self.assertEqual(self.grouped.get_the_end_of_group(start=2, entries=entries), 2)
        self.assertEqual(self.grouped.get_the_end_of_group(start=3, entries=entries), 5)
        self.assertEqual(self.grouped.get_the_end_of_group(start=4, entries=entries), 5)
        self.assertEqual(self.grouped.get_the_end_of_group(start=5, entries=entries), 5)
        self.assertEqual(self.grouped.get_the_end_of_group(start=6, entries=entries), 6)
        self.assertEqual(self.grouped.get_the_end_of_group(start=7, entries=entries), 8)
        self.assertEqual(self.grouped.get_the_end_of_group(start=8, entries=entries), 8)
        self.assertEqual(self.grouped.get_the_end_of_group(start=9, entries=entries), 9)

    def test_update_group_value_of_entry(self):
        empty_array = np.array([]).astype(int)
        # update with an empty array
        self.assertTrue(np.array_equal(self.grouped.update_group_value_of_entry(empty_array, empty_array), empty_array))
        # update
        self.assertTrue(np.array_equal(self.grouped.update_group_value_of_entry(np.array([0, 1]), np.array([0, 1])), np.array([0, 1])))
        # update with an empty array again
        self.assertTrue(np.array_equal(self.grouped.update_group_value_of_entry(empty_array, empty_array), np.array([0, 1])))
        # update once again
        self.assertTrue(np.array_equal(self.grouped.update_group_value_of_entry(np.array([2, 3]), np.array([2, 3])), np.array([0, 1, 2, 3])))
        # update existing group_values
        self.assertTrue(np.array_equal(self.grouped.update_group_value_of_entry(np.array([1, 2]), np.array([4, 5])), np.array([0, 4, 5, 3])))

    def test_get_index_of_most_recent_entry_in_group(self):
        report_dates = np.array([[1], [1], [2], [2], [2], [3], [5], [6], [6], [7]]).astype(float)
        # before the first report_date
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 0), -1)
        # same as the first report_date
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 1), -1)
        # later the first report_date
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 2),  1)
        # between two report_dates
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 4),  5)
        # before the last report_date
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 6),  6)
        # same as the last report_date
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 7),  8)
        # later the last report_date
        self.assertEqual(self.grouped.get_index_of_most_recent_entry_in_group(report_dates, 8),  9)


class GroupedHistoryTestCase(unittest.TestCase):
    def test_init(self):
        with patch('grouped_history.GroupedHistory._GroupedHistory__init_grouped_entry') as mock_init_grouped_entry:
            grouped_history = GroupedHistory('groups', 'sequence_length', 'rfq_features', 'trade_features', 'rfq_labels')
            self.assertTrue(grouped_history._GroupedHistory__trades is None)
            self.assertTrue(grouped_history._GroupedHistory__trade_feature_array is None)
            self.assertEqual(grouped_history._GroupedHistory__sequence_length, 'sequence_length')
            self.assertEqual(grouped_history._GroupedHistory__rfq_features, 'rfq_features')
            self.assertEqual(grouped_history._GroupedHistory__trade_features, 'trade_features')
            self.assertEqual(grouped_history._GroupedHistory__rfq_labels, 'rfq_labels')

    def test_init_grouped_entry(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_init:
            mock_init.return_value = None
            grouped_history = GroupedHistory(None, None, None, None, None)
            groups = ['group_1', 'group_2', 'group_3']
            grouped_history._GroupedHistory__init_grouped_entry(groups)
            self.assertEqual(grouped_history._GroupedHistory__groups, groups)
            self.assertEqual(list(grouped_history._GroupedHistory__grouped_entry.keys()), groups)
            for group in groups:
                self.assertTrue(isinstance(grouped_history._GroupedHistory__grouped_entry[group], GroupedEntry))

    def test_init_column_indices(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_init:
            mock_init.return_value = None
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__groups = ['figi', 'issuer', 'industry', 'sector', 'rating']
            grouped_history._GroupedHistory__rfq_features = ['ats_indicator', 'buy_sell', 'coupon', 'execution_date', 'issue_date', 'maturity', 'outstanding', 'quantity', 'rating', 'report_date', 'side']
            grouped_history._GroupedHistory__trade_features = ['ats_indicator', 'buy_sell', 'coupon', 'execution_date', 'issue_date', 'maturity', 'outstanding', 'price', 'quantity', 'rating', 'report_date', 'side']
            grouped_history._GroupedHistory__rfq_labels = ['price']
            columns = ['ats_indicator', 'buy_sell', 'coupon', 'execution_date', 'figi', 'industry', 'issue_date', 'issuer', 'maturity', 'outstanding', 'price', 'quantity', 'rating', 'report_date', 'sector', 'side', 'yield']
            grouped_history._GroupedHistory__init_column_indices(columns)
            self.assertEqual(grouped_history._GroupedHistory__group_indices, {'figi': 4, 'issuer': 7, 'industry': 5, 'sector': 14, 'rating': 12})
            self.assertEqual(grouped_history._GroupedHistory__rfq_feature_indices, [0, 1, 2, 3, 6, 8, 9, 11, 12, 13, 15])
            self.assertEqual(grouped_history._GroupedHistory__trade_feature_indices, [0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 15])
            self.assertEqual(grouped_history._GroupedHistory__rfq_label_indices, [10])
            self.assertTrue(isinstance(grouped_history._GroupedHistory__trade_feature_array, TradeFeatureArray))

    def test_get_entry_of_next_trade_by_execution_date(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_init:
            mock_init.return_value = None
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__trades = pa.table({'execution_date': [1, 1, 2, 2, 2, 3, 5, 6, 6, 7]})
            # before the first execution_date
            self.assertEqual(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(0), 0)
            # same as the first execution_date
            self.assertEqual(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(1), 0)
            # later the first execution_date
            self.assertEqual(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(2), 2)
            # between two different execution_dates
            self.assertEqual(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(4), 6)
            # before the last execution_date
            self.assertEqual(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(6), 7)
            # same as the last execution_date
            self.assertEqual(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(7), 9)
            # later the last execution_date
            self.assertTrue(grouped_history._GroupedHistory__get_entry_of_next_trade_by_execution_date(8) is None)

    def test_append_new_trades_to_groups(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_grouped_history_init, \
             patch('grouped_history.GroupedHistory._GroupedHistory__init_column_indices') as mock_init_column_indices, \
             patch('grouped_history.GroupedHistory._GroupedHistory__sort_by_report_date_and_group_value') as mock_sort_by_report_date_and_group_value:
            mock_grouped_history_init.return_value = None
            group = 'group'
            mock_sort_by_report_date_and_group_value.return_value = pa.table({group: [1.], 'entry': [2], 'report_date': [3.]})
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__trades = None
            grouped_history._GroupedHistory__groups = [group]
            grouped_history._GroupedHistory__grouped_entry = {group: Mock()}
            grouped_history._GroupedHistory__grouped_entry[group].append_entries = Mock()
            grouped_history._GroupedHistory__trade_feature_indices = [0]
            grouped_history._GroupedHistory__trade_feature_array = TradeFeatureArray(2, [0, 1])
            # append new trades to the empty grouped_history
            new_trades_1 = pa.table({group: [1., 1., 1., 2., 2.], 'report_date': [0., 1., 2., 3., 4.]})
            grouped_history._GroupedHistory__append_new_trades_to_groups(new_trades_1)
            mock_init_column_indices.assert_called_once_with([group, 'report_date'])
            self.assertTrue(grouped_history._GroupedHistory__trades.equals(new_trades_1))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__trade_feature_array.get_feature_array(), np.array([[1, 0], [1, 1], [1, 2], [2, 3], [2, 4]]).astype(float)))
            new_trades_1_with_entry = pa.table({group: [1., 1., 1., 2., 2.], 'report_date': [0., 1., 2., 3., 4.], 'entry': [0, 1, 2, 3, 4]})
            mock_sort_by_report_date_and_group_value.assert_called_once_with(new_trades_1_with_entry, group)
            grouped_history._GroupedHistory__grouped_entry[group].append_entries.assert_called_once()
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][0], np.array([1])))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][1], np.array([2])))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][2], grouped_history._GroupedHistory__trade_feature_array))
            self.assertEqual(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][3], 2)
            # append once again
            new_trades_2 = pa.table({group: [3., 3., 4., 4., 5.], 'report_date': [5., 6., 7., 8., 9.]})
            grouped_history._GroupedHistory__append_new_trades_to_groups(new_trades_2)
            self.assertTrue(grouped_history._GroupedHistory__trades.equals(pa.concat_tables([new_trades_1, new_trades_2])))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__trade_feature_array.get_feature_array(), np.array([[1, 0], [1, 1], [1, 2], [2, 3], [2, 4], [3, 5], [3, 6], [4, 7], [4, 8], [5, 9]]).astype(float)))
            new_trades_2_with_entry = pa.table({group: [3., 3., 4., 4., 5.], 'report_date': [5., 6., 7., 8., 9.], 'entry': [5, 6, 7, 8, 9]})
            mock_sort_by_report_date_and_group_value.assert_called_with(new_trades_2_with_entry, group)
            grouped_history._GroupedHistory__grouped_entry[group].append_entries.assert_called()
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][0], np.array([1])))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][1], np.array([2])))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][2], grouped_history._GroupedHistory__trade_feature_array))
            self.assertEqual(grouped_history._GroupedHistory__grouped_entry[group].append_entries.call_args[0][3], 2)

    def test_remove_entries_from_groups(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_grouped_history_init:
            mock_grouped_history_init.return_value = None
            group = 'group'
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__groups = [group]
            grouped_history._GroupedHistory__grouped_entry = {group: Mock()}
            grouped_history._GroupedHistory__grouped_entry[group].remove_entries = Mock()
            grouped_history._GroupedHistory__remove_entries_from_groups('first_entry', 'last_entry')
            grouped_history._GroupedHistory__grouped_entry[group].remove_entries.assert_called_once_with('first_entry', 'last_entry')

    def test_append_trades(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_init, \
             patch('grouped_history.GroupedHistory._GroupedHistory__get_entry_of_next_trade_by_execution_date') as mock_get_entry_of_next_trade_by_execution_date, \
             patch('grouped_history.GroupedHistory._GroupedHistory__remove_entries_from_groups') as mock_remove_entries_from_groups, \
             patch('grouped_history.GroupedHistory._GroupedHistory__append_new_trades_to_groups') as mock_append_new_trades_to_groups, \
             patch('grouped_history.combine_trace') as mock_combine_trace:
            mock_init.return_value = None
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__trades = pa.table({'execution_date': [1., 2., 3., 4., 5.],
                                                                'report_date'   : [1., 2., 3., 4., 5.]})
            grouped_history._GroupedHistory__trade_feature_array = TradeFeatureArray(2, [0, 1])
            new_trades = pa.table({'execution_date': [7., 3., 4., 6., 5.],
                                   'report_date'   : [4., 3., 5., 7., 6.]})
            mock_get_entry_of_next_trade_by_execution_date.return_value = 2
            sorted_trades = pa.table({'execution_date': [1., 2., 3.]})
            mock_combine_trace.return_value = sorted_trades
            grouped_history.append_trades(new_trades)
            mock_get_entry_of_next_trade_by_execution_date.assert_called_once_with(3)  # first execution_date in new_trades
            mock_remove_entries_from_groups.assert_called_once_with(2, 5)              # first_entry and last_entry for reordering
            mock_combine_trace.assert_called_once()
            self.assertTrue(mock_combine_trace.call_args[0][0].equals(pa.table({'execution_date': [7., 3., 4., 6., 5., 3., 4., 5.],
                                                                                'report_date'   : [4., 3., 5., 7., 6., 3., 4., 5.]})))
            mock_append_new_trades_to_groups.assert_called_once()
            self.assertTrue(mock_append_new_trades_to_groups.call_args[0][0].equals(sorted_trades))

    def test_generate_batch(self):
        with patch('grouped_history.GroupedHistory.__init__') as mock_init:
            mock_init.return_value = None
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__rfq_label_indices = [0]
            grouped_history._GroupedHistory__sequence_length = 2
            grouped_history._GroupedHistory__rfq_features = ['rfq_feature_1', 'rfq_feature_2']
            grouped_history._GroupedHistory__rfq_feature_indices = [1, 2]
            grouped_history._GroupedHistory__trade_features = ['execution_date', 'report_date']
            grouped_history._GroupedHistory__trade_feature_indices = [3, 4]
            grouped_history._GroupedHistory__trades = pa.table({'execution_date': [1, 2, 3], 'report_date': [2, 3, 4]})
            grouped_history._GroupedHistory__groups = ['group']
            grouped_history._GroupedHistory__group_indices = {'group': 5}
            grouped_history._GroupedHistory__grouped_entry = {'group': Mock()}
            grouped_history._GroupedHistory__grouped_entry['group'].set_rows = Mock()
            grouped_history._GroupedHistory__report_date_index = 1
            Z_b = np.random.randint(0, 10, size=(10, 10)).astype(float)
            grouped_history._GroupedHistory__trade_feature_array = TradeFeatureArray(0, [])
            X_b, Y_b = grouped_history.generate_batch(Z_b)
            self.assertTrue(np.array_equal(X_b[:, [0, 1]], Z_b[:, [1, 2]]))
            self.assertTrue(np.array_equal(Y_b, Z_b[:, [0]]))
            grouped_history._GroupedHistory__grouped_entry['group'].set_rows.assert_called_once()
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry['group'].set_rows.call_args[0][0], np.zeros((10, 4))))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry['group'].set_rows.call_args[0][1], Z_b[:, 5]))
            self.assertTrue(np.array_equal(grouped_history._GroupedHistory__grouped_entry['group'].set_rows.call_args[0][2], Z_b[:, 0]))
            self.assertEqual(grouped_history._GroupedHistory__grouped_entry['group'].set_rows.call_args[0][3], 2)
            self.assertEqual(grouped_history._GroupedHistory__grouped_entry['group'].set_rows.call_args[0][5], 1)

    def test_get_last_figi_trades(self):
        feature_array = np.array([[1, 2], [3, 4]])
        table = pa.table({'figi': pa.array([1, 3]), 'value': pa.array([2, 4])})
        with patch('grouped_history.GroupedHistory.__init__') as mock_grouped_history_init:
            mock_grouped_history_init.return_value = None
            grouped_history = GroupedHistory(None, None, None, None, None)
            grouped_history._GroupedHistory__groups = ['figi']
            grouped_history._GroupedHistory__grouped_entry = {}
            grouped_history._GroupedHistory__grouped_entry['figi'] = Mock()
            grouped_history._GroupedHistory__grouped_entry['figi'].get_last_entries = Mock(return_value='entries')
            grouped_history._GroupedHistory__trade_feature_array = Mock()
            grouped_history._GroupedHistory__trade_feature_array.get_trade_features = Mock(return_value=feature_array)
            grouped_history._GroupedHistory__trades = table
            self.assertTrue(grouped_history.get_last_figi_trades('figi_value', 'trade_count').equals(table))
            grouped_history._GroupedHistory__grouped_entry['figi'].get_last_entries.assert_called_once_with('figi_value', 'trade_count')
            grouped_history._GroupedHistory__trade_feature_array.get_trade_features.assert_called_once_with('entries')


class TradeFeatureArrayTestCase(unittest.TestCase):
    def test_append_and_resize(self):
        trades_1 = np.random.rand(10, 10)
        trades_2 = np.random.rand(10, 10)
        trade_feature_array = TradeFeatureArray(10, [])
        trade_feature_array.append(trades_1)
        trade_feature_array.append(trades_2)
        self.assertTrue(np.array_equal(trade_feature_array.get_feature_array(), np.concatenate((trades_1, trades_2), axis=0)))
        trade_feature_array.resize(15)
        self.assertTrue(np.array_equal(trade_feature_array.get_feature_array(), np.concatenate((trades_1, trades_2), axis=0)[:15, :]))


if __name__ == '__main__':
    unittest.main()