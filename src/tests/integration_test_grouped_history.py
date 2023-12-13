import sys
sys.path.insert(0, '../')

standard_settings = {"$.data_path": "s3://deepmm.test.data/deepmm.parquet/v0.7.zip"}
import settings
settings.override(standard_settings)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import time
import unittest

from tests.helpers import CleanEnvironment
import pyximport
pyximport.install(language_level=3)
from grouped_history import GroupedHistoryTest, combine_trace
from load_data import get_initial_trades
from performance_helper import create_profiler
from evaluate_experiment import getsize


class TestGroupedHistory(unittest.TestCase):
    def __sort_table(self, table: pa.Table, column_list: list[str]) -> pa.Table:
        return table.sort_by([(col, 'ascending') for col in column_list])

    def test_append_trades(self):
        with CleanEnvironment(lambda _: {**standard_settings}):
            trades = get_initial_trades()

            for col in trades.column_names:
                filled_col = pc.fill_null(trades.column(col), 0)
                trades = trades.set_column(trades.column_names.index(col), col, filled_col)
            perm = np.random.permutation(trades.num_rows)
            trades = trades.take(perm)

            grouped_history = GroupedHistoryTest(groups=['figi', 'issuer', 'industry', 'sector', 'rating'], \
                                                sequence_length=10, \
                                                rfq_features=['ats_indicator', 'buy_sell', 'coupon', 'execution_date', 'issue_date', 'maturity', 'outstanding', 'quantity', 'rating', 'report_date', 'side'], \
                                                trade_features=['ats_indicator', 'buy_sell', 'coupon', 'execution_date', 'issue_date', 'maturity', 'outstanding', 'price', 'quantity', 'rating', 'report_date', 'side'], \
                                                rfq_labels=['price'])
            initial_size = getsize(grouped_history)

            profile = create_profiler('APPENDING')
            appending_count = 20
            appending_size = len(trades) // appending_count
            for i in range(appending_count):
                with profile(f'APPENDING {i}TH TRADES'):
                    if i < appending_count - 1:
                        appending_trades = trades.slice(i * appending_size, appending_size)
                    else:
                        appending_trades = trades.slice(i * appending_size, len(trades) - i * appending_size)
                    grouped_history.append_trades(appending_trades)

            combined_trades = self.__sort_table(combine_trace(trades), trades.column_names)
            trades = self.__sort_table(grouped_history._GroupedHistory__trades, trades.column_names)
            self.assertTrue(combined_trades.equals(trades))

            for group in grouped_history._GroupedHistory__groups:
                with profile(f'comparing {group} group'):
                    entries = grouped_history._GroupedHistory__grouped_entry[group].get_entry_groups()
                    column_list = trades.column_names
                    column_list.remove(group)
                    column_list.remove('report_date')
                    column_list = [group, 'report_date'] + column_list
                    sorted_trades = self.__sort_table(combined_trades, column_list)
                    self.assertTrue(np.array_equal(grouped_history._GroupedHistory__trade_feature_array.get_feature_array()[entries, :], sorted_trades.to_pandas().to_numpy()))

            # check the latency of the 'get_last_figi_trades' function
            time_to_get_whole_last_figi_trades, time_to_get_partial_last_figi_trades = 0, 0
            for figi in trades.column('figi').value_counts():
                value, count = figi['values'].as_py(), figi['counts'].as_py()
                # get the whole last figi trades
                calculation_time, figi_trades = self.time_to_get_last_figi_trades(grouped_history, value, count)
                time_to_get_whole_last_figi_trades += calculation_time
                # get the partial last figi trades
                split_count = 10
                for _ in range(split_count):
                    size = count // split_count
                    calculation_time, last_figi_trades = self.time_to_get_last_figi_trades(grouped_history, value, size)
                    time_to_get_partial_last_figi_trades += calculation_time
                    self.assertTrue(last_figi_trades.equals(figi_trades[-size:]))
            # compare time
            self.assertTrue(time_to_get_whole_last_figi_trades > time_to_get_partial_last_figi_trades)

            # check the process memory cleanup
            grouped_history.cleanup()
            self.assertTrue(getsize(grouped_history) <= initial_size)

    def time_to_get_last_figi_trades(self, grouped_history, value, count):
        current_time = time.time()
        figi_trades = grouped_history.get_last_figi_trades(value, count)
        return time.time() - current_time, figi_trades


if __name__ == '__main__':
    unittest.main()