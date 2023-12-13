import settings
settings.override_if_main(__name__, 2)

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from generator import TraceDataGenerator
import helpers
from load_data import get_ordinals
from performance_helper import create_profiler

def calculate_metrics(na_filter, error, quantity, spread):
    # Apply na_filter
    error = error[na_filter]
    quantity = quantity[na_filter]

    absolute_error = np.abs(error)
    squared_error = error * error
    vw_absolute_error = absolute_error * quantity
    vw_squared_error = squared_error * quantity

    # Calculate metrics
    count = error.shape[0]
    mse = np.sum(squared_error) / count
    rmse = np.sqrt(mse)
    mae = np.sum(absolute_error) / count
    vw_mse = np.sum(vw_squared_error) / np.sum(quantity)
    vw_rmse = np.sqrt(vw_mse)
    vw_mae = np.sum(vw_absolute_error) / np.sum(quantity)

    tiles = list(range(1, settings.get('$.tiles') + 1))

    def calculate_percentile(error):
        return [np.percentile(error, percentile) for percentile in tiles]

    metrics =  {
        'count'    : count,
        'mse'      : mse,
        'rmse'     : rmse,
        'mae'      : mae,
        'vw_mse'   : vw_mse,
        'vw_rmse'  : vw_rmse,
        'vw_mae'   : vw_mae,
        'error_tiles': {
            'tiles' : tiles,
            'values'   : calculate_percentile(error)
        }
    }
    if spread is not None:
        spread = spread[na_filter]
        vw_spread = spread * quantity
        avg_spread = np.sum(spread) / count
        vw_avg_spread = np.sum(vw_spread) / np.sum(quantity)

        spread_adjusted_error = error / spread

        metrics['spread_adjusted_error_tiles'] = {
            'tiles' : tiles,
            'values' : calculate_percentile(spread_adjusted_error)
        }
        metrics['avg_spread'] = avg_spread
        metrics['vw_avg_spread'] = vw_avg_spread

    return metrics


class TraceModel(ABC):
    def __init__(self,
                 model_name: str,
                 model_settings: dict,
                 train_generator: TraceDataGenerator,
                 validation_generator: TraceDataGenerator,
                 test_generator: TraceDataGenerator):
        self.model_name = model_name
        self.model_settings = model_settings
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator

        # Set the 'buy_sell' and 'side' indices.
        self.index_of_buy_sell = self.test_generator.get_rfq_feature_Z_index('buy_sell')
        self.index_of_side = self.test_generator.get_rfq_feature_Z_index('side')

        # Set feature values for buy, sell, and dealer.
        self.buy_features    = {'buy_sell': float(get_ordinals()['buy_sell'].index('B').as_py()),
                                'side'    : float(get_ordinals()['side'].index('C').as_py())}
        self.sell_features   = {'buy_sell': float(get_ordinals()['buy_sell'].index('S').as_py()),
                                'side'    : float(get_ordinals()['side'].index('C').as_py())}
        self.dealer_features = {'buy_sell': float(get_ordinals()['buy_sell'].index('S').as_py()),
                                'side'    : float(get_ordinals()['side'].index('D').as_py())}

    def cleanup(self):
        del self.model_name
        del self.model_settings
        # three generators share the same data_cube
        self.train_generator.cleanup(True)           # cleanup 'data_cube'
        self.validation_generator.cleanup(False)     # not cleanup
        self.test_generator.cleanup(False)           # not cleanup
        del self.train_generator
        del self.validation_generator
        del self.test_generator
        del self.index_of_buy_sell
        del self.index_of_side
        del self.buy_features
        del self.sell_features
        del self.dealer_features

    def get_model_setting(self, setting_name: str) -> any:
        return self.model_settings[setting_name]

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate_batch(self, X_b):
        pass

    def evaluate_batch_index(self, index):
        # Generate a batch by index.
        Z_b = self.test_generator.get_Z_b(index).copy()
        X_b, Y_b = self.test_generator.generate_batch_np(Z_b)

        # Evaluate the batch.
        Y_b_hat = self.evaluate_batch(X_b)

        # We only want to count the error for
        # cases where the Y_hat was valid (where
        # there was at least one related trade which
        # happened before each target RFQ typically)
        na_filter = ~np.isnan(Y_b_hat).any(axis=1)

        # Get error values.
        error = Y_b_hat - Y_b

        # Get quantity values.
        quantity = self.test_generator.get_rfq_feature(X_b, 'quantity')

        spread = None
        if settings.get('$.calculate_spread'):
            # Get buy prices.
            Z_b[:, self.index_of_buy_sell] = self.buy_features['buy_sell']
            Z_b[:, self.index_of_side] = self.buy_features['side']
            X_b, _ = self.test_generator.generate_batch_np(Z_b)
            Y_b_hat_buy = self.evaluate_batch(X_b)

            # Get sell prices.
            Z_b[:, self.index_of_buy_sell] = self.sell_features['buy_sell']
            X_b, _ = self.test_generator.generate_batch_np(Z_b)
            Y_b_hat_sell = self.evaluate_batch(X_b)

            # Get spread values.
            spread = np.abs(Y_b_hat_buy - Y_b_hat_sell)

            # Check if 'Y_b_hat_buy' and 'Y_b_hat_sell' are invalid, even 'Y_b_hat' is valid.
            assert(np.array_equal(na_filter, ~np.isnan(spread).any(axis=1)))

        return na_filter, error, quantity, spread

    def evaluate_batches(self):
        for i in range(self.test_generator.__len__()):
            na_filter, error, quantity, spread = self.evaluate_batch_index(i)
            if i == 0:
                na_filters, errors, quantities, spreads = na_filter, error, quantity, spread
            else:
                na_filters = np.append(na_filters, na_filter)
                errors = np.append(errors, error)
                quantities = np.append(quantities, quantity)
                if settings.get('$.calculate_spread'):
                    spreads = np.append(spreads, spread)

        # This function returns overall na_filter, error, quantity, and spread.
        return na_filters, errors, quantities, spreads

    def get_buy_filter(self, na_filter: np.ndarray):
        buy_filter = (self.test_generator.get_Z()[:, self.index_of_buy_sell] == self.buy_features['buy_sell']) & \
            (self.test_generator.get_Z()[:, self.index_of_side] == self.buy_features['side'])
        return buy_filter[:na_filter.shape[0]] & na_filter

    def get_sell_filter(self, na_filter: np.ndarray):
        sell_filter = (self.test_generator.get_Z()[:, self.index_of_buy_sell] == self.sell_features['buy_sell']) & \
            (self.test_generator.get_Z()[:, self.index_of_side] == self.sell_features['side'])
        return sell_filter[:na_filter.shape[0]] & na_filter

    def get_dealer_filter(self, na_filter: np.ndarray):
        dealer_filter = (self.test_generator.get_Z()[:, self.index_of_buy_sell] == self.dealer_features['buy_sell']) & \
            (self.test_generator.get_Z()[:, self.index_of_side] == self.dealer_features['side'])
        return dealer_filter[:na_filter.shape[0]] & na_filter

    def get_metrics(self, na_filter, error, quantity, spread):
        metrics = {}
        metrics['overall'] = calculate_metrics(na_filter, error, quantity, spread)
        metrics['buy'] = calculate_metrics(self.get_buy_filter(na_filter), error, quantity, spread)
        metrics['sell'] = calculate_metrics(self.get_sell_filter(na_filter), error, quantity, spread)
        metrics['dealer'] = calculate_metrics(self.get_dealer_filter(na_filter), error, quantity, spread)
        return metrics

    def evaluate(self):
        na_filter, error, quantity, spread = self.evaluate_batches()
        return self.get_metrics(na_filter, error, quantity, spread)

    def infer_test(self):
        Z = None
        Z_Ig = None
        Z_Hy = None
        ordinals = get_ordinals()
        reverse_ordinals = helpers.reverse_ordinals(ordinals)
        investment_grade_ratings = settings.get('$.investment_grade_ratings')

        for i in range(self.test_generator.__len__()):
            count, Z_b, Y_b, Y_b_hat, quantities = self.evaluate_batch_index(i)
            Z_df = self.test_generator.get_Z_df_from_Z_b(Z_b)
            Z_df = Z_df[['execution_date', 'rating', 'report_date', 'figi', 'quantity', 'buy_sell',
                         'side', 'ats_indicator', 'price']]
            Z_df['deepmm_price'] = Y_b_hat
            Z_df.quantity = Z_df.quantity.astype(int)
            Z_df.report_date = pd.to_datetime(Z_df.report_date)
            Z_df.execution_date = pd.to_datetime(Z_df.execution_date)
            for column in ['figi', 'buy_sell', 'side', 'ats_indicator', 'rating']:
                Z_df[column] = Z_df[column].map(reverse_ordinals[column])

            Z_df_Ig = Z_df[Z_df.rating.isin(investment_grade_ratings)]
            Z_df_Hy = Z_df[~Z_df.rating.isin(investment_grade_ratings)]
            if Z is None:
                Z = Z_df
                Z_Ig = Z_df_Ig
                Z_Hy = Z_df_Hy
            else:
                Z = pd.concat([Z, Z_df])
                Z_Ig = pd.concat([Z_Ig, Z_df_Ig])
                Z_Hy = pd.concat([Z_Hy, Z_df_Hy])
        return Z, Z_Ig, Z_Hy

    @abstractmethod
    def create(self):
        pass

    @classmethod
    def infer_historical(cls):
        model, test_data_generator = cls.load_data_and_create_model()
        return model.infer_test()

    # TODO: fix this to be fully ported to pyarrow.
    @staticmethod
    def get_date_ranges_from_trace(date_ranges, pa_trace: pa.Table):
        profile = create_profiler('model.get_date_ranges_from_trace')
        with profile():
            column_names = pa_trace.column_names
            if date_ranges is None:
                return None
            if not isinstance(date_ranges, list):
                raise ValueError('Data set date ranges must be a list of start and end date dictionaries')
            trace_subsets = []

            date_range_strptime_pattern = '%Y-%m-%d'

            for date_range in date_ranges:
                with profile(f'get date range for {date_range}'):
                    if not isinstance(date_range, dict):
                        raise ValueError('Data set date ranges must be a list of start and end date dictionaries')
                    timezone = settings.get('$.finra_timezone')
                    start_date = pc.assume_timezone(pc.strptime(date_range['start'],date_range_strptime_pattern,'ns'), timezone=timezone)
                    end_date = pc.assume_timezone(pc.strptime(date_range['end'],date_range_strptime_pattern,'ns'), timezone=timezone)
                    report_dates = pc.assume_timezone(pc.cast(pa_trace['report_date'], pa.timestamp('ns')), timezone=timezone)
                    trace_subset = pa_trace.filter(pc.and_(pc.greater(report_dates, start_date), pc.less_equal(report_dates, end_date)))
                    trace_subsets.append(trace_subset)
            filtered = pa.concat_tables(trace_subsets)
            assert column_names == filtered.column_names
            return filtered

    @staticmethod
    def percent_split(trace: pa.Table) -> tuple[pa.Table, pa.Table, pa.Table]:
        profile = create_profiler('model.percent_split')
        with profile('Getting training data split index'):
            training_data_split = TraceModel.get_split_index(trace, 'training_data_split')
        with profile('Getting validation data split index'):
            validation_data_split = TraceModel.get_split_index(trace, 'validation_data_split')
        with profile('Split training data'):
            training_data = trace.slice(0, training_data_split)
        with profile('Split validation data'):
            validation_data = trace.slice(training_data_split, validation_data_split - training_data_split)
        with profile('Split test data'):
            test_data = trace.slice(validation_data_split)
        return training_data, validation_data, test_data

    @staticmethod
    def date_split(trace):
        profile = create_profiler('model.date_split')
        with profile('get date ranges for training.'):
            train_data = TraceModel.get_date_ranges_from_trace(settings.get('$.training_date_ranges'), trace)
        with profile('get date ranges for validation.'):
            validation_data = TraceModel.get_date_ranges_from_trace(settings.get('$.validation_date_ranges'), trace)
        with profile('get date ranges for test data.'):
            test_data = TraceModel.get_date_ranges_from_trace(settings.get('$.test_date_ranges'), trace)
        return train_data, validation_data, test_data

    @staticmethod
    def get_split_index(trace, split_name):
        return int(settings.get(f'$.{split_name}') * trace.shape[0])