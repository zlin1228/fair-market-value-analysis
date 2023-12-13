import numpy as np
import pandas as pd

from feed_forward_nn import FeedForwardNN
from generator import TraceDataGenerator
from trade_history import TradeHistory
from helpers import get_isin_from_cusip
from trace_data import get_trace_data
import helpers
import datetime


# FROM https://gist.github.com/martinthenext/ed9e7b475494061bc5eca2dce7385122


tp_icap = pd.read_csv('/data/tp_icap/BEP_26052022_18.csv')
cusip_to_figi = pd.read_csv('/data/figi/cusip_to_figi.uniq.csv')
cusip_to_figi['ISIN'] = cusip_to_figi.cusip.apply(get_isin_from_cusip)
isin_to_figi = cusip_to_figi[['ISIN', 'figi']]
tp_icap = tp_icap[tp_icap.Date == '26 MAY 2022']
tp_icap = isin_to_figi.merge(tp_icap, on='ISIN')[['figi', 'ISIN', 'Bid Price', 'Mid Price', 'Ask Price']]
tp_icap = tp_icap.rename(columns={'ISIN': 'isin',
                                  'Bid Price': 'tp_icap_bid_price',
                                  'Mid Price': 'tp_icap_mid_price',
                                  'Ask Price': 'tp_icap_ask_price'})
tp_icap.to_csv('../data/tp_icap_summary.csv')
print(tp_icap.columns)
print(tp_icap)

print('Locading TRACE')
trace, ordinals = get_trace_data()
trace = trace.sort_values(by='report_date', ascending=True)
data_cube = TradeHistory(trace, ordinals)
generator = TraceDataGenerator(data_cube, should_shuffle=False)
model = FeedForwardNN(generator, generator, generator)
model.create()

target_start_date = helpers.from_single_date_time_to_timestamp(datetime.datetime.strptime('20220527', "%Y%m%d"))
target_end_date = helpers.from_single_date_time_to_timestamp(datetime.datetime.strptime('20220528', "%Y%m%d"))

Z_date = generator.get_date_Z_b(target_start_date, target_end_date)
Z_date_df = generator.get_Z_df_from_Z_b(Z_date)
X_date, Y_date = generator.generate_batch(Z_date)
Y_date_deepmm = model.evaluate_batch(X_date)

deepmm_overall_error = (Y_date - Y_date_deepmm)

print(f'On May 27th, the overall MSE of Deep MM\'s model: ' +
      f'{np.sum(deepmm_overall_error * deepmm_overall_error)/Y_date.shape[0]}')

print(f'Z_date_df shape: {Z_date_df.shape}')
print(f'Y_date_deepmm shape: {Y_date_deepmm.shape}')
print(f'Y_date shape: {Y_date.shape}')

Z_date_df['deepmm_price'] = Y_date_deepmm

reverse_ordinals = helpers.reverse_ordinals(ordinals)


def deordinal(c):
    Z_date_df[c] = Z_date_df[c].apply(lambda x: reverse_ordinals[c][x])


deordinal('figi')
deordinal('buy_sell')
deordinal('side')
joined = Z_date_df.merge(tp_icap, on='figi')

conditions = [
    joined.buy_sell == 'B',  # Buy
    (joined.buy_sell == 'S') &
        (joined.side != 'D'),  # Sell
    (joined.buy_sell == 'S') &
        (joined.side == 'D')  # Dealer
]

choices = [
    joined.tp_icap_bid_price,
    joined.tp_icap_ask_price,
    joined.tp_icap_mid_price
]

joined['tp_icap_price'] = np.select(conditions, choices)

print('tp_icap_price value counts:')
print(joined.tp_icap_price.value_counts())

print(joined[['tp_icap_price', 'buy_sell', 'side', 'tp_icap_bid_price', 'tp_icap_ask_price', 'tp_icap_mid_price']])
print(joined[['price', 'deepmm_price', 'tp_icap_price', 'buy_sell', 'side']])


def compute_error(price_column_name):
    errors = (joined[price_column_name] - joined.price).abs()
    mae = errors.sum() / errors.shape[0]
    mse = (errors * errors).sum() / errors.shape[0]
    print(f'{price_column_name} MSE: {mse}, MAE: {mae}')
    return mae, mse


deepmm_mae, deepmm_mse = compute_error('deepmm_price')
tp_icap_mae, tp_icap_mse = compute_error('tp_icap_price')
joined.to_csv('../data/tp_icap_deepmm_joined.csv')
