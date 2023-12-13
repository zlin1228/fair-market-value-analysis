# Read in textual lines of a filename specified by an argument
# and output a CSV file with the same name but with a .csv extension
# The CSV file will have the following columns:
#   - CUSIP
#   - ISIN
#   - Date
#   - Price
import os
import re
import sys
from typing import List
import pandas_market_calendars as mcal

import numpy as np
import pandas as pd
import s3, settings
from model import TraceModel
from feed_forward_nn import FeedForwardNN
import helpers


def parse_lines(lines, pattern):
    price_lines = []
    for line in lines:
        match = pattern.match(line)
        if match:
            price_lines.append(match.groupdict())
    data_df = pd.DataFrame(price_lines)
    return data_df


# "AZ718240     Corp",2020/03/09,#N/A,ETF,2022/11/02,107.1124,,,
# Create a regular expression to match the above line format
price_line_re = re.compile(r'"(?P<cusip>[A-Z0-9]{8})\s+Corp",(?P<date>\d{4}/\d{2}/\d{2}),#N/A,ETF,\d{4}/\d{2}/\d{2},(?P<index_price>\d+\.\d+),,,')

# "QZ552080     Corp",,,,,,ETF,2022/11/02,US63938CAE84
# Create a regular expression to match the above line format
isin_line_re = re.compile(r'"(?P<cusip>[A-Z0-9]{8})\s+Corp",,,,,,ETF,\d{4}/\d{2}/\d{2},(?P<isin>[A-Z0-9]{12})')


def load_index_file(filename):
    print('Loading ', filename)
    # Open the file for reading
    with open(filename, 'r') as f:
        # Read in all of the lines in the file
        lines = f.readlines()
    # Process all the lines matching price_line_re
    data_df = parse_lines(lines, price_line_re)
    isin_map_df = parse_lines(lines, isin_line_re)
    # Merge the data and ISINs
    combined_df = pd.merge(data_df, isin_map_df, on='cusip')
    print(combined_df)
    return combined_df


merged_df = pd.concat([load_index_file(file_path) for file_path in s3.get_object("s3://deepmm.indices/latest.zip")],
                      ignore_index=True)

# filter merged_df to only include the dates we care about
merged_df = merged_df[merged_df.date.isin(["2020/03/20", "2020/03/23", "2020/03/24", "2020/03/25",
                                           "2020/03/26", "2022/09/30", "2022/10/31"])]


# Load cbonds emissions data
file_paths: List[str] = s3.get_object(settings.get('$.cbonds_path'))
emissions_re = re.compile(r'.*emissions(_[\d]*)?.csv$')
emissions_file_paths = [file_path for file_path in file_paths if emissions_re.fullmatch(file_path)]

# Load cbonds emissions data and concatenate
cbonds_df = pd.concat([pd.read_csv(file_path, sep=';') for file_path in emissions_file_paths], ignore_index=True)
print(cbonds_df.columns)
cbonds_df = cbonds_df[['FIGI / FIGI RegS', 'ISIN / ISIN RegS', 'ISIN 144A', "Isin code 3"]]
# Rename columns
cbonds_df.rename(columns={'FIGI / FIGI RegS': 'figi',
                          'ISIN / ISIN RegS': 'isin',
                          "ISIN 144A":"isin_144a",
                          "CUSIP / CUSIP RegS":"cusip_cbonds",
                          "CUSIP 144A":"cusip_144a_cbonds"}, inplace=True)
print(cbonds_df.head())

# Merge the cbonds data and ISINs with the merged_df
isin_merged_df = pd.merge(merged_df, cbonds_df, on='isin')
print(isin_merged_df)

# Merge with the cbonds data on isin_144a
isin_144a_merged_df = pd.merge(merged_df, cbonds_df, left_on='isin', right_on='isin_144a')
print(isin_144a_merged_df)


# Add a column for the next trading day after the date
# After this we will join with the parquet file.
# We will use the next trading day to get the relevant trades from the parquet file,
# And compare the accuracy of the index price with the model's prices.

timezone_str = settings.get('$.finra_timezone')

# Get the NASDAQ calendar
nyse = mcal.get_calendar('NYSE')
valid_days = nyse.valid_days(start_date='2020-03-01', end_date='2022-12-31')
valid_days = valid_days.tz_localize(None).tz_localize(timezone_str)
# Set the timezone of valid_days to timezone in settings
valid_days = pd.to_datetime(pd.Series(valid_days))

# Convert date column to ns timestamp
isin_merged_df['date'] = pd.to_datetime(isin_merged_df['date']).dt.tz_localize(timezone_str)
# Add a new column for the next trading day after the date
isin_merged_df['next_trading_day'] = isin_merged_df['date'].apply(lambda x: valid_days[valid_days > x].iloc[0]).dt.date

model, test_data_generator = FeedForwardNN.load_data_and_create_model()

# Get the unique dates
unique_dates = isin_merged_df['next_trading_day'].unique()

def to_timestamp(t):
    return float(pd.to_datetime(t).to_datetime64().astype(int))

unique_timestamps = [to_timestamp(date) for date in unique_dates]
# Add one day to the unique_timestamps in nanoseconds
one_day_in_ns = 24 * 60 * 60 * 1000 * 1000 * 1000
unique_timestamps_day_after = [timestamp + one_day_in_ns for timestamp in unique_timestamps]


# Use get_date_Z_b to get the Z_b for each date to the next day and concatenate
Z_b = np.concatenate([test_data_generator.get_date_Z_b(date, next_day)
                 for date, next_day in zip(unique_timestamps, unique_timestamps_day_after)])
Z_date_df = test_data_generator.get_Z_df_from_Z_b(Z_b)
X_date, Y_date = test_data_generator.generate_batch(Z_b)
Y_date_deepmm = model.evaluate_batch(X_date)

assert(Z_date_df.shape[0] == Y_date_deepmm.shape[0])

Z_date_df['deepmm_price'] = Y_date_deepmm
Z_date_df['next_trading_day'] = pd.to_datetime(Z_date_df['execution_date'], unit='ns').dt.date
ordinals = test_data_generator.get_ordinals()
# Reverse the ordinals
reverse_ordinals = helpers.reverse_ordinals(ordinals)
def deordinal(c):
    Z_date_df[c] = Z_date_df[c].apply(lambda x: reverse_ordinals[c][x])

deordinal('figi')

joined = pd.merge(Z_date_df, isin_merged_df, on=['figi', 'next_trading_day'])

def compute_error(price_column_name):
    errors = (joined[price_column_name] - joined.price).abs()
    mae = errors.sum() / errors.shape[0]
    mse = (errors * errors).sum() / errors.shape[0]
    print(f'{price_column_name} MSE: {mse}, MAE: {mae}')
    return mae, mse

deepmm_mae, deepmm_mse = compute_error('deepmm_price')
joined['index_price'] = joined['index_price'].astype(float)
index_mae, index_mse = compute_error('index_price')
joined.to_csv('../data/index_deepmm_joined.csv')