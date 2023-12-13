import numpy as np
import pandas as pd
import math

from feed_forward_nn import FeedForwardNN
from generator import TraceDataGenerator
from helpers import get_isin_from_cusip
from trace_data import get_trace_data
import helpers
import datetime
import settings

model = None

# evaluate batch index
def evaluate_batch_index(index):
    Z_b = model.test_generator.get_Z_b(index)
    X_b, Y_b = model.test_generator.generate_batch_np(Z_b)
    assert(Z_b.shape[0] == Y_b.shape[0])
    Y_b_hat = model.evaluate_batch(X_b)

    # We only want to count the error for
    # cases where the Y_hat was valid (where
    # there was at least one related trade which
    # happened before each target RFQ typically)

    na_filter = ~np.isnan(Y_b_hat).any(axis=1)
    X_b = X_b[na_filter]
    Z_b = Z_b[na_filter]
    Y_b = Y_b[na_filter]
    count = Y_b.shape[0]
    Y_b_hat = Y_b_hat[na_filter]
    quantities = model.test_generator.get_rfq_feature(X_b, 'quantity')

    assert(Z_b.shape[0] == Y_b.shape[0])
    assert(Y_b.shape[0] == Y_b_hat.shape[0])

    return count, Z_b, Y_b, Y_b_hat, quantities

# evaluate batch_wrapper
def evaluate_batch_wrapper(index):
    count, Z_b, Y_b, Y_b_hat, quantities = evaluate_batch_index(index)
    error = Y_b - Y_b_hat
    mse = np.mean(error * error)
    Z_df = model.test_generator.get_Z_df_from_Z_b(Z_b)
    diff_price = Z_df.groupby('figi').diff().apply(abs).join(Z_df['figi']).groupby('figi').mean()['price'].mean()
    date = Z_df['report_date'].iloc[Z_df.shape[0] // 2]
    return [mse, diff_price, date]

# evaluate batch
def evaluate():
    data_list = list()
    for i in range(model.test_generator.__len__()):
        print(i, model.test_generator.__len__())
        data_list.append(evaluate_batch_wrapper(i))
    return data_list

if __name__ == "__main__":
    # load trace
    print('Loading TRACE')
    trace, ordinals = get_trace_data()
    trace = trace.sort_values(by='report_date', ascending=True)

    # load model
    generator = TraceDataGenerator(trace, ordinals, should_shuffle=False)
    model = FeedForwardNN(generator, generator, generator)
    model.create()

    #analyze by histogram
    data_list = evaluate()

    mse = list()
    diff_price = list()
    date = list()
    for data in data_list:
        mse.append(data[0])
        diff_price.append(data[1])
        date.append(datetime.date.fromtimestamp(data[2] // 1e9).strftime('%Y-%m-%d'))
    
    plt.figure(figsize=(30, 15))
    plt.gca().set_xlabel('report_date', fontsize=20)
    plt.gca().set_ylabel('mse/diff_price', fontsize=20)
    plt.gca().set_xticklabels(date, rotation=90)
    plt.bar(date, mse, width=-0.4, align='edge')
    plt.bar(date, diff_price, width=0.4, align='edge')
    plt.savefig('histogram.png')