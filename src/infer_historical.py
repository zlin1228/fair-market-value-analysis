import settings
settings.override_if_main(__name__, 2)

import sys
from model import TraceModel
from feed_forward_nn import FeedForwardNN
from SimpleRfq import SimpleRfq
from SimpleLSTM import SimpleLSTM
from SimplestLSTM import SimplestLSTM
from SimpleConvolutional import SimpleConvolutional
from SimpleAttention import SimpleAttention
from ema_model import ExponentialMovingAverageModel
from last_price_model import LastPriceModel
import numpy as np


def main():
    if len(sys.argv) < 2:
        print('Usage: <model_class> <settings_overrides (optional)>')

    model_class_name = sys.argv[1]
    model_class = eval(model_class_name)

    if not issubclass(model_class, TraceModel):
        raise Exception(f"Model class with name: {model_class_name} not recognized")

    Z, Z_Ig, Z_Hy = model_class.infer_historical()
    Z.sort_values(by='report_date')
    Z_Ig.sort_values(by='report_date')
    Z_Hy.sort_values(by='report_date')

    def calculate_stats(my_Z, name, file_name):
        print(f"{name} Stats for {model_class_name}:")
        e = np.abs(my_Z.price - my_Z.deepmm_price)
        mse = np.sum(e*e)/e.shape[0]
        rmse = np.sqrt(mse)
        mae = np.sum(e)/e.shape[0]
        print(f'Count: {e.shape[0]}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
        my_Z.to_csv(file_name)

    calculate_stats(Z, "Overall", f"../data/Deep_MM_sample_{model_class_name}.csv")
    calculate_stats(Z_Ig, "Investment Grade", f"../data/Deep_MM_sample_IG_{model_class_name}.csv")
    calculate_stats(Z_Hy, "High Yield", f"../data/Deep_MM_sample_HY_{model_class_name}.csv")


if __name__ == "__main__":
    main()
