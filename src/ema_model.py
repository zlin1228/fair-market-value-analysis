from model import TraceModel
import numpy as np
import settings
from generator import TraceDataGenerator


def exponential_moving_average(generator: TraceDataGenerator, X_b):
    # Based on:
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    labels, label_counts, quantities = generator.get_figi_trade_features_and_count(X_b, generator.get_rfq_labels())
    sequence_length = settings.get('$.data.trades.sequence_length')
    alpha = 2.0 / (sequence_length + 1)
    weights = np.power((1 - alpha), np.arange(sequence_length))
    weights = np.flip(weights)
    weighted_label_sum = np.sum(np.multiply(labels, weights.reshape(-1, 1)), axis=1)

    is_present = np.flip(
        np.tile(np.arange(sequence_length), label_counts.shape[0]).reshape((label_counts.shape[0], -1)),
        axis=1) < label_counts.reshape((label_counts.shape[0], 1))
    sum_weights = np.sum(np.multiply(is_present, weights), axis=1)
    Y_b_hat = np.divide(weighted_label_sum, sum_weights.reshape((sum_weights.shape[0], 1)))
    Y_b_hat = np.where(np.isinf(Y_b_hat), np.nan, Y_b_hat)
    return Y_b_hat


def volume_weighted_exponential_moving_average(generator: TraceDataGenerator, X_b):
    # Based on:
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    labels, label_counts, quantities = generator.get_figi_trade_features_and_count(X_b, generator.get_rfq_labels())
    sequence_length = settings.get('$.data.trades.sequence_length')
    alpha = 2.0 / (sequence_length + 1)
    weights = np.power((1 - alpha), np.arange(sequence_length))
    weights = np.flip(weights)
    weighted_label_sum = np.sum(np.multiply(labels, np.multiply(quantities, weights).reshape(-1, settings.get('$.data.trades.sequence_length'), 1)), axis=1)
    sum_weights = np.sum(np.multiply(quantities, weights), axis=1)
    Y_b_hat = np.divide(weighted_label_sum, sum_weights.reshape((sum_weights.shape[0], 1)))
    Y_b_hat = np.where(np.isinf(Y_b_hat), np.nan, Y_b_hat)
    return Y_b_hat


#VWAP
def volume_weighted_average_label(generator: TraceDataGenerator, X_b):
    # Based on:
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    labels, label_counts, quantities = generator.get_figi_trade_features_and_count(X_b, generator.get_rfq_labels())
    weighted_label_sum = np.sum(np.multiply(labels, quantities.reshape((-1, settings.get('$.data.trades.sequence_length'), 1))),
                                axis=1)
    total_quantities = np.sum(quantities, axis=1).reshape(-1, 1)
    # Volume Weighted Average Label
    Y_b_hat = np.divide(weighted_label_sum, total_quantities)
    Y_b_hat = np.where(np.isinf(Y_b_hat), np.nan, Y_b_hat)
    return Y_b_hat


class ExponentialMovingAverageModel(TraceModel):
    '''
    Simple baseline model which only uses the exponential moving average
    '''

    def fit(self):
        pass

    def create(self):
        pass

    def evaluate_batch(self, X_b):
        return exponential_moving_average(self.test_generator, X_b)


class VolumeWeightedAverageModel(TraceModel):
    '''
    Simple baseline model which only uses VWAP
    '''

    def fit(self):
        pass

    def create(self):
        pass

    def evaluate_batch(self, X_b):
        return volume_weighted_average_label(self.test_generator, X_b)


class VolumeWeightedExponentialMovingAverageModel(TraceModel):
    '''
    Simple baseline model which only uses VWEMA
    '''

    def fit(self):
        pass

    def create(self):
        pass

    def evaluate_batch(self, X_b):
        return volume_weighted_exponential_moving_average(self.test_generator, X_b)


def main():
    ExponentialMovingAverageModel.train_and_evaluate()


if __name__ == "__main__":
    main()
