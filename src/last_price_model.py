import numpy as np

from load_data import get_ordinals
from model import TraceModel

class LastPriceModel(TraceModel):
    '''
    Simple baseline model which only uses the previous price with the same figi
    '''

    def fit(self):
        pass

    def create(self):
        pass

    def evaluate_batch(self, X_b):
        _, label_counts, _ = self.train_generator.get_figi_trade_features_and_count(X_b, self.train_generator.get_rfq_labels())
        Y_b_hat = np.full((X_b.shape[0], len(self.train_generator.get_rfq_label_indices())), 0.0)
        rfq_labels = self.train_generator.get_rfq_labels()
        half_assumed_spread = self.model_settings['assumed_spread'] / 2.0
        for j in range(self.train_generator.get_rfq_label_count()):
            rfq_label = rfq_labels[j]
            # We assume that the first group is always the figi
            Y_b_hat[:, j] = self.train_generator.get_trade_feature_values(X_b, rfq_label)[:, 0, -1]

            # We adjust it up or down depending on the side
            def categorical_rfq_feature_equals(ordinal_feature, target_value):
                return np.equal(get_ordinals()[ordinal_feature].index(target_value).as_py(), self.train_generator.get_rfq_feature(X_b, ordinal_feature))
            is_buy = categorical_rfq_feature_equals('buy_sell', 'B')
            is_sell = categorical_rfq_feature_equals('buy_sell', 'S') & categorical_rfq_feature_equals('side', 'C')
            Y_b_hat[:, j] += np.where(is_buy, half_assumed_spread, 0.0)
            Y_b_hat[:, j] -= np.where(is_sell, half_assumed_spread, 0.0)

        Y_b_hat = np.where(label_counts.reshape((-1, 1)) <= 0, np.nan, Y_b_hat)
        return Y_b_hat


def main():
    LastPriceModel.train_and_evaluate()


if __name__ == "__main__":
    main()
