from generator import TraceDataGenerator
import settings
import tensorflow as tf
from keras_trace_model import KerasTraceModel
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten


class FeedForwardNN(KerasTraceModel):
    '''
    Simple baseline neural network model which directly processes all of the features.
    '''

    def define_model(self):
        self.model_input, rfq_features, rfq_feature_indices, trade_features, trade_features_to_indices = self.embed_features()
        trade_features = Flatten()(trade_features)

        features = tf.concat([rfq_features, trade_features], axis=-1)

        # do something
        layer = BatchNormalization()(features)
        layer = Dropout(self.get_model_setting('dropout'))(layer)
        layer = Dense(self.get_model_setting('width'), activation=self.get_model_setting('activation'))(layer)
        layer = BatchNormalization()(layer)
        layer = Dropout(self.get_model_setting('dropout'))(layer)
        self.output = Dense(self.trace_train_generator.get_rfq_label_count(), activation='linear')(layer)


def main():
    FeedForwardNN.train_and_evaluate()


if __name__ == "__main__":
    main()