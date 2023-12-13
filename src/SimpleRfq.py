from generator import TraceDataGenerator
import tensorflow as tf
from keras_trace_model import KerasTraceModel, Slice
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, \
    Conv1D, Reshape, Flatten, MultiHeadAttention, Concatenate
import settings

class SequenceStd(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SequenceStd, self).__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, sequence):
        '''
        We assume the sequence is of shape: (batch_size, sequence_length, feature_count)
        This layer returns the standard deviation of each of the features along the
        sequence_length
        '''
        return tf.math.reduce_std(sequence, axis=-2)


class SimpleRfq(KerasTraceModel):
    '''
    Simple baseline neural network model which directly processes all of the features.
    '''
    def define_model(self):
        selected_features = self.get_model_setting('rfq_features')
        self.model_input, rfq_features, rfq_feature_indices, trade_features, trade_features_to_indices = self.embed_features()

        if self.get_model_setting('select_rfq_features'):
            slices = []
            for f in selected_features:
                i = rfq_feature_indices[f]
                s = Slice([0, i[0]], [-1, i[1]-i[0]])(rfq_features)
                slices.append(s)
            rfq_features = Concatenate(axis=-1)(slices)

        if self.get_model_setting('multiply_by_std'):
            trade_label_indices = map(lambda x: trade_features_to_indices[x], self.trace_train_generator.get_rfq_labels())
            trade_label_slices = []
            for i in trade_label_indices:
                left = i[0]
                size = i[1]-i[0]
                trade_label = Slice([0, 0, 0, left], [-1, 1, -1, size])(trade_features)
                trade_label = Reshape((self.trace_train_generator.sequence_length, size))(trade_label)
                trade_label_slices.append(trade_label)
            trade_labels = Concatenate(axis=-1)(trade_label_slices)
            trade_labels_std = SequenceStd()(trade_labels)

            rfq_features = Concatenate(axis=-1)([rfq_features, trade_labels_std])

        rfq_embedding = BatchNormalization()(rfq_features)
        rfq_embedding = Dropout(self.get_model_setting('dropout'))(rfq_embedding)
        rfq_embedding = Dense(self.get_model_setting('dense_width'), activation=self.get_model_setting('activation'))(rfq_embedding)
        rfq_embedding = BatchNormalization()(rfq_embedding)
        previous = rfq_embedding = Dropout(self.get_model_setting('dropout'))(rfq_embedding)

        for i in range(self.get_model_setting('residual_layers')):
            rfq_embedding = Dense(self.get_model_setting('dense_width'), activation=self.get_model_setting('activation'))(rfq_embedding)
            rfq_embedding = BatchNormalization()(rfq_embedding)
            rfq_embedding = Dropout(self.get_model_setting('dropout'))(rfq_embedding)
            rfq_embedding = Dense(self.get_model_setting('dense_width'), activation=self.get_model_setting('activation'))(rfq_embedding)
            rfq_embedding = previous = previous + rfq_embedding
            rfq_embedding = BatchNormalization()(rfq_embedding)
            rfq_embedding = Dropout(self.get_model_setting('dropout'))(rfq_embedding)

        self.output = Dense(self.trace_train_generator.get_rfq_label_count(),
                            activation='linear', use_bias=False)(rfq_embedding)

def main():
    SimpleRfq.train_and_evaluate()


if __name__ == "__main__":
    main()
