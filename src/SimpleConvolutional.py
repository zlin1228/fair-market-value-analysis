from generator import TraceDataGenerator
import settings
import tensorflow as tf
from keras_trace_model import KerasTraceModel, Slice
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, \
    Conv1D, Concatenate, Reshape, MaxPooling1D


class MeanPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError("Expected two inputs: [group_slice, group_convolved]")
        group_slice = inputs[0]
        group_convolved = inputs[1]
        neqs = tf.math.not_equal(group_slice, 0.0)
        reduced = tf.math.reduce_any(neqs, axis=-1)
        reduced_ints = tf.cast(reduced, tf.float32)
        valid_counts = tf.math.reduce_sum(reduced_ints, axis=-1)
        valid_counts = tf.reshape(valid_counts, (-1, 1))
        sum_group_convolved = tf.reduce_sum(group_convolved, axis=1)
        print(f'valid_counts shape: {valid_counts}')
        print(f'sum_group_convolved shape: {sum_group_convolved.shape}')
        group_mean = tf.math.divide(sum_group_convolved, valid_counts)

        return group_mean


class SimpleConvolutional(KerasTraceModel):
    '''
    Simple baseline neural network model which directly processes all of the features.
    '''

    def define_model(self):
        self.model_input, rfq_features, rfq_feature_indices, trade_features, trade_features_to_indices = self.embed_features()

        #rfq_features = BatchNormalization()(rfq_features)
        #rfq_features = Dropout(self.get_setting('dropout'))(rfq_features)
        #rfq_features = Dense(self.get_setting('rfq_feature_dense_layer_width'),
        #                     activation=self.get_setting('activation'))(rfq_features)
        #rfq_features = BatchNormalization()(rfq_features)
        #rfq_features = Dropout(self.get_setting('dropout'))(rfq_features)
        #rfq_features = Dense(self.get_setting('rfq_feature_dense_layer_width'),
        #                     activation=self.get_setting('activation'))(rfq_features)

        group_count = self.train_generator.generator.get_group_count()
        sequence_length = self.train_generator.generator.get_sequence_length()
        num_filters = self.get_model_setting('trade_conv_filters')
        trade_sequences_ungrouped = Reshape((group_count*sequence_length, trade_features.shape[-1]))(trade_features)
        trades_convolved = Conv1D(num_filters, 1, activation=self.get_model_setting('activation'))(trade_sequences_ungrouped)
        trades_convolved = BatchNormalization()(trades_convolved)
        trades_convolved = Dropout(self.get_model_setting('dropout'))(trades_convolved)
        trades_convolved = Conv1D(num_filters, 1, activation=self.get_model_setting('activation'))(trades_convolved)
        trades_convolved_regrouped = Reshape((group_count, sequence_length, num_filters))(trades_convolved)
        trades_pooled = []
        for i in range(self.train_generator.generator.get_group_count()):
            convolved_trades_group_slice = Slice([0, i, 0, 0],
                                [-1, 1, sequence_length, num_filters])(trades_convolved_regrouped)
            convolved_trades_group_slice = Reshape((convolved_trades_group_slice.shape[-2],
                                                    convolved_trades_group_slice.shape[-1]))(convolved_trades_group_slice)
            group_max_pooled = MaxPooling1D(sequence_length)(convolved_trades_group_slice)
            group_max_pooled = Reshape((num_filters,))(group_max_pooled)
            trades_pooled.append(group_max_pooled)
        trades_pooled = Concatenate(axis=-1)(trades_pooled)
        '''
        group_features = []
        for i in range(self.train_generator.generator.get_group_count()):
            group_slice = Slice([0, i, 0, 0],
                                [-1, 1, self.train_generator.generator.get_sequence_length(),
                                 trade_features.shape[-1]])(trade_features)
            group_slice = Reshape((group_slice.shape[-2], group_slice.shape[-1]))(group_slice)
            # Get count of valid rows so that we can normalize the summation we will do.
            print(f'group_slice shape: {group_slice.shape}')
            group_convolved = Conv1D(self.get_setting('trade_conv_filters'), 1,
                                     activation=self.get_setting('activation'))(group_slice)
            print(f'group_convolved shape after first convolution: {group_convolved.shape}')
            group_convolved = BatchNormalization()(group_convolved)
            group_convolved = Dropout(self.get_setting('dropout'))(group_convolved)
            group_convolved = Conv1D(self.get_setting('trade_conv_filters'), 1,
                                     activation=self.get_setting('activation'))(group_convolved)
            print(f'group_convolved shape after second convolution: {group_convolved.shape}')
            #group_mean = MeanPooling()([group_slice, group_convolved])
            group_mean = MaxPooling1D(group_convolved.shape[-2])(group_convolved)
            group_mean = Reshape((group_mean.shape[-1],))(group_mean)
            group_features.append(group_mean)
        
        convolved_trade_features = Concatenate(axis=-1)(group_features)
        '''

        embedded_features = Concatenate(axis=-1)([rfq_features, trades_pooled])
        embedded_features = BatchNormalization()(embedded_features)
        embedded_features = Dropout(self.get_model_setting('dropout'))(embedded_features)
        embedded_features = Dense(self.get_model_setting('embedded_features_dense_layer_width'),
                                  activation=self.get_model_setting('activation'))(embedded_features)
        embedded_features = BatchNormalization()(embedded_features)
        embedded_features = Dropout(self.get_model_setting('dropout'))(embedded_features)
        # Final output of the model
        self.output = Dense(self.train_generator.generator.get_rfq_label_count(), activation='linear')(
            embedded_features)


def main():
    SimpleConvolutional.train_and_evaluate()


if __name__ == "__main__":
    main()
