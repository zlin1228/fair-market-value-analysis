from generator import TraceDataGenerator
import settings
import tensorflow as tf
from keras_trace_model import KerasTraceModel, Slice
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, \
    Conv1D, Reshape, Flatten, MultiHeadAttention


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.embedding_width = self.get_setting('rfq_query_width')
        self.heads = self.get_setting('heads')
        self.head_width = int(self.embedding_width / self.heads)
        self.dropout = self.get_setting('dropout')

        self.attention_layer = MultiHeadAttention(num_heads=self.heads, key_dim=self.head_width,
                                                  dropout=self.dropout)

        def create_conv1d(activation, filters=self.head_width, use_bias=True):
            return Conv1D(filters, 1, activation=activation, use_bias=use_bias)
        self.ff_conv1d_1 = create_conv1d(filters=self.embedding_width, activation=self.get_setting('activation'))
        self.dropout = Dropout(self.dropout)
        self.ff_conv1d_2 = create_conv1d(filters=self.embedding_width, activation=self.get_setting('activation'))

        self.batch_normalization_post_attention = BatchNormalization()
        self.batch_normalization_post_ff_nn = BatchNormalization()

    def get_setting(self, setting):
        return settings.get(f'$.keras_models.implementations.att.{setting}')

    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        '''
        inputs[0]: The embedding tensor from which the target query will be derived
        inputs[1]: The embedding tensor from which the source key will be derived. If inputs[0] == inputs[1],
                    then this is a self-attention layer
        '''
        target_embeddings = inputs[0]
        source_embeddings = inputs[1]

        attention_output = self.attention_layer(target_embeddings, source_embeddings)
        post_attention_value = self.batch_normalization_post_attention(attention_output + target_embeddings)
        post_attention_ff_nn = self.dropout(post_attention_value)
        post_attention_ff_nn = self.ff_conv1d_1(post_attention_ff_nn)
        post_attention_ff_nn = self.dropout(post_attention_ff_nn)
        post_attention_ff_nn = self.ff_conv1d_2(post_attention_ff_nn)
        post_attention_ff_nn = self.batch_normalization_post_ff_nn(post_attention_ff_nn + post_attention_value)

        return post_attention_ff_nn


class SimpleAttention(KerasTraceModel):
    '''
    Simple baseline neural network model which directly processes all of the features.
    '''

    def define_model(self):
        self.model_input, rfq_features, rfq_feature_indices,  trade_features, trade_features_to_indices = self.embed_features()
        sequence_length = self.train_generator.generator.get_sequence_length()
        # Get only the figi-matching group, assuming the figi matching group is at the begining
        trade_features = Slice([0, 0, 0, 0], [-1, 1, -1, -1])(trade_features)
        trade_features = Reshape((sequence_length, -1))(trade_features)

        trade_embedding = Conv1D(self.get_model_setting('rfq_query_width'), 1, activation=self.get_model_setting('activation'))(trade_features)
        trade_embedding = BatchNormalization()(trade_embedding)
        trade_embedding = Dropout(self.get_model_setting('dropout'))(trade_embedding)
        trade_embedding = Conv1D(self.get_model_setting('rfq_query_width'), 1, activation=self.get_model_setting('activation'))(trade_embedding)

        rfq_embedding = Dense(self.get_model_setting('rfq_query_width'), activation=self.get_model_setting('activation'))(rfq_features)
        rfq_embedding = BatchNormalization()(rfq_embedding)
        rfq_embedding = Dropout(self.get_model_setting('dropout'))(rfq_embedding)
        rfq_embedding = Dense(self.get_model_setting('rfq_query_width'), activation=self.get_model_setting('activation'))(rfq_embedding)
        assert (len(rfq_embedding.shape) == 2)
        new_rfq_embedding_shape = (1, rfq_embedding.shape[1])
        rfq_embedding = Reshape(new_rfq_embedding_shape)(rfq_embedding)

        trade_self_attention_values = AttentionLayer()([trade_embedding, trade_embedding])
        attention_values = AttentionLayer()([rfq_embedding, trade_self_attention_values])

        repeated_self_attention_layer = AttentionLayer()
        repeated_attention_layer = AttentionLayer()
        for i in range(5):
            trade_self_attention_values = repeated_self_attention_layer(
                                            [trade_self_attention_values, trade_self_attention_values])
            attention_values = repeated_attention_layer([attention_values, trade_self_attention_values])

        '''
        repeated_self_attention_layer = AttentionLayer()
        repeated_attention_layer = AttentionLayer()
        for i in range(10):
            trade_self_attention_values = repeated_self_attention_layer(
                                            [trade_self_attention_values, trade_self_attention_values])
            attention_values = repeated_attention_layer([attention_values, trade_self_attention_values])
        '''

        attention_values = Flatten()(attention_values)

        self.output = Dense(self.trace_train_generator.get_rfq_label_count(), activation='linear', use_bias=False)(attention_values)



def main():
    SimpleAttention.train_and_evaluate()


if __name__ == "__main__":
    main()
