from generator import TraceDataGenerator
from keras_trace_model import KerasTraceModel, Slice
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, \
    Conv1D, Concatenate, Reshape, LSTM
import settings


class SimpleLSTM(KerasTraceModel):
    '''
    Simple baseline neural network model which directly processes all of the features.
    '''

    def define_model(self):
        self.model_input, rfq_features, rfq_feature_indices, trade_features, trade_features_to_indices = self.embed_features()
        trades_lstm = []

        group_count = self.train_generator.generator.get_group_count()
        sequence_length = self.train_generator.generator.get_sequence_length()
        num_filters = self.get_model_setting('trade_conv_filters_1')
        trade_sequences_ungrouped = Reshape((group_count * sequence_length, trade_features.shape[-1]))(trade_features)
        trades_convolved = Conv1D(num_filters, 1, activation=self.get_model_setting('activation'))(trade_sequences_ungrouped)
        trades_convolved = BatchNormalization()(trades_convolved)
        trades_convolved = Dropout(self.get_model_setting('dropout'))(trades_convolved)
        num_filters = self.get_model_setting('trade_conv_filters_2')
        trades_convolved = Conv1D(num_filters, 1, activation=self.get_model_setting('activation'))(trades_convolved)
        trades_convolved_regrouped = Reshape((group_count, sequence_length, num_filters))(trades_convolved)

        for i in range(self.train_generator.generator.get_group_count()):
            print(trade_features.shape)
            trades_group_slice = Slice([0, i, 0, 0],
                                       [-1, 1, trades_convolved_regrouped.shape[-2],
                                        trades_convolved_regrouped.shape[-1]])(trades_convolved_regrouped)
            trades_group_slice = Reshape((trades_group_slice.shape[-2],
                                          trades_group_slice.shape[-1]))(trades_group_slice)
            lstm_layer = LSTM(self.get_model_setting('lstm_units'), return_sequences=True)(trades_group_slice)
            lstm_layer = BatchNormalization()(lstm_layer)
            lstm_layer = Dropout(self.get_model_setting('dropout'))(lstm_layer)
            print(f'Shape before second LSTM layer: {lstm_layer.shape}')
            lstm_layer = LSTM(self.get_model_setting('lstm_units'))(lstm_layer)
            trades_lstm.append(lstm_layer)
        trades_concat = Concatenate(axis=-1)(trades_lstm)
        embedded_features = Concatenate(axis=-1)([rfq_features, trades_concat])
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
    SimpleLSTM.train_and_evaluate()


if __name__ == "__main__":
    main()
