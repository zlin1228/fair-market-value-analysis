import settings
settings.override_if_main(__name__, 2)
from performance_helper import create_profiler
import generator
from trade_history import TradeHistory
from load_data import get_initial_trades

import sys
import pyarrow as pa

from model import TraceModel
from feed_forward_nn import FeedForwardNN
from residual_nn import ResidualNN
from SimpleRfq import SimpleRfq
from SimpleLSTM import SimpleLSTM
from SimplestLSTM import SimplestLSTM
from SimpleConvolutional import SimpleConvolutional
from SimpleAttention import SimpleAttention
from ema_model import ExponentialMovingAverageModel,  VolumeWeightedAverageModel, \
    VolumeWeightedExponentialMovingAverageModel
from last_price_model import LastPriceModel

def get_model_class_from_name(name: str) -> TraceModel:
    model_class = eval(name)
    if not issubclass(model_class, TraceModel):
        raise Exception(f"Model class with name: {name} not recognized")
    return model_class

def get_model_class(model_name: str) -> tuple[TraceModel, dict]:
    model_settings = settings.get('$.models')
    if model_name not in model_settings:
        raise ValueError(f"Model name: {model_name} not found in settings")
    model_settings = model_settings[model_name]
    if 'class' not in model_settings:
        raise ValueError(f"Model name: {model_name} does not have a class defined in settings")
    return get_model_class_from_name(model_settings['class']), model_settings


def split_data(trace: pa.table) -> tuple[pa.table, pa.table, pa.table]:
    if settings.get('$.use_date_split'):
        return TraceModel.date_split(trace)
    else:
        return TraceModel.percent_split(trace)

def load_generators() -> tuple[generator.TraceDataGenerator,
                                                        generator.TraceDataGenerator,
                                                        generator.TraceDataGenerator]:
    profile = create_profiler('model.load_generators')
    with profile():
        with profile('load data'):
            trades = get_initial_trades()
            trade_history = TradeHistory()
            trade_history.append(trades)
        training_data, validation_data, test_data = split_data(trades)
        with profile('training_data_generator'):
            training_data_generator = generator.TraceDataGenerator(trade_history, Z=training_data)
        with profile('validation_data_generator'):
            validation_data_generator = generator.TraceDataGenerator(trade_history, Z=validation_data)
        with profile('test_data_generator'):
            test_data_generator = generator.TraceDataGenerator(trade_history, Z=test_data)

        return training_data_generator, validation_data_generator, test_data_generator


def load_data_and_create_model(model_name: str) -> tuple[TraceModel, generator.TraceDataGenerator]:
    profile = create_profiler('model.load_data_and_create_model')

    cls, model_settings = get_model_class(model_name)

    if 'overrides' in model_settings:
        settings.override(model_settings['overrides'])

    training_data_generator, validation_data_generator, test_data_generator = \
            load_generators()

    if settings.get('$.filter_for_evaluation.apply_filter'):
        with profile('filter_for_evaluation'):
            test_data_generator.filter_for_evaluation()

    model = cls(model_name, model_settings, training_data_generator, validation_data_generator, test_data_generator)
    with profile('create model'):
        model.create()

    return model, test_data_generator

def train_and_evaluate(class_name, train=None):
    profile = create_profiler('model.train_and_evaluate')
    with profile('load model'):
        model, test_data_generator = load_data_and_create_model(class_name)
    if train is None and settings.get('$.train') or train:
        with profile('fit'):
            model.fit()
    with profile('evaluate'):
        loss = model.evaluate()
    loss['labels'] = test_data_generator.get_rfq_labels()
    print('Test loss:')
    print(loss)


def main():
    if len(sys.argv) < 2:
        print('Usage: <keras model name or TraceModel class name> <settings_overrides (optional)>')
        exit()
    model_name = sys.argv[1]
    train_and_evaluate(model_name)

if __name__ == "__main__":
    main()
