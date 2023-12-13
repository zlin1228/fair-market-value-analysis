import settings
settings.override_if_main(__name__, 2)

import sys
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


def main():
    if len(sys.argv) < 2:
        print('Usage: <model_class> <settings_overrides (optional)>')
        exit()

    model_class_name = sys.argv[1]
    model_class = eval(model_class_name)

    if not issubclass(model_class, TraceModel):
        raise Exception(f"Model class with name: {model_class_name} not recognized")

    model, _ = model_class.load_data_and_create_model()
    model.save()


if __name__ == "__main__":
    main()
