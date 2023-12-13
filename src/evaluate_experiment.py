from types import ModuleType, FunctionType

import json
import numpy as np
import sys

import argparse

import settings

args = None


# Example:
# python evaluate_experiment.py -o '{Setting overrides}' -m last_price residual_nn_last_price_normalization residual_nn -j ../experiments/evaluate_test.jso

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model names.')
    parser.add_argument('-m', '--model_names', nargs='+', help='Model names to process')
    parser.add_argument('-j', '--output_json_path', action='store', help='Output json path')
    parser.add_argument('-o', '--override', action='store', default=None,
                                                help='Override option')

    args = parser.parse_args(sys.argv[1:])
    if args.override:
        print(f"Override: {args.override}")
        settings.override(json.loads(args.override))


from settings import checkpoint_settings, restore_settings
from train_evaluate import load_data_and_create_model
from performance_helper import create_profiler
import gc
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


'''
    In the metrics, there are histograms. A histogram contains lists.
    To show a list in one line, we need to convert the list to a string.
    This function takes a dict, and converts all lists of numbers to strings.
'''
def convert_list_to_string(metrics):
    if isinstance(metrics, dict):
        for key in metrics.keys():
            metrics[key] = convert_list_to_string(metrics[key])
    elif isinstance(metrics, list):
        metrics = '[' + ', '.join(map(str, metrics)) + ']'
    return metrics

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

def get_intersection(na_filters):
    return np.all(np.array(na_filters), axis=0)

def evaluate_experiment(model_names, output_json_path):
    profile = create_profiler('evaluate_experiment')
    evaluation_results = {}
    model = None

    for model_name in model_names:
        if model is not None:
            with profile('===================================\nCleaning previous model data'):
                print('Size of model: ' + str(getsize(model)))
                model.cleanup()
                gc.collect()
                print ('Size of evaluation_results object: ' + str(getsize(evaluation_results)))

        settings_checkpoint = checkpoint_settings()
        with profile('Creating model with name: ' + model_name):
            model, _ = load_data_and_create_model(model_name)

        if settings.get('$.train'):
            model.fit()

        # get the evaluation results
        evaluation_results[model_name] = model.evaluate_batches()
        restore_settings(settings_checkpoint)

    # Check all counts are the same.
    assert(len(set([evaluation_results[model_name][0].shape[0] for model_name in model_names])) == 1)

    # Get the intersection of na_filters.
    na_filter = get_intersection([evaluation_results[model_name][0] for model_name in model_names])

    # Get losses of models.
    losses = {}
    for model_name in model_names:
        metrics = model.get_metrics(na_filter, *evaluation_results[model_name][1:])
        losses[model_name] = convert_list_to_string(metrics)

    # Save the JSON object to a file
    with open(output_json_path, 'w') as json_file:
        json.dump(losses, json_file, indent=4)

    return losses

def main():
    for model in args.model_names:
        print(f"Model: {model}")

    evaluate_experiment(args.model_names, args.output_json_path)


if __name__ == "__main__":
    main()
