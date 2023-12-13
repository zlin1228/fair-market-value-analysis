import json
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
import numpy as np
import argparse

# Initialize the ArgumentParser object
parser = argparse.ArgumentParser(description='Generate HTML report from JSON data.')

# Add the arguments for input and output file paths
parser.add_argument('-i', '--input', type=str, required=True, help='Path to input JSON file.')
parser.add_argument('-o', '--output', type=str, required=True, help='Path to output HTML file.')

# Parse the arguments
args = parser.parse_args()

# Read the JSON data from the input file path specified as a command line argument
with open(args.input, 'r') as f:
    json_data = f.read()

def generate_buy_sell_chart(json_data, output_path):
    # Load json data into a dictionary
    data = json.loads(json_data)

    # Create a Jinja2 environment and specify the templates directory
    env = Environment(loader=FileSystemLoader('.'))

    models = list(data.keys())

    min_tile = 2
    max_tile = 98

    tiles = np.array(json.loads(data[models[0]]['overall']['error_tiles']['tiles'])[min_tile:max_tile])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)  # 10 by 6 inches
    patterns = ['--', '-', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']

    model_name_map = {
        'last_price': 'Last Price Heuristic',
        'residual_nn_last_price_normalization': 'Deep MM'
    }

    def get_model_print_name(model):
        return model_name_map[model] if model in model_name_map else model

    percentile_label_models = ['residual_nn_last_price_normalization']
    def plot_models(tiles, category, color):
        for i, model in enumerate(models):
            tiles_data = data[model][category]['error_tiles']
            values = json.loads(tiles_data['values'])[min_tile:max_tile]

            # Name to print out for map by using model_name_map
            ax.plot(values, tiles, alpha=0.5, color=color,
                    label=f'{category.capitalize()} {get_model_print_name(model)}', linestyle=patterns[i])

            if model in percentile_label_models:
                # Adding annotation

                def label_percentile(percentile):
                    percentile_index = list(tiles).index(percentile)
                    percentile_value = values[percentile_index]

                    ax.annotate(f'{percentile}th percentile\n(${percentile_value:.2f})',
                                xy=(percentile_value, percentile),
                                xytext=(percentile_value, percentile+7),
                                arrowprops=dict(facecolor='black', shrink=0.05))

                label_percentile(80)

    plt.title(f'Buy Sell Model Residual Percentiles')
    plot_models(tiles, 'sell', color='red')
    plot_models(100 - tiles, 'buy', color='green')

    # You can adjust the tick marks as you see fit.
    # For instance, if your x-axis data is between 0 and 100, you might want ticks every 10.
    #ax.xaxis.set_ticks(np.arange(min(tiles), max(tiles), 10))  # Replace 10 with the desired step size for x-axis
    #ax.yaxis.set_ticks(np.arange(min(values), max(values), 10))  # Replace 10 with the desired step size for y-axis

    ax.xaxis.set_major_formatter('${x:1.2f}')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylabel('Percentile')  # Label y axis with 'Percentile'
    ax.set_xlabel('Model Error')  # Label x axis with 'Residual'
    ax.grid(True)  # This turns on the gridlines
    plt.legend()
    # Save the figure to the output path
    fig.savefig(f'{output_path}/buy_sell_error_tiles.png', bbox_inches='tight')

# Generate the report
generate_buy_sell_chart(json_data, args.output)