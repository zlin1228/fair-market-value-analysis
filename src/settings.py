from copy import deepcopy
import json
import logging
import sys

from jsonpath_ng import parse

__defalt_settings = {
    'environment': 'development',
    'environment_settings': {
        'development': {
            'authentication': {
                'api': 'https://np6y8vv9k1.execute-api.us-west-2.amazonaws.com/Dev'
            },
            'memorydb': {
                'api': 'https://cw908w7sxc.execute-api.us-west-2.amazonaws.com/Dev',
                'finra_cache_key': 'finra_cache'
            },
            'model_bucket': 'deepmm.models.dev',
        },
        'production': {
            'authentication': {
                'api': 'https://zux00mas3a.execute-api.us-east-1.amazonaws.com/Prod'
            },
            'memorydb': {
                'api': 'https://ah9m2meigb.execute-api.us-east-1.amazonaws.com/Prod',
                'finra_cache_key': 'finra_cache'
            },
            'model_bucket': 'deepmm.models.prod',
        }
    },

    'logging': {
        'level': {
            'console': {
                'finra_btds_multicast': 'INFO',
                'finra_144a_multicast': 'INFO',
                'finra': 'INFO',
                'fmv_websocket_server': 'INFO',
                'fmv_websocket_server__data': 'INFO',
                'fmv_websocket_server__websockets': 'INFO',
                'health_checker': 'INFO',
                'health_checker__memorydb': 'INFO',
                'log': 'INFO',
                'memory_db': 'INFO',
                'settings': 'INFO'
            },
            'logfile': {
                'finra_btds_multicast': 1,
                'finra_144a_multicast': 1,
                'finra': 1,
                'fmv_websocket_server': 1,
                'fmv_websocket_server__data': 1,
                'fmv_websocket_server__websockets': 1,
                'health_checker': 'INFO',
                'health_checker__memorydb': 'INFO',
                'log': 1,
                'memory_db': 1,
                'settings': 1
            }
        }
    },

    'server': {
        'fmv': {
            'initial_model_price_count': 500,
            'interval': {
                # data_loop interval must be > 0 so we aren't forced to process messages one at a time
                # (allows for batch processing during high-volume bursts)
                'data_loop': 0.5,
                'heartbeat': 10,
                'memorydb': 10,
                'model': 10
            },
            'memorydb': {
                'propogation_wait_time': 60,
                'batch_size': 1000
            },
            'authorize_websocket': True,
            'model': 'residual_nn',
            'port': 8855
        }
    },

    'health_checker': {
        'fmv': {
            'interval': 60
        }
    },

    'finra_btds_multicast': {
        'ip': '224.0.17.34',
        'port': 55265
    },

    'finra_144a_multicast': {
        'ip': '224.3.0.21',
        'port': 55267
    },

    'local_data_path': '../data',
    's3_cache_path': './s3-cache',
    'finra_timezone': 'US/Eastern',

    'raw_data_path': 's3://deepmm.data/',
    'data_path': 's3://deepmm.parquet/v0.7.1.zip',
    'bond_index_path': 's3://deepmm.public/bond_data.json', # path to bond index file used by the UI
    'data_compression_level': 4, # compression level for new 'parquet' data, possible values: 0-9, 4 seems like the best trade-off for our data

    'tolerate_missing_reference_data': False,

    'use_date_split': True,
    # The dates for each data set, which are inclusive.
    # So for example the date range 2019-01-01 to 2019-01-31 would include the days until midnight 2019-01-30, because
    # all timestamps on 2019-01-31 that happen throughout the day would be greater then that midnight timestamp.
    #
    # !! All of the data sets must be non-overlapping with the other data sets, or this will cause unexpected behavior.!!
    # (There's currently no checking in the code to make sure they are non-overlapping).
    'training_date_ranges': [
        {'start': '2019-08-02', 'end': '2020-03-24'},
        {'start': '2020-04-18', 'end': '2023-07-11'},
    ],
    'validation_date_ranges': [
        {'start': '2020-04-01', 'end': '2020-04-17'},
        {'start': '2019-04-02', 'end': '2019-08-01'},
    ],
    'test_date_ranges': [
        {'start': '2020-03-25', 'end': '2020-03-31'},
        {'start': '2019-01-01', 'end': '2019-04-01'},
    ],

    'multicast_data_split': 0.95,
    'training_data_split': 0.9,
    'validation_data_split': 0.95,
    'train': True,
    'enable_inference': False,
    'enable_bondcliq': False,
    'ratings_enabled': True,
    'profile': False,

    'batch_size': 4096,
    'should_shuffle': True,
    'calculate_spread': True,

    # When evaluating models, how many tiles (as in percentiles) to compute for the error.
    'tiles': 100,

    'typical_row_skip_columns': {'figi', 'quantity', 'price','coupon','maturity','rating','issue_date','issuer',
                                 'industry','sector',
                                 'yield', 'execution_date', 'report_date'},

    'test_generator_figi_z_filter': None,

    'ema_normalization': False,

    'filter_for_evaluation': {
        'apply_filter': False,
        'minimum_quantity': 0,
        'maximum_quantity': sys.maxsize,
        'minimum_liquidity': 0,
        'maximum_liquidity': sys.maxsize
    },

    'data': {
        'trades': {
            'columns': [    'ats_indicator',
                            'buy_sell',
                            'coupon',
                            'cusip',
                            'execution_date',
                            'figi',
                            'industry',
                            'issue_date',
                            'issuer',
                            'maturity',
                            'outstanding',
                            'price',
                            'quantity',
                            'rating',
                            'report_date',
                            'sector',
                            'side',
                            'yield'],
            'groups': ['figi', 'issuer', 'industry', 'sector', 'rating'],
            'remove_features': ['cusip', 'figi', 'issuer', 'industry', 'sector', 'yield', '__index_level_0__'],
            'rfq_labels': ['price'],
            'replace_nulls': {'yield': -100.0},
            'sequence_length': 10,
            'proportion_of_trades_to_load': 1.0
        },
        'quotes': {
            'features': [   'figi',
                            'party_id',
                            'entry_type',
                            'entry_date',
                            'price',
                            'quantity'],
            "dealer_window": 10
        },
        'ratings': {
            'investment_grade_ratings': [   'Aaa',
                                            'Aa1',
                                            'Aa2',
                                            'Aa3',
                                            'A1',
                                            'A2',
                                            'A3',
                                            'Baa1',
                                            'Baa2',
                                            'Baa3']
        }
    },

    'models': {
        'ema': {'class': 'ExponentialMovingAverageModel'},
        'vwap': {'class': 'VolumeWeightedAverageModel'},
        'vwap_ema': {'class': 'VolumeWeightedExponentialMovingAverageModel'},
        'last_price': {'class': 'LastPriceModel','assumed_spread': 0.29},
        'feed_forward_nn': {
            'class': 'FeedForwardNN',
            #'width': 16384,
            'width': 256,
            'activation': 'sigmoid',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'history_filepath': 'history/feed_forward_nn'
        },
        'residual_nn': {
            'class': 'ResidualNN',
            'width': 1024,
            'residual_blocks': 25,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'history_filepath': 'history/residual_nn'
        },
        'residual_nn_log_embed_dims': {
            'class': 'ResidualNN',
            'overrides':
                {
                    '$.keras_models.use_log_embedding_for_ordinals':True
                },
            'width': 1024,
            'residual_blocks': 25,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'history_filepath': 'history/residual_nn_log_embed_dims'
        },
        # Uses the combined btds and 144A training data
        'residual_last_price_combined': {
            'class': 'ResidualNN',
            'overrides': 
                {
                    'data_path': 's3://deepmm.trace/data/v0.5/combined/latest.zip'      
                },
            'width': 1024,
            'residual_blocks': 25,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'history_filepath': 'history/residual_nn_last_price_combined'
        },
        'residual_nn_ratings_control': {
            'class': 'ResidualNN',
            'overrides':
                {
                    "$.ratings_enabled": True
                },
            'width': 1024,
            'residual_blocks': 25,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'history_filepath': 'history/residual_nn_ratings_control'
        },
        'residual_nn_ratings_treatment': {
            'class': 'ResidualNN',
            'overrides':
                {
                    "$.ratings_enabled": False
                },
            'width': 1024,
            'residual_blocks': 25,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'history_filepath': 'history/residual_nn_ratings_treatment'
        },
        'residual_nn_no_normalization': {
            'class': 'ResidualNN',
            'overrides': {"$.keras_models.normalize": False},
            'width': 1024,
            'residual_blocks': 25,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5000,
            'optimizer': 'adam',
            'save_full_model_filepath': 'models/residual_nn.v0.3.4.50',
            'history_filepath': 'history/residual_nn.v0.4.5.2048'
        },
        'residual_nn_integration_test': {
            'class': 'ResidualNN',
            'width': 16,
            'residual_blocks': 2,
            'activation': 'relu',
            'dropout': 0.5,
            'epochs': 5,
            'optimizer': 'adam',
            'save_full_model_filepath': 'models/residual_nn_integration_test',
            'history_filepath': 'history/residual_nn_integration_test'
        },
        'conv': {
            'class': 'SimpleConvolutional',
            'rfq_feature_dense_layer_width': 64,
            'trade_conv_filters': 512,
            'embedded_features_dense_layer_width': 64,
            'activation': 'relu',
            'epochs': 5000,
            'dropout': 0.2,
            'optimizer': 'adadelta',
            'history_filepath': 'history/conv'
        },
        'att': {
            'class': 'SimpleAttention',
            'rfq_query_width': 32,
            'heads': 4,
            'activation': 'relu',
            'epochs': 5000,
            'dropout': 0.5,
            'optimizer': 'adadelta',
            'history_filepath': 'history/att'
        },
        'rfq': {
            'class': 'SimpleRfq',
            'select_rfq_features': True,
            'rfq_features': ['buy_sell', 'side', 'quantity'],
            'multiply_by_std': False,
            'dense_width': 128,
            'residual_layers': 0,
            'activation': 'relu',
            'epochs': 5000,
            'dropout': 0.5,
            'optimizer': 'adadelta',
            'history_filepath': 'history/rfq'
        },
        'lstm': {
            'class': 'SimpleLSTM',
            'rfq_feature_dense_layer_width': 32,
            'trade_conv_filters_1': 32,
            'trade_conv_filters_2': 32,
            'lstm_units': 128,
            'embedded_features_dense_layer_width': 128,
            'activation': 'relu',
            'epochs': 5000,
            'dropout': 0.2,
            'optimizer': 'adadelta',
            'history_filepath': 'history/lstm'
        },
        's_lstm': {
            'class': 'SimplestLSTM',
            'lstm_units': 128,
            'embedded_features_dense_layer_width': 128,
            'activation': 'relu',
            'epochs': 20000,
            'dropout': 0.5,
            'optimizer': 'adadelta',
            'history_filepath': 'history/s_lstm'
        }
    },
    'keras_models': {
        "max_batch_size":17_000,
        "use_log_embedding_for_ordinals": True,
        "min_log_embedding_ordinal_length": 5,
        "batch_escalation_factor": 2.0,
        "learning_rate_decay": 0.75,
        "minimum_learning_rate": 0.000001,
        "early_stopping_patience": 2,
        # For now, this can be 'mirrored', 'multi_worker_mirrored', 'default', and 'one'
        'distribution_strategy': 'multi_worker_mirrored',
        'loss': 'mean_squared_error',
        'use_gpu': True,
        # Whether to load from the most recent model checkpoint.
        'local_model_dir': '../models',
        'load': True,
        'parallel_model': False,
        'num_gpus': 1,
        'normalize': True,
    }
}

__settings = deepcopy(__defalt_settings)
__settings_overridden = False
__path_cache = None

def initialize_cache():
    global __path_cache
    # initialize with the default values for anything accessed during our own import
    __path_cache = {
        '$.logging.level.console.settings': __settings['logging']['level']['console']['settings'],
        '$.logging.level.logfile.settings': __settings['logging']['level']['logfile']['settings']
    }

initialize_cache()


def get(path):
    if path not in __path_cache:
        matches = parse(path).find(__settings)
        if len(matches) == 0:
            raise LookupError(f'Setting not found: "{path}"')
        if len(matches) > 1:
            raise NotImplementedError('Retrieving multiple settings not currently supported')
        __path_cache[path] = matches[0].value
        __logger.debug(f'cache miss, adding: "{path}" = "{__path_cache[path]}"')
    return __path_cache[path]

def checkpoint_settings():
    __logger.debug('Checkpointing the settings.')
    return deepcopy(__settings)

def restore_settings(restored_settings):
    global __settings
    __logger.debug('Restoring the settings.')
    __settings = restored_settings
    initialize_cache()


def override_if_main(name, argv_index):
    if name == '__main__' and len(sys.argv) > argv_index:
        global __settings_overridden
        if __settings_overridden:
            raise RuntimeError('Overriding settings multiple times')
        __settings_overridden = True

        paths_to_overrides = json.loads(sys.argv[argv_index])

        override(paths_to_overrides)

def override(paths_to_overrides):
    __logger.info(f'overriding settings\n{paths_to_overrides}')
    global __settings
    for item in paths_to_overrides.items():
        jsonpath_expr = parse(item[0])
        matches = jsonpath_expr.find(__settings)
        if len(matches) == 0:
            raise LookupError(f'Attempting to override non-existent setting: "{item[0]}"')
        prev = ', '.join([str(m.value) for m in matches])
        jsonpath_expr.update(__settings, item[1])
        updated = ', '.join([str(m.value) for m in jsonpath_expr.find(__settings)])
        __logger.info(f'updated setting\npath:     {item[0]}\noriginal: {prev}\nupdated:  {updated}')
    __path_cache.clear()
    # re-apply any settings we initialized to allow our own import
    for handler in __logger.handlers:
        if isinstance(handler, logging.FileHandler):  # FileHandler inherits from StreamHandler
            handler.setLevel(get('$.logging.level.logfile.settings'))
        else:  # StreamHandler
            handler.setLevel(get('$.logging.level.console.settings'))


# wait until now because of circular import
from log import create_logger
__logger = create_logger('settings')
