import math
from abc import ABC, abstractmethod
import os
import shutil
from uuid import uuid4
from zipfile import ZipFile

import git
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Embedding, Reshape, Concatenate

from generator import TraceDataGenerator
from load_data import get_ordinals
from model import TraceModel
from normalized_trace_data_generator import NormalizedTraceDataGenerator
from s3 import upload_file, get_object, get_all_paths_with_prefix
import settings


if settings.get('$.keras_models.use_gpu'):
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
    #    for gpu in gpus:
    #        tf.config.experimental.set_memory_growth(gpu, True)
    pass
else:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Slice(tf.keras.layers.Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)

class SaveModelWrapper(tensorflow.keras.Model):
    def __init__(self, trace_keras_class, inputs, outputs, name):
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        self.keras_model = trace_keras_class

    @staticmethod
    def __save(self, filepath, save_method, **kwargs):
        if not filepath.startswith('s3://'):
            save_method(filepath)
            return

        # We need to save the files to a temporary local directory and then zip them up and upload them to s3

        # Create a temporary directory
        uuid = str(uuid4())
        temp_dir = os.path.join(settings.get('$.keras_models.local_model_dir'), uuid)
        os.makedirs(temp_dir, exist_ok=False)
        model_path = os.path.join(temp_dir, 'model')
        save_method(model_path, **kwargs)

        # Zip up the files in the temporary directory
        zip_path = os.path.join(temp_dir, 'model.zip')

        # Zip up all of the files that are not a zip file in the temporary directory
        with ZipFile(zip_path, 'w') as zip:
            for file in os.listdir(temp_dir):
                if not file.endswith('.zip'):
                    zip.write(os.path.join(temp_dir, file), file)

        # Upload the zip file to s3
        upload_file(zip_path, filepath)
        # Delete the temporary directory
        shutil.rmtree(temp_dir)

    def save(self, filepath, **kwargs):
        self.__save(self, filepath, super().save, **kwargs)

    def save_weights(self, filepath, **kwargs):
        self.__save(self, filepath, super().save_weights, **kwargs)

    def __get_load_filepath(self, filepath):
        # If the filepath is s3, then load it from s3.
        if filepath.startswith('s3://'):
            filepaths = get_object(filepath)
            # Filepaths are now a list of local paths to the files.
            # We need to find the longest common prefix of the filepaths and set
            # the filepath to that.
            filepath = os.path.join(os.path.commonpath(filepaths), 'model')
        return filepath

    def load(self, filepath, **kwargs):
        filepath = self.__get_load_filepath(filepath)
        if filepath is None:
            return
        super().load(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        filepath = self.__get_load_filepath(filepath)
        if filepath is None:
            return
        super().load_weights(filepath, **kwargs)


class KerasTraceModel(TraceModel, ABC):
    '''
    Simple baseline neural network model which directly processes all of the features.
    '''

    def __init__(self,
                 model_name: str,
                 model_settings: dict,
                 train_generator: TraceDataGenerator,
                 validation_generator: TraceDataGenerator,
                 test_generator: TraceDataGenerator):
        if settings.get('$.keras_models.normalize'):
            super().__init__(model_name,
                             model_settings,
                             NormalizedTraceDataGenerator(train_generator),
                             NormalizedTraceDataGenerator(validation_generator),
                             NormalizedTraceDataGenerator(test_generator))
        else:
            super().__init__(model_name,
                             model_settings,
                             train_generator,
                             validation_generator,
                             test_generator)
        self.trace_train_generator = train_generator
        self.model = None
        self.model_input = None
        self.output = None
        self.history = None
        self.distribution_strategy = None
        strategy_string = settings.get('$.keras_models.distribution_strategy')
        if strategy_string == 'mirrored':
            self.distribution_strategy = tf.distribute.MirroredStrategy()
        elif strategy_string == 'multi_worker_mirrored':
            self.distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        elif strategy_string == 'one':
            self.distribution_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif strategy_string == 'default':
            self.distribution_strategy == tf.distribute.get_strategy()
        else:
            raise ValueError(f'Unsupported distribution strategy in settings: {strategy_string}')


    @staticmethod
    def get_model_bucket() -> str:
        return settings.get(f'$.environment_settings.{settings.get("$.environment")}.model_bucket')

    @staticmethod
    def get_local_model_dir():
        return settings.get('$.keras_models.local_model_dir')

    @staticmethod
    def get_current_commit_hash():
        if commit_history := os.getenv('COMMIT_HISTORY'):
            return commit_history.split()[0]
        return git.Repo(search_parent_directories=True).head.object.hexsha

    def get_relative_model_path(self) -> str:
        return self.model_name

    def get_relative_model_path_with_hash(self) -> str:
        return os.path.join(self.get_relative_model_path(), KerasTraceModel.get_current_commit_hash())

    def get_s3_model_load_path(self) -> str:
        """
        Returns an s3 path for the model artifact zip file
        """
        latest_hash = self.get_latest_commit_hash()
        if latest_hash is None:
            return None
        return f's3://{KerasTraceModel.get_model_bucket()}/{self.get_relative_model_path()}/{latest_hash}.zip'

    def get_s3_model_save_path(self) -> str:
        """
        Returns an s3 path for the model artifact zip file
        """
        return f's3://{KerasTraceModel.get_model_bucket()}/{self.get_relative_model_path_with_hash()}.zip'

    def get_s3_model_path_for_hash(self, commit_hash: str) -> str:
        if commit_hash is None:
            return None
        return f's3://{KerasTraceModel.get_model_bucket()}/{os.path.join(self.get_relative_model_path(), commit_hash)}'

    def get_latest_commit_hash(self):
        prefix = f's3://{KerasTraceModel.get_model_bucket()}/{self.get_relative_model_path()}/'
        paths = get_all_paths_with_prefix(prefix)
        if len(paths) == 0:
            return None

        # Create a set of hashes from the file names after removing the .zip extension
        hashes = set([path.split('/')[-1].split('.')[0] for path in paths])

        # Find the latest commit hash in the model hash set.
        if commit_history := os.getenv('COMMIT_HISTORY'):
            for commit_hash in commit_history.split():
                if commit_hash in hashes:
                    return commit_hash
        else:
            repo = git.Repo('.', search_parent_directories=True)
            for commit in repo.iter_commits():
                if commit.hexsha in hashes:
                    return commit

        # If we didn't find a commit hash, return None
        return None

    def embed_features(self):
        print('Starting to embed features.')
        model_input = features = tf.keras.Input(shape=(self.trace_train_generator.get_features_total_size(),))

        ordinals = get_ordinals()

        rfq_features = Slice([0, 0],
                             [
                                 #self.trace_train_generator.get_batch_size(),
                                 -1,
                              self.trace_train_generator.get_rfq_features_count()], name='Slice_RFQ_Features')(features)
        rfq_feature_tensors = []
        rfq_feature_names = self.trace_train_generator.get_rfq_features()
        rfq_feature_indices = []
        rfq_feature_current_index = 0
        for i in range(self.trace_train_generator.get_rfq_features_count()):
            rfq_feature_name = rfq_feature_names[i]
            rfq_feature_tensor = Slice([0, i], [-1, 1],
                                       name=f'Slice_{rfq_feature_name}_RFQ_Feature')(rfq_features)
            if rfq_feature_name in ordinals:
                ordinal = ordinals[rfq_feature_name]
                embedding_dim = self.__embedding_dim(len(ordinal))
                rfq_feature_tensor = Embedding(len(ordinal) + 1, embedding_dim,
                                               input_length=1)(rfq_feature_tensor)
                rfq_feature_tensor = Reshape((embedding_dim,))(rfq_feature_tensor)
                rfq_feature_indices.append((rfq_feature_current_index, rfq_feature_current_index + embedding_dim))
                rfq_feature_current_index += embedding_dim
            else:
                rfq_feature_indices.append((rfq_feature_current_index, rfq_feature_current_index + 1))
                rfq_feature_current_index += 1
            rfq_feature_tensors.append(rfq_feature_tensor)
        rfq_features = Concatenate(axis=-1)(rfq_feature_tensors)
        rfq_feature_names_to_indices = dict(zip(rfq_feature_names, rfq_feature_indices))

        trades_count = self.trace_train_generator.get_group_count() * self.trace_train_generator.get_sequence_length()
        trade_features = Slice([0, self.trace_train_generator.get_rfq_features_count()],
                               [-1,
                                self.trace_train_generator.get_trade_features_total_size()],
                               name='Slice_Trade_Features')(features)
        trade_features = Reshape((trades_count,
                                  self.trace_train_generator.get_trade_features_count()))(trade_features)
        trade_feature_names = self.trace_train_generator.get_trade_features()
        trade_feature_tensors = []
        new_trade_feature_length = 0
        trade_feature_indices = []
        trade_feature_current_index = 0
        for i in range(self.trace_train_generator.get_trade_features_count()):
            trade_feature_name = trade_feature_names[i]
            trade_feature_tensor = Slice([0, 0, i], [-1, trades_count, 1],
                                         name=f'Slice_{trade_feature_name}_trade_feature')(trade_features)
            added_feature_length = 1
            if trade_feature_name in ordinals:
                ordinal = ordinals[trade_feature_name]
                embedding_dim = self.__embedding_dim(len(ordinal))
                trade_feature_tensor = Embedding(len(ordinal) + 1, embedding_dim,
                                                 input_length=trades_count)(trade_feature_tensor)
                trade_feature_tensor = Reshape((trades_count, embedding_dim))(trade_feature_tensor)
                added_feature_length = embedding_dim
                trade_feature_indices.append((trade_feature_current_index, trade_feature_current_index + embedding_dim))
                trade_feature_current_index += embedding_dim
            else:
                trade_feature_indices.append((trade_feature_current_index, trade_feature_current_index + 1))
                trade_feature_current_index += 1
            new_trade_feature_length += added_feature_length
            trade_feature_tensors.append(trade_feature_tensor)
        trade_features = Concatenate(axis=-1)(trade_feature_tensors)
        trade_features = Reshape((self.trace_train_generator.get_group_count(),
                                  self.trace_train_generator.get_sequence_length(),
                                  new_trade_feature_length))(trade_features)
        trade_features_names_to_indices = dict(zip(trade_feature_names, trade_feature_indices))

        print('Done embedding features. Returning.')

        return model_input, rfq_features, rfq_feature_names_to_indices, trade_features, trade_features_names_to_indices

    def __embedding_dim(self, ordinal_len):
        if settings.get('$.keras_models.use_log_embedding_for_ordinals') and \
                ordinal_len > settings.get("$.keras_models.min_log_embedding_ordinal_length"):
            embedding_dim = int(math.log(ordinal_len) / math.log(2)) + 1
        else:
            embedding_dim = ordinal_len + 1
        return embedding_dim

    @abstractmethod
    def define_model(self):
        pass

    def save(self):
        self.model.save(self.get_model_setting('save_full_model_filepath'))

    def load(self):
        if not settings.get('$.keras_models.load'):
            return

        path = self.get_s3_model_load_path()

        if path is None:
            return

        self.model.load_weights(path)

    def train(self):
        self.create()
        self.fit_model()

    def create(self):
        if self.distribution_strategy is not None:
            with self.distribution_strategy.scope():
                self.create_model()
        else:
            self.create_model()

    def create_model(self):
        self.define_model()
        self.compile_model()
        self.load()

    def compile_model_cpu(self):
        if settings.get('$.keras_models.parallel_model'):
            with tf.device('/cpu:0'):
                self.compile_model()
        else:
            self.compile_model()

    def compile_model(self):
        self.model = SaveModelWrapper(self.__class__, inputs=self.model_input,
                                                      outputs=self.output, name='TraceNNModel')
        self.model.summary()
        self.model.compile(loss=settings.get('$.keras_models.loss'),
                           optimizer=self.get_model_setting('optimizer'), metrics=['mean_absolute_error'])

    def fit(self):
        batch_size = settings.get('$.batch_size')
        max_batch_size = settings.get('$.keras_models.max_batch_size')
        batch_escalation_factor = settings.get('$.keras_models.batch_escalation_factor')
        # Set the validation_generator to a fixed batch size which will be efficient to compute.
        self.validation_generator.set_batch_size(16_384)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_s3_model_save_path(),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings.get('$.keras_models.early_stopping_patience'),
            restore_best_weights=True,
        )
        history_filepath = self.get_model_setting('history_filepath') + '.'
        fit_train = self.train_generator
        fit_val = self.validation_generator

        while batch_size <= max_batch_size:
            print(f'Fitting with batch size {batch_size}')

            os.makedirs(os.path.dirname(history_filepath), exist_ok=True)
            csv_logger = CSVLogger(history_filepath + str(batch_size), append=True)
            fit_train.set_batch_size(batch_size)
            self.history = self.model.fit(fit_train, validation_data=fit_val,
                                          epochs=self.get_model_setting('epochs'),
                                          steps_per_epoch=self.train_generator.__len__(),
                                          callbacks=[early_stopping_callback, model_checkpoint_callback, csv_logger])
            batch_size = int(batch_size * batch_escalation_factor)


        def custom_lr_scheduler(epoch, current_lr):
            if current_lr < settings.get("$.keras_models.minimum_learning_rate"):
                return current_lr
            return current_lr * settings.get("$.keras_models.learning_rate_decay")

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(custom_lr_scheduler)

        csv_logger = CSVLogger(history_filepath + "lr", append=True)
        fit_train.set_batch_size(max_batch_size)
        self.history = self.model.fit(fit_train, validation_data=fit_val,
                                      epochs=self.get_model_setting('epochs'),
                                      steps_per_epoch=self.train_generator.__len__(),
                                      callbacks=[lr_schedule, early_stopping_callback, model_checkpoint_callback, csv_logger])

        fit_train.set_batch_size(settings.get('$.batch_size'))
        fit_val.set_batch_size(settings.get('$.batch_size'))



    def evaluate_batch(self, X_b):
        na_filter, X_b, previous_labels = self.train_generator.normalize_x(X_b)
        return NormalizedTraceDataGenerator.unnormalize_y(
            previous_labels, self.model.predict(X_b))
