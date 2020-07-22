import datetime
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import mlflow
import tensorflow as tf
from tensorflow import keras

from src.config import TrainConfig
from src.data import preprocessors
from src.data.data_generator import DataGenerator
from src.losses import pairwise_losses
from src.models import naive, nrmf
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[1]


# https://www.tensorflow.org/api_docs/python/tf/keras/backend/set_floatx
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


def train(config: TrainConfig):
    mlflow.log_params(asdict(config))

    get_logger().info('Transform examples into dataset')
    data_processor = config.data_processor

    train_df = data_processor.listwise_to_df(f'{config.dataset}.train.pkl')
    val_df = data_processor.listwise_to_df(f'{config.dataset}.val.pkl')
    data_processor.fit(train_df)
    with open(os.path.join(project_dir, 'models', f'{config.data_processor_filename}.pkl'), 'wb') as file:
        pickle.dump(data_processor, file)
    train_generator = DataGenerator(train_df, data_processor)
    val_generator = DataGenerator(val_df, data_processor)

    get_logger().info('Build model')
    model = config.model(data_processor).build()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'label': pairwise_losses.cross_entropy_loss},
        metrics=['accuracy']
    )

    get_logger().info('Train model')
    log_dir = os.path.join(project_dir, 'logs', 'fit',
                           f'{model.name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    ]
    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=config.verbose,
    )

    get_logger().info(history.history)
    for metric in history.history:
        for i, value in enumerate(history.history[metric]):
            mlflow.log_metric(metric, value, step=i)

    get_logger().info('Save model')
    model.save(os.path.join(project_dir, 'models', model.name))

    get_logger().info('Done')


def naive_config() -> TrainConfig:
    return TrainConfig(
        dataset='listwise.small',
        data_processor=preprocessors.ConcatDataProcessor(dataset_size='small'),
        data_processor_filename='concat_data_processor.small',
        model=naive.Naive,
        model_filename='naive.h5',
        epochs=1,
    )


def nrmf_config() -> TrainConfig:
    return TrainConfig(
        dataset='listwise.small',
        data_processor=preprocessors.MultiInstanceDataProcessor(dataset_size='small'),
        data_processor_filename='multi_instance_data_processor.small',
        model=nrmf.NRMF,
        model_filename='nrmf.h5',
        epochs=1,
    )


def nrmf_concat_config() -> TrainConfig:
    return TrainConfig(
        dataset='listwise.small',
        data_processor=preprocessors.ConcatDataProcessor(dataset_size='small'),
        data_processor_filename='concat_data_processor.small',
        model=nrmf.NRMFSimple,
        model_filename='nrmf_concat.h5',
        epochs=3,
    )


if __name__ == '__main__':
    create_logger()
    set_seed()
    mlflow.set_tracking_uri(os.path.join(project_dir, 'logs', 'mlruns'))
    mlflow.start_run()

    config = naive_config()
    # config = nrmf_config()
    # config = nrmf_concat_config()
    train(config)

    mlflow.log_artifact(os.path.join(project_dir, 'logs', '1.log'))
    mlflow.end_run()
