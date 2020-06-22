import datetime
import gc
import os
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.data import data_processors
from src.losses import pairwise_losses
from src.models import naive, nrmf, nrmf_concat
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[1]


def train(config):
    get_logger().info('Transform examples into dataset')
    data_processor = config['data_processor']
    train_df = data_processor.listwise_to_df(f'{config["dataset"]}.train.pkl')
    train_dataset = data_processor.fit_transform(train_df)
    with open(os.path.join(project_dir, 'models', f'{config["data_processor_filename"]}.pkl'), 'wb') as file:
        pickle.dump(data_processor, file)
    del train_df
    gc.collect()

    test_df = data_processor.listwise_to_df(f'{config["dataset"]}.test.pkl')
    test_dataset = data_processor.transform(test_df)
    del test_df
    gc.collect()

    get_logger().info('Build model')
    model = config['model'](data_processor).build()
    model.summary()

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
    verbose = config['verbose'] if 'verbose' in config else 1
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=verbose,
    )

    if config['model_filename']:
        get_logger().info('Save model')
        model.save(os.path.join(project_dir, 'models', config['model_filename']))

    get_logger().info('Done')


def train_naive():
    # loss: 0.5273 - accuracy: 0.7328 - val_loss: 0.5410 - val_accuracy: 0.7163
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size='small'),
        'data_processor_filename': 'concat_data_processor.small',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 3,
    }
    train(config)


def train_nrmf():
    # loss: 0.5326 - accuracy: 0.7274 - val_loss: 0.5495 - val_accuracy: 0.7073
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.MultiInstanceDataProcessor(dataset_size='small'),
        'data_processor_filename': 'multi_instance_data_processor.small',
        'model': nrmf.NRMF,
        'model_filename': 'nrmf.h5',
        'epochs': 3,
    }
    train(config)


def train_nrmf_concat():
    # loss: 0.5213 - accuracy: 0.7378 - val_loss: 0.5327 - val_accuracy: 0.7268
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size='small'),
        'data_processor_filename': 'concat_data_processor.small',
        'model': nrmf_concat.NRMFConcat,
        'model_filename': 'nrmf_concat.h5',
        'epochs': 3,
    }
    train(config)


if __name__ == '__main__':
    create_logger()
    set_seed()
    train_naive()
    # train_nrmf()
    # train_nrmf_concat()
