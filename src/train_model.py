import datetime
import os
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.data import data_processors
from src.losses import pairwise_losses
from src.models import naive, nrmf
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

    test_df = data_processor.listwise_to_df(f'{config["dataset"]}.test.pkl')
    test_dataset = data_processor.transform(test_df)

    get_logger().info('Build model')
    model = config['model'](
        data_processor.total_words,
        data_processor.total_countries
    ).build()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'label': pairwise_losses.cross_entropy_loss},
        metrics=['accuracy']
    )

    get_logger().info('Train model')
    log_dir = os.path.join(project_dir, 'logs', 'fit', f'{model.name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=test_dataset,
        callbacks=[tensorboard_callback]
    )

    if config['model_filename']:
        get_logger().info('Save model')
        model.save(os.path.join(project_dir, 'models', config['model_filename']))

    get_logger().info('Done')


def train_naive():
    # loss: 0.5260 - accuracy: 0.7331 - val_loss: 0.5359 - val_accuracy: 0.7222
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(),
        'data_processor_filename': 'concat_data_processor',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 3,
    }
    train(config)


def train_nrmf():
    # loss: 0.5406 - accuracy: 0.7206 - val_loss: 0.5554 - val_accuracy: 0.7024
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.MultiInstanceDataProcessor(),
        'data_processor_filename': 'multi_instance_data_processor',
        'model': nrmf.NRMF,
        'model_filename': 'nrmf.h5',
        'epochs': 3,
    }
    train(config)


if __name__ == '__main__':
    create_logger()
    set_seed()
    train_naive()
    # train_nrmf()
#