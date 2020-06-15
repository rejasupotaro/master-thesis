import datetime
import os
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.models import naive, nrmf
from src.data import data_processors
from src.losses import pairwise_losses
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def train(config):
    get_logger().info('Transform examples into dataset')
    data_processor = config['data_processor']
    train_dataset, tokenizer, country_encoder = data_processor.listwise_to_dataset(f'{config["dataset"]}.train.pkl')
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'wb') as file:
        pickle.dump(tokenizer, file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'wb') as file:
        pickle.dump(country_encoder, file)
    total_words = len(tokenizer.word_index) + 1
    total_countries = len(country_encoder.classes_)

    test_dataset, _, _ = data_processor.listwise_to_dataset(f'{config["dataset"]}.test.pkl', tokenizer, country_encoder)

    get_logger().info('Build model')
    model = config['build_model_fn'](total_words, total_countries)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'label': pairwise_losses.cross_entropy_loss},
        metrics=['accuracy']
    )

    get_logger().info('Train model')
    log_dir = os.path.join(project_dir, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=test_dataset,
        callbacks=[tensorboard_callback]
    )

    get_logger().info('Save model')
    model.save(os.path.join(project_dir, 'models', config['model_filename']))

    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    set_seed()
    # loss: 0.6142 - accuracy: 0.6297 - val_loss: 0.6266 - val_accuracy: 0.6161
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(),
        'build_model_fn': naive.build_model,
        'model_filename': 'naive.h5',
        'epochs': 3,
    }
    # loss: 0.6151 - accuracy: 0.6304 - val_loss: 0.6261 - val_accuracy: 0.6153
    # config = {
    #     'dataset': 'listwise.small',
    #     'data_processor': data_processors.MultiInstanceDataProcessor(),
    #     'build_model_fn': nrmf.build_model,
    #     'model_filename': 'nrmf.h5',
    #     'epochs': 3,
    # }
    train(config)
