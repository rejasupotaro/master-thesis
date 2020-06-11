import datetime
import os
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.data import triples_to_dataset_concat, triples_to_dataset_multiple
from src.models import simple_model, nrmf
from src.losses import pairwise_losses
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def train(config):
    get_logger().info('Convert triples into dataset')
    data_processor = config['data_processor']
    train_dataset, tokenizer, country_encoder = data_processor.process_triples('triples_100_100.train.pkl')
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'wb') as file:
        pickle.dump(tokenizer, file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'wb') as file:
        pickle.dump(country_encoder, file)
    total_words = len(tokenizer.word_index) + 1
    total_countries = len(country_encoder.classes_)

    test_dataset, _, _ = data_processor.process_triples('triples_100_100.test.pkl', tokenizer, country_encoder)

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
    # config = {
    #     'data_processor': triples_to_dataset_concat,
    #     'build_model_fn': simple_model.build_model,
    #     'model_filename': 'simple_model.h5',
    #     'epochs': 3,
    # }
    config = {
        'data_processor': triples_to_dataset_multiple,
        'build_model_fn': nrmf.build_model,
        'model_filename': 'nrmf.h5',
        'epochs': 3,
    }
    train(config)
