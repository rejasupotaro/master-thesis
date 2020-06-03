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


def train_model(model, train_dataset, test_dataset, epochs):
    get_logger().info('Train model')
    log_dir = os.path.join(project_dir, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback]
    )

    get_logger().info('Save model')
    model.save(os.path.join(project_dir, 'models', 'model.h5'))

    get_logger().info('Done')


def train(build_model_fn):
    get_logger().info('Convert triples into dataset')
    data_processor = triples_to_dataset_multiple
    train_dataset, tokenizer, country_encoder = data_processor.process('triples_100_100.train.pkl')
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'wb') as file:
        pickle.dump(tokenizer, file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'wb') as file:
        pickle.dump(country_encoder, file)
    total_words = len(tokenizer.word_index) + 1
    total_countries = len(country_encoder.classes_)

    test_dataset, _, _ = data_processor.process('triples_100_100.test.pkl', tokenizer, country_encoder)

    get_logger().info('Build model')
    model = build_model_fn(total_words, total_countries)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'label': pairwise_losses.cross_entropy_loss},
        metrics=['accuracy']
    )

    train_model(model, train_dataset, test_dataset, epochs=10)


if __name__ == '__main__':
    create_logger()
    set_seed()
    train(nrmf.build_model)
