import os
import pickle
from pathlib import Path

from tensorflow import keras

from src.data import triples_to_dataset
from src.models import simple_model
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def train(model, train_dataset, test_dataset, epochs):
    get_logger().info('Train model')
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset
    )

    get_logger().info('Save model')
    model.save(os.path.join(project_dir, 'models', 'model.h5'))

    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    set_seed()

    get_logger().info('Convert triples into dataset')
    train_dataset, tokenizer, country_encoder = triples_to_dataset.process('triples_100_100.train.pkl')
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'wb') as file:
        pickle.dump(tokenizer, file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'wb') as file:
        pickle.dump(country_encoder, file)
    total_words = len(tokenizer.word_index) + 1
    total_countries = len(country_encoder.classes_)

    test_dataset, _, _ = triples_to_dataset.process('triples_100_100.test.pkl', tokenizer, country_encoder)

    get_logger().info('Build model')
    model = simple_model.build_model(total_words, total_countries)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'relevance': keras.losses.BinaryCrossentropy(from_logits=True)},
        metrics=['accuracy']
    )

    train(model, train_dataset, test_dataset, epochs=10)
