import os
import pickle
from pathlib import Path

import tensorflow as tf

from src.data import triples_to_dataset_concat
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def evaluate_model(model, dataset):
    get_logger().info('Evaluate model')
    test_loss, test_acc = model.evaluate(dataset)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    create_logger()
    set_seed()

    get_logger().info('Convert triples into dataset')
    triples_filename = 'triples_100_100.test.pkl'
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'rb') as file:
        tokenizer = pickle.load(file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'rb') as file:
        country_encoder = pickle.load(file)
    dataset, _, _ = triples_to_dataset_concat.process_triples(triples_filename, tokenizer, country_encoder)

    get_logger().info('Load model')
    project_dir = Path(__file__).resolve().parents[2]
    model = tf.keras.models.load_model(os.path.join(project_dir, 'models', 'model.h5'))

    evaluate_model(model, dataset)
