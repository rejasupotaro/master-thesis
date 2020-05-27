import os
from pathlib import Path

import tensorflow as tf

from src.data import triples_to_dataset
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed


def evaluate(model, test_dataset):
    get_logger().info('Evaluate model')
    test_loss, test_acc = model.evaluate(test_dataset)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    create_logger()
    set_seed()

    get_logger().info('Convert triples to dataset')
    triples_filename = 'triples_100_100.pkl'
    train_dataset, test_dataset, tokenizer = triples_to_dataset.process(triples_filename)

    get_logger().info('Load model')
    project_dir = Path(__file__).resolve().parents[2]
    model = tf.keras.models.load_model(os.path.join(project_dir, 'models', 'model.h5'))

    evaluate(model, test_dataset)
