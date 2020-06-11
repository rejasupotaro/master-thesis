import os
import pickle
from pathlib import Path

from tensorflow import keras

from src.data import triples_to_dataset_multiple
from src.losses import pairwise_losses
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def predict():
    get_logger().info('Load test dataset')
    data_processor = triples_to_dataset_multiple
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'rb') as file:
        tokenizer = pickle.load(file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'rb') as file:
        country_encoder = pickle.load(file)

    test_dataset, _, _ = data_processor.process('triples_100_100.test.pkl', tokenizer, country_encoder)

    get_logger().info('Load model')
    filepath = os.path.join(project_dir, 'models', 'model.h5')
    custom_objects = {
        'cross_entropy_loss': pairwise_losses.cross_entropy_loss
    }
    model = keras.models.load_model(filepath, custom_objects=custom_objects)
    model.summary()
    preds = model.predict(test_dataset)


if __name__ == '__main__':
    create_logger()
    set_seed()
    predict()
