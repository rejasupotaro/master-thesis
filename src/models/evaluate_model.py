import os
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from src.data import triples_to_dataset_concat
from src.data import triples_to_dataset_multiple
from src.losses import pairwise_losses
from src.metrics import metrics
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

from src.data import triples_to_dataset_concat
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed


def evaluate(config):
    project_dir = Path(__file__).resolve().parents[2]

    get_logger().info('Load model')
    filepath = os.path.join(project_dir, 'models', config['model_filename'])
    custom_objects = {
        'cross_entropy_loss': pairwise_losses.cross_entropy_loss
    }
    model = keras.models.load_model(filepath, custom_objects=custom_objects)

    get_logger().info('Convert triples into dataset')
    triples_filename = 'triples_100_100.test.pkl'
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'rb') as file:
        tokenizer = pickle.load(file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'rb') as file:
        country_encoder = pickle.load(file)
    dataset, _, _ = config['data_processor'].process_triples(triples_filename, tokenizer, country_encoder)

    get_logger().info('Evaluate model')
    test_loss, test_acc = model.evaluate(dataset)
    print(f'Loss: {test_loss}, Acc: {test_acc}')
    # [Baseline] Loss: 0.5366079807281494, Acc: 0.7225499749183655
    # [NRM-F] 0.5448315739631653, Acc: 0.7185999751091003


if __name__ == '__main__':
    create_logger()
    set_seed()
    # config = {
    #     'data_processor': triples_to_dataset_concat,
    #     'model_filename': 'simple_model.h5',
    # }
    config = {
        'data_processor': triples_to_dataset_multiple,
        'model_filename': 'nrmf.h5',
    }
    evaluate(config)
