import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from src.losses import pairwise_losses
from src.data.data_generator import DataGenerator
from src.metrics import metrics
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[1]


def evaluate(config):
    get_logger().info('Load model')
    filepath = os.path.join(project_dir, 'models', config['model_filename'])
    custom_objects = {
        'cross_entropy_loss': pairwise_losses.cross_entropy_loss
    }
    model = keras.models.load_model(filepath, custom_objects=custom_objects)

    get_logger().info('Load test dataset')
    with open(os.path.join(project_dir, 'models', f'{config["data_processor_filename"]}.pkl'), 'rb') as file:
        data_processor = pickle.load(file)
    with open(os.path.join(project_dir, 'data', 'processed', f'{config["dataset"]}.test.pkl'), 'rb') as file:
        test_dataset = pickle.load(file)

    get_logger().info('Predict')
    map_scores = []
    ndcg_scores = []
    verbose = config['verbose'] if 'verbose' in config else 1
    for example in (tqdm(test_dataset) if verbose > 0 else test_dataset):
        rows = []
        for doc in example['docs']:
            row = {
                'query': example['query'],
                'doc_id': doc['doc_id'],
                'label': doc['label']
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        x, y = data_processor.process_batch(df)
        dataset = tf.data.Dataset.from_tensor_slices((x, {'label': y})).batch(128)
        preds = model.predict(dataset, verbose=0)
        df['pred'] = preds
        y_true = df['label'].tolist()
        y_pred = df['pred'].tolist()
        map_scores.append(metrics.mean_average_precision(y_true, y_pred))
        ndcg_scores.append(metrics.normalized_discount_cumulative_gain(y_true, y_pred))
    get_logger().info(f'MAP: {np.mean(map_scores)}, NDCG: {np.mean(ndcg_scores)}')


def naive_config():
    # MAP: 0.6037210380315634, NDCG: 0.6978183206053716
    return {
        'dataset': 'listwise.small',
        'data_processor_filename': 'concat_data_processor.small',
        'model_filename': 'naive.h5',
    }


def nrmf_config():
    # MAP: 0.5582181750294172, NDCG: 0.6601386484807065
    return {
        'dataset': 'listwise.small',
        'data_processor_filename': 'multi_instance_data_processor.small',
        'model_filename': 'nrmf.h5',
    }


def nrmf_concat_config():
    # MAP: 0.6245518197329846, NDCG: 0.7136872418647048
    return {
        'dataset': 'listwise.small',
        'data_processor_filename': 'concat_data_processor.small',
        'model_filename': 'nrmf_concat.h5',
    }


if __name__ == '__main__':
    create_logger()
    set_seed()
    config = naive_config()
    # config = nrmf_config()
    # config = nrmf_concat_config()
    evaluate(config)
