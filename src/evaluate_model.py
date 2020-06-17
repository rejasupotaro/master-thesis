import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from src.losses import pairwise_losses
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
    for example in tqdm(test_dataset):
        rows = []
        for doc in example['docs']:
            row = {
                'query': example['query'],
                'doc_id': doc['doc_id'],
                'label': doc['label']
            }
            rows.append(row)
        test_df = pd.DataFrame(rows)
        test_dataset = data_processor.transform(test_df)
        preds = model.predict(test_dataset)
        test_df['pred'] = preds
        y_true = test_df['label'].tolist()
        y_pred = test_df['pred'].tolist()
        map_scores.append(metrics.mean_average_precision(y_true, y_pred))
        ndcg_scores.append(metrics.normalized_discount_cumulative_gain(y_true, y_pred))
    get_logger().info(f'MAP: {np.mean(map_scores)}, NDCG: {np.mean(ndcg_scores)}')


def evaluate_naive():
    # MAP: 0.45707103204748883, NDCG: 0.5631012458613334
    config = {
        'dataset': 'listwise.small',
        'data_processor_filename': 'concat_data_processor',
        'model_filename': 'naive.h5',
    }
    evaluate(config)


def evaluate_nrmf():
    # MAP: 0.4616903083060701, NDCG: 0.5648055579634502
    config = {
        'dataset': 'listwise.small',
        'data_processor_filename': 'multi_instance_data_processor',
        'model_filename': 'nrmf.h5',
    }
    evaluate(config)


if __name__ == '__main__':
    create_logger()
    set_seed()
    evaluate_naive()
    # evaluate_nrmf()
