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

project_dir = Path(__file__).resolve().parents[2]


def predict(config):
    get_logger().info('Load model')
    filepath = os.path.join(project_dir, 'models', config['model_filename'])
    custom_objects = {
        'cross_entropy_loss': pairwise_losses.cross_entropy_loss
    }
    model = keras.models.load_model(filepath, custom_objects=custom_objects)

    get_logger().info('Load test dataset')
    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'rb') as file:
        tokenizer = pickle.load(file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'rb') as file:
        country_encoder = pickle.load(file)
    with open(os.path.join(project_dir, 'data', 'processed', f'{config["dataset"]}.test.pkl'), 'rb') as file:
        dataset = pickle.load(file)

    get_logger().info('Predict')
    map_scores = []
    ndcg_scores = []
    for example in tqdm(dataset[:10]):
        rows = []
        for doc in example['docs']:
            row = {
                'query': example['query'],
                'doc_id': doc['doc_id'],
                'label': doc['label']
            }
            rows.append(row)
        test_df = pd.DataFrame(rows)
        test_dataset, _, _ = config['data_processor'].process(test_df, tokenizer, country_encoder)
        preds = model.predict(test_dataset)
        test_df['pred'] = preds
        y_true = test_df['label'].tolist()
        y_pred = test_df['pred'].tolist()
        map_scores.append(metrics.mean_average_precision(y_true, y_pred))
        ndcg_scores.append(metrics.normalized_discount_cumulative_gain(y_true, y_pred))
    print(f'MAP: {np.mean(map_scores)}, NDCG: {np.mean(ndcg_scores)}')
    # [Baseline] MAP: 0.6971153846153846, NDCG: 0.7365986575892964
    # [NRM-F] MAP: 0.5833333333333334, NDCG: 0.6888569943706637


if __name__ == '__main__':
    create_logger()
    set_seed()
    config = {
        'dataset': 'listwise.small',
        'data_processor': triples_to_dataset_concat,
        'model_filename': 'simple_model.h5',
    }
    # config = {
    #     'dataset': 'listwise.small',
    #     'data_processor': triples_to_dataset_multiple,
    #     'model_filename': 'nrmf.h5',
    # }
    predict(config)
