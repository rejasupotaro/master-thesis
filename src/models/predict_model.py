import os
import pandas as pd
import pickle
from pathlib import Path

from tensorflow import keras
from collections import defaultdict

from src.metrics import metrics
from src.data import triples_to_dataset_multiple
from src.losses import pairwise_losses
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def load_test_dataset():
    interactions_df = pd.read_csv(os.path.join(project_dir, 'data', 'raw', 'interactions.csv'), nrows=1000)
    dataset = defaultdict(dict)
    for index, row in interactions_df.head(10).iterrows():
        if row['page'] != 1:
            continue

        example = dataset[row['session_id']]
        example['query'] = row['query']
        positive_doc_id = row['recipe_id']
        if 'docs' not in example:
            example['docs'] = []
        doc_ids = [doc['doc_id'] for doc in example['docs']]
        new_doc_ids = [int(doc_id) for doc_id in row['fetched_recipe_ids'].split(',')]
        new_doc_ids = new_doc_ids[:row['position'] + 1]
        for doc_id in new_doc_ids:
            if doc_id in doc_ids:
                continue
            example['docs'].append({
                'doc_id': doc_id,
                'label': 1 if doc_id == positive_doc_id else 0
            })
        if example['docs'][-1]['label'] == 0:
            example['docs'].pop()

    for example in dataset.values():
        rows = []
        for doc in example['docs']:
            row = {
                'query': example['query'],
                'doc_id': doc['doc_id'],
                'label': doc['label']
            }
            rows.append(row)
        if len(rows) > 5:
            test_df = pd.DataFrame(rows)
        break
    print(test_df)

    with open(os.path.join(project_dir, 'models', 'tokenizer.pkl'), 'rb') as file:
        tokenizer = pickle.load(file)
    with open(os.path.join(project_dir, 'models', 'country_encoder.pkl'), 'rb') as file:
        country_encoder = pickle.load(file)

    data_processor = triples_to_dataset_multiple
    test_dataset, _, _ = data_processor.process(test_df, tokenizer, country_encoder)
    return test_df, test_dataset


def predict():
    get_logger().info('Load test dataset')
    test_df, test_dataset = load_test_dataset()

    get_logger().info('Load model')
    filepath = os.path.join(project_dir, 'models', 'model.h5')
    custom_objects = {
        'cross_entropy_loss': pairwise_losses.cross_entropy_loss
    }
    model = keras.models.load_model(filepath, custom_objects=custom_objects)
    preds = model.predict(test_dataset)
    print(preds)
    test_df['pred'] = preds
    print(test_df[['query', 'label', 'pred']])
    y_true = test_df['label'].tolist()
    y_pred = test_df['pred'].tolist()
    map_score = metrics.mean_average_precision(y_true, y_pred)
    ndcg_score = metrics.normalized_discount_cumulative_gain(y_true, y_pred)
    print(f'MAP: {map_score}, NDCG: {ndcg_score}')


if __name__ == '__main__':
    create_logger()
    set_seed()
    predict()
