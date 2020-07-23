import pickle
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from src.config import EvalConfig
from src.losses import pairwise_losses
from src.metrics import metrics
from src.utils.logger import get_logger
from tensorflow import keras
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[1]


def evaluate(config: EvalConfig):
    get_logger().info('Load model')
    filepath = f'{project_dir}/models/{config.model_name}.h5'
    custom_objects = {
        'cross_entropy_loss': pairwise_losses.cross_entropy_loss
    }
    model = keras.models.load_model(filepath, custom_objects=custom_objects)

    get_logger().info('Load val dataset')
    with open(f'{project_dir}/models/{config.data_processor_filename}.pkl', 'rb') as file:
        data_processor = pickle.load(file)
    with open(f'{project_dir}/data/processed/{config.dataset}.val.pkl', 'rb') as file:
        val_dataset = pickle.load(file)

    get_logger().info('Predict')
    map_scores = []
    ndcg_scores = []
    for example in (tqdm(val_dataset) if config.verbose > 0 else val_dataset):
        rows = []
        for doc in example['docs']:
            row = {
                'query': example['query'],
                'doc_id': doc['doc_id'],
                'label': doc['label']
            }
            rows.append(row)
        df = DataFrame(rows)
        x, y = data_processor.process_batch(df)
        dataset = tf.data.Dataset.from_tensor_slices((x, {'label': y})).batch(128)
        preds = model.predict(dataset, verbose=0)
        df['pred'] = preds
        y_true = df['label'].tolist()
        y_pred = df['pred'].tolist()
        map_scores.append(metrics.mean_average_precision(y_true, y_pred))
        ndcg_scores.append(metrics.normalized_discount_cumulative_gain(y_true, y_pred))

    map_score = round(np.mean(map_scores), 4)
    ndcg_score = round(np.mean(ndcg_scores), 4)
    get_logger().info(f'MAP: {map_score}, NDCG: {ndcg_score}')
    mlflow.log_metric('MAP', map_score)
    mlflow.log_metric('NDCG', ndcg_score)
