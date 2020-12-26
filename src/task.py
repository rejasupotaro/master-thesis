import gc
import json
import os
import sys
from pathlib import Path
from time import time
from typing import Dict, Tuple

import click
import tensorflow as tf
from loguru import logger
from pandas import DataFrame

from src import config
from src.data.cloud_storage import CloudStorage
from src.data.cookpad.preprocessors import ConcatDataProcessor
from src.data.cookpad.recipes import load_raw_recipes
from src.evaluation import evaluate_ranking_model
from src.training import train_ranking_model

project_dir = Path(__file__).resolve().parents[1]


def run_experiment(model_name: str, dataset_id: int, epochs: int, batch_size: int, recipes: Dict) -> Tuple[Dict, float]:
    data_processor = ConcatDataProcessor(recipes)
    train_config, eval_config = {
        'ebr': config.ebr_config,
        'naive': config.naive_config,
        'nrmf_simple_query': config.nrmf_simple_query_config,
        'nrmf_simple_all': config.nrmf_simple_all_config,
        'nrmf_simple_query_with_1st': config.nrmf_simple_query_with_1st_config,
        'fwfm_query': config.fwfm_query_config,
        'fwfm_all': config.fwfm_all_config,
        'fwfm_selected': config.fwfm_selected_config,
        'fwfm_all_without_1st': config.fwfm_all_without_1st_config,
    }[model_name](dataset_id, epochs, data_processor)

    logger.info('Train model')
    history = train_ranking_model(train_config, batch_size)

    logger.info('Evaluate model')
    ndcg_score = evaluate_ranking_model(eval_config)
    return history, ndcg_score


@click.command()
@click.option('--job-dir', type=str)
@click.option('--bucket-name', type=str)
@click.option('--env', type=str)
@click.option('--dataset-id', type=str)
@click.option('--model-name', type=str)
@click.option('--epochs', type=int)
@click.option('--batch-size', type=int)
def main(job_dir: str, bucket_name: str, env: str, dataset_id: str, model_name: str, epochs: int, batch_size: int):
    logger.add(sys.stdout, format='{time} {level} {message}')
    log_filepath = f'{project_dir}/logs/{int(time())}.log'
    logger.add(log_filepath)

    if env == 'cloud':
        tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
        # ClusterSpec({'chief': ['cmle-training-master-e118e0f997-0:2222'], 'ps': [...], 'worker': [...]})
        cluster_info = tf_config.get('cluster', None)
        cluster_spec = tf.train.ClusterSpec(cluster_info)
        # {'type': 'worker', 'index': 3, 'cloud': 'w93a1503672d4dd09-ml'}
        task_info = tf_config.get('task', None)
        job_name, task_index = task_info['type'], task_info['index']
        logger.info(f'cluster_spec {cluster_spec}, job_name: {job_name}, task_index: {task_index}')
        logger.info(f'Task is lauched with arguments job-dir: {job_dir}, bucket-name: {bucket_name}')
    else:
        job_name = 'chief'

    if '-' in dataset_id:
        start, stop = [int(n) for n in dataset_id.split('-')]
        dataset_ids = range(start, stop + 1)
    else:
        dataset_ids = [int(dataset_id)]

    if env == 'cloud':
        logger.info('Download data')
        bucket = CloudStorage(bucket_name)
        Path(f'{project_dir}/data/raw').mkdir(parents=True, exist_ok=True)
        Path(f'{project_dir}/data/processed').mkdir(parents=True, exist_ok=True)
        Path(f'{project_dir}/models').mkdir(exist_ok=True)
        filepaths = []
        for dataset_id in dataset_ids:
            filepaths.append(f'data/processed/listwise.{dataset_id}.train.pkl')
            filepaths.append(f'data/processed/listwise.{dataset_id}.val.pkl')
        filepaths.append('data/raw/recipes.json')
        for filepath in filepaths:
            source = filepath
            destination = f'{project_dir}/{source}'
            logger.info(f'Download {source} to {destination}')
            bucket.download(source, destination)

    logger.info(f'Load recipes')
    recipes = load_raw_recipes()

    results = []
    for dataset_id in dataset_ids:
        logger.info(f'Run an experiment on {model_name} with dataset: {dataset_id}')
        history, ndcg_score = run_experiment(model_name, dataset_id, epochs, batch_size, recipes)
        results.append({
            'dataset_id': dataset_id,
            'model': model_name,
            'val_loss': history['val_loss'][-1],
            'ndcg': ndcg_score,
        })
        gc.collect()
    results_df = DataFrame(results)
    logger.info(results_df)
    results_df.to_csv(f'{project_dir}/logs/{model_name}_results.csv', index=False)

    if env == 'cloud' and job_name == 'chief':
        for filepath in [
            f'logs/{model_name}_results.csv'
        ]:
            source = f'{project_dir}/{filepath}'
            destination = filepath
            logger.info(f'Upload {source} to {destination}')
            bucket.upload(source, destination)

    logger.info('Done')


if __name__ == '__main__':
    main()
