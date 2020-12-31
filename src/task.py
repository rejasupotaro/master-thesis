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

from src.config import config
from src.data.cloud_storage import CloudStorage
from src.evaluation import evaluate_ranking_model
from src.training import train_ranking_model

project_dir = Path(__file__).resolve().parents[1]


def run_experiment(dataset: str, dataset_id: int, model_name: str, epochs: int, batch_size: int, docs: Dict) -> Tuple[
    Dict, float]:
    train_config, eval_config = config.get_config(dataset, dataset_id, model_name, epochs, docs)

    logger.info(f'Train model ({model_name})')
    model, history = train_ranking_model(train_config, batch_size)

    logger.info('Evaluate model')
    ndcg_score = evaluate_ranking_model(eval_config, model)
    return history, ndcg_score


@click.command()
@click.option('--job-dir', type=str)
@click.option('--bucket-name', type=str)
@click.option('--env', type=str)
@click.option('--dataset', type=str)
@click.option('--dataset-id', type=str)
@click.option('--model-name', type=str)
@click.option('--epochs', type=int)
@click.option('--batch-size', type=int)
def main(job_dir: str, bucket_name: str, env: str, dataset: str, dataset_id: str, model_name: str, epochs: int,
         batch_size: int):
    logger.add(sys.stdout, format='{time} {level} {message}')
    log_filepath = f'{project_dir}/logs/{int(time())}.log'
    logger.add(log_filepath)

    if dataset not in ['cookpad', 'msmarco']:
        raise Exception(f'Unknown dataset is specified: {dataset}')

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
            filepaths.append(f'data/processed/listwise.{dataset}.{dataset_id}.train.pkl')
            filepaths.append(f'data/processed/listwise.{dataset}.{dataset_id}.val.pkl')

        if dataset == 'cookpad':
            filepaths.append('data/raw/recipes.json')
            filepaths.append(f'data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl')
        else:
            filepaths.append('data/raw/msmarco-docs.tsv.gz')
        for filepath in filepaths:
            source = filepath
            destination = f'{project_dir}/{source}'
            logger.info(f'Download {source} to {destination}')
            bucket.download(source, destination)

    logger.info(f'Load docs')
    if dataset == 'cookpad':
        from src.data.cookpad.recipes import load_raw_recipes
        docs = load_raw_recipes()
    else:
        from src.data.msmarco.docs import load_raw_docs
        docs = load_raw_docs()

    results = []
    for dataset_id in dataset_ids:
        logger.info(f'Run an experiment on {model_name} with dataset: {dataset}.{dataset_id}')
        history, ndcg_score = run_experiment(dataset, dataset_id, model_name, epochs, batch_size, docs)
        results.append({
            'dataset_id': dataset_id,
            'model': model_name,
            'val_loss': history['val_loss'][-1],
            'ndcg': ndcg_score,
        })
        gc.collect()
    results_df = DataFrame(results)
    logger.info(results_df)
    results_df.to_csv(f'{project_dir}/logs/{dataset}_{model_name}_results.csv', index=False)

    if env == 'cloud' and job_name == 'chief':
        for filepath in [
            f'logs/{dataset}_{model_name}_results.csv'
        ]:
            source = f'{project_dir}/{filepath}'
            destination = filepath
            logger.info(f'Upload {source} to {destination}')
            bucket.upload(source, destination)

    logger.info('Done')


if __name__ == '__main__':
    main()
