import json
import os
import sys
from pathlib import Path
from time import time

import click
import mlflow
import tensorflow as tf
from loguru import logger
from mlflow.tracking import MlflowClient
from pandas import DataFrame

from src import config
from src.data.cloud_storage import CloudStorage
from src.evaluate_model import evaluate
from src.train_model import train

project_dir = Path(__file__).resolve().parents[1]


def run_experiment(model_name: str, dataset_id: int, epochs: int) -> float:
    train_config, eval_config = {
        'ebr': config.ebr_config,
        'naive': config.naive_config,
        'nrmf_simple_query': config.nrmf_simple_query_config,
        'nrmf_simple_all': config.nrmf_simple_all_config,
        'fm_query': config.fm_query_config,
        'fm_all': config.fm_all_config,
    }[model_name](dataset_id, epochs)

    logger.info('Train model')
    train(train_config)

    logger.info('Evaluate model')
    ndcg_score = evaluate(eval_config)
    return ndcg_score


@click.command()
@click.option('--job-dir', type=str)
@click.option('--bucket-name', type=str)
@click.option('--env', type=str)
@click.option('--dataset-id', type=str)
@click.option('--model-name', type=str)
@click.option('--epochs', type=int)
def main(job_dir: str, bucket_name: str, env: str, dataset_id: str, model_name: str, epochs: int):
    logger.add(sys.stdout, format='{time} {level} {message}')
    log_filepath = f'{project_dir}/logs/{int(time())}.log'
    logger.add(log_filepath)
    mlflow.set_tracking_uri(f'{project_dir}/logs/mlruns')
    mlflow.start_run()
    client = MlflowClient()
    experiments = client.list_experiments()
    experiment_id = experiments[0].experiment_id
    run = client.create_run(experiment_id)  # returns mlflow.entities.Run
    run_id = run.info.run_id
    logger.info(f'experiment_id: {experiment_id}, run_id: {run_id}')

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

    results = []
    for dataset_id in dataset_ids:
        logger.info(f'Run an experiment on {model_name} with dataset: {dataset_id}')
        ndcg_score = run_experiment(model_name, dataset_id, epochs)
        results.append({
            'dataset_id': dataset_id,
            'model': model_name,
            'NDCG': ndcg_score,
        })
    results_df = DataFrame(results)
    logger.info(results_df)
    results_df.to_csv(f'{project_dir}/logs/{model_name}_results.csv', index=False)

    mlflow.log_artifact(log_filepath)
    if env == 'cloud' and job_name == 'chief':
        logger.info('Upload results')
        bucket = CloudStorage(bucket_name)
        base_filepath = f'{project_dir}/logs/mlruns/{experiment_id}'
        for file in Path(base_filepath).rglob('*'):
            if file.is_file():
                filename = str(file)[len(base_filepath) + 1:]
                destination = f'logs/mlruns/{experiment_id}/{filename}'
                bucket.upload(str(file), destination)

        for filepath in [
            f'logs/{model_name}_results.csv'
        ]:
            source = f'{project_dir}/{filepath}'
            destination = filepath
            logger.info(f'Upload {source} to {destination}')
            bucket.upload(source, destination)

    mlflow.end_run()
    logger.info('Done')


if __name__ == '__main__':
    main()
