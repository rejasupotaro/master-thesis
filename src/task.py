import json
import os
from pathlib import Path
from typing import Tuple

import click
import mlflow
import tensorflow as tf
from mlflow.tracking import MlflowClient

from src.config import TrainConfig, EvalConfig
from src.data import preprocessors
from src.data.cloud_storage import CloudStorage
from src.evaluate_model import evaluate
from src.models import naive, nrmf, nrmf_concat
from src.train_model import train
from src.utils.logger import create_logger, get_logger

project_dir = Path(__file__).resolve().parents[1]


def naive_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=preprocessors.ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=naive.Naive,
        model_filename='naive.h5',
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_filename='naive.h5',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=preprocessors.MultiInstanceDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'multi_instance_data_processor.{dataset_size}',
        model=nrmf.NRMF,
        model_filename='nrmf.h5',
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'multi_instance_data_processor.{dataset_size}',
        model_filename='nrmf.h5',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_concat_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=preprocessors.ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=nrmf_concat.NRMFConcat,
        model_filename='nrmf_concat.h5',
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_filename='nrmf_concat.h5',
        verbose=0,
    )
    return train_config, eval_config


@click.command()
@click.option('--job-dir', type=str)
@click.option('--bucket-name', type=str)
@click.option('--env', type=str)
@click.option('--dataset-size', type=str)
@click.option('--model-name', type=str)
@click.option('--epochs', type=int)
def main(job_dir: str, bucket_name: str, env: str, dataset_size: str, model_name: str, epochs: int):
    mlflow.set_tracking_uri(os.path.join(project_dir, 'logs', 'mlruns'))
    mlflow.start_run()
    client = MlflowClient()
    experiments = client.list_experiments()
    experiment_id = experiments[0].experiment_id
    run = client.create_run(experiment_id)  # returns mlflow.entities.Run
    run_id = run.info.run_id
    get_logger().info(f'experiment_id: {experiment_id}, run_id: {run_id}')

    if env == 'cloud':
        tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
        # ClusterSpec({'chief': ['cmle-training-master-e118e0f997-0:2222'], 'ps': [...], 'worker': [...]})
        cluster_info = tf_config.get('cluster', None)
        cluster_spec = tf.train.ClusterSpec(cluster_info)
        # {'type': 'worker', 'index': 3, 'cloud': 'w93a1503672d4dd09-ml'}
        task_info = tf_config.get('task', None)
        job_name, task_index = task_info['type'], task_info['index']
        get_logger().info(f'cluster_spec {cluster_spec}, job_name: {job_name}, task_index: {task_index}')
        get_logger().info(f'Task is lauched with arguments job-dir: {job_dir}, bucket-name: {bucket_name}')
    else:
        job_name = 'chief'

    if env == 'cloud':
        get_logger().info('Download data')
        bucket = CloudStorage(bucket_name)
        Path(os.path.join(project_dir, 'data', 'raw')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(project_dir, 'data', 'processed')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(project_dir, 'models')).mkdir(exist_ok=True)
        for source in [
            f'data/processed/recipes.{dataset_size}.pkl',
            f'data/processed/listwise.{dataset_size}.train.pkl',
            f'data/processed/listwise.{dataset_size}.test.pkl',
        ]:
            destination = os.path.join(project_dir, source)
            get_logger().info(f'Download {source} to {destination}')
            bucket.download(source, destination)

    train_config, eval_config = {
        'naive': naive_config,
        'nrmf': nrmf_config,
        'nrmf_concat': nrmf_concat_config,
    }[model_name](dataset_size, epochs)

    get_logger().info('Train model')
    train(train_config)

    get_logger().info('Evaluate model')
    evaluate(eval_config)

    mlflow.log_artifact(os.path.join(project_dir, 'logs', '1.log'))
    if env == 'cloud' and job_name == 'chief':
        get_logger().info('Upload results')
        bucket = CloudStorage(bucket_name)
        base_filepath = os.path.join(project_dir, 'logs', 'mlruns', experiment_id)
        for file in Path(base_filepath).rglob('*'):
            if file.is_file():
                filename = str(file)[len(base_filepath) + 1:]
                destination = f'logs/mlruns/{experiment_id}/{filename}'
                bucket.upload(str(file), destination)

    mlflow.end_run()
    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    get_logger()
    main()
