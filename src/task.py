import json
import os
from pathlib import Path

import click
import tensorflow as tf

from src.data import data_processors
from src.data.cloud_storage import CloudStorage
from src.evaluate_model import evaluate
from src.models import naive, nrmf, nrmf_concat
from src.train_model import train
from src.utils.logger import create_logger, get_logger


def naive_config(dataset_size):
    train_config = {
        'dataset': f'listwise.{dataset_size}',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size=dataset_size),
        'data_processor_filename': f'concat_data_processor.{dataset_size}',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 1,
        'verbose': 2,
    }
    eval_config = {
        'dataset': f'listwise.{dataset_size}',
        'data_processor_filename': f'concat_data_processor.{dataset_size}',
        'model_filename': 'naive.h5',
        'verbose': 0,
    }
    return train_config, eval_config


def nrmf_config(dataset_size):
    train_config = {
        'dataset': f'listwise.{dataset_size}',
        'data_processor': data_processors.MultiInstanceDataProcessor(dataset_size=dataset_size),
        'data_processor_filename': f'multi_instance_data_processor.{dataset_size}',
        'model': nrmf.NRMF,
        'model_filename': 'nrmf.h5',
        'epochs': 1,
        'verbose': 2,
    }
    eval_config = {
        'dataset': f'listwise.{dataset_size}',
        'data_processor_filename': f'multi_instance_data_processor.{dataset_size}',
        'model_filename': 'nrmf.h5',
        'verbose': 0,
    }
    return train_config, eval_config


def nrmf_concat_config(dataset_size):
    train_config = {
        'dataset': f'listwise.{dataset_size}',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size=dataset_size),
        'data_processor_filename': f'concat_data_processor.{dataset_size}',
        'model': nrmf_concat.NRMFConcat,
        'model_filename': 'nrmf_concat.h5',
        'epochs': 1,
        'verbose': 2,
    }
    eval_config = {
        'dataset': f'listwise.{dataset_size}',
        'data_processor_filename': f'concat_data_processor.{dataset_size}',
        'model_filename': 'nrmf_concat.h5',
        'verbose': 0,
    }
    return train_config, eval_config


@click.command()
@click.option('--job-dir')
@click.option('--bucket-name')
@click.option('--model-name')
@click.option('--dataset-size')
def main(job_dir, bucket_name, model_name, dataset_size):
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    # ClusterSpec({'chief': ['cmle-training-master-e118e0f997-0:2222'], 'ps': [...], 'worker': [...]})
    cluster_info = env.get('cluster', None)
    cluster_spec = tf.train.ClusterSpec(cluster_info)
    # {'type': 'worker', 'index': 3, 'cloud': 'w93a1503672d4dd09-ml'}
    get_logger().info(f'[ClusterSpec] {cluster_spec}')

    get_logger().info(f'Task is lauched with arguments job-dir: {job_dir}, bucket-name: {bucket_name}')

    get_logger().info('Download data')
    project_dir = Path(__file__).resolve().parents[1]
    Path(os.path.join(project_dir, 'data', 'raw')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'processed')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'models')).mkdir(exist_ok=True)
    CloudStorage(bucket_name).download(f'data/processed/recipes.{dataset_size}.pkl',
                                       os.path.join(project_dir, 'data', 'processed', f'recipes.{dataset_size}.pkl'))
    CloudStorage(bucket_name).download(f'data/processed/listwise.{dataset_size}.train.pkl',
                                       os.path.join(project_dir, 'data', 'processed',
                                                    f'listwise.{dataset_size}.train.pkl'))
    CloudStorage(bucket_name).download(f'data/processed/listwise.{dataset_size}.test.pkl',
                                       os.path.join(project_dir, 'data', 'processed',
                                                    f'listwise.{dataset_size}.test.pkl'))

    train_config, eval_config = {
        'naive': naive_config,
        'nrmf': nrmf_config,
        'nrmf_concat': nrmf_concat_config,
    }[model_name](dataset_size)

    get_logger().info('Train model')
    train(train_config)
    get_logger().info('Evaluate model')
    evaluate(eval_config)

    get_logger().info('Upload log')
    CloudStorage(bucket_name).upload(os.path.join(project_dir, 'logs', '1.log'), 'logs/1.log')

    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    get_logger()
    main()
