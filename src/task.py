from pathlib import Path
import os

import click

from src.data import data_processors
from src.data.cloud_storage import CloudStorage
from src.models import naive, nrmf, nrmf_concat
from src.train_model import train
from src.evaluate_model import evaluate
from src.utils.logger import create_logger, get_logger


def naive_config():
    train_config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size='small'),
        'data_processor_filename': 'concat_data_processor',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 1,
        'verbose': 2,
    }
    eval_config = {
        'dataset': 'listwise.small',
        'data_processor_filename': 'concat_data_processor',
        'model_filename': 'naive.h5',
        'verbose': 0,
    }
    return train_config, eval_config


def nrmf_config():
    train_config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.MultiInstanceDataProcessor(dataset_size='small'),
        'data_processor_filename': 'multi_instance_data_processor',
        'model': nrmf.NRMF,
        'model_filename': 'nrmf.h5',
        'epochs': 1,
        'verbose': 2,
    }
    eval_config = {
        'dataset': 'listwise.small',
        'data_processor_filename': 'multi_instance_data_processor',
        'model_filename': 'nrmf.h5',
        'verbose': 0,
    }
    return train_config, eval_config


def nrmf_concat_config():
    train_config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size='small'),
        'data_processor_filename': 'concat_data_processor',
        'model': nrmf_concat.NRMFConcat,
        'model_filename': 'nrmf_concat.h5',
        'epochs': 1,
        'verbose': 2,
    }
    eval_config = {
        'dataset': 'listwise.small',
        'data_processor_filename': 'concat_data_processor',
        'model_filename': 'nrmf_concat.h5',
        'verbose': 0,
    }
    return train_config, eval_config


@click.command()
@click.option('--job-dir')
@click.option('--bucket-name')
def main(job_dir, bucket_name):
    get_logger().info(f'Task is lauched with arguments job-dir: {job_dir}, bucket-name: {bucket_name}')

    get_logger().info('Download data')
    project_dir = Path(__file__).resolve().parents[1]
    Path(os.path.join(project_dir, 'data', 'raw')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'processed')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'models')).mkdir(exist_ok=True)
    CloudStorage(bucket_name).download('data/processed/recipes.small.pkl',
                                       os.path.join(project_dir, 'data', 'processed', 'recipes.small.pkl'))
    CloudStorage(bucket_name).download('data/processed/listwise.small.train.pkl',
                                       os.path.join(project_dir, 'data', 'processed', 'listwise.small.train.pkl'))
    CloudStorage(bucket_name).download('data/processed/listwise.small.test.pkl',
                                       os.path.join(project_dir, 'data', 'processed', 'listwise.small.test.pkl'))

    # train_config, eval_config = naive_config()
    # train_config, eval_config = nrmf_config()
    train_config, eval_config = nrmf_concat_config()

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
