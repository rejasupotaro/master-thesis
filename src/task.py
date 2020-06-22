from pathlib import Path
import os

import click

from src.data import data_processors
from src.data.cloud_storage import CloudStorage
from src.models import naive
from src.train_model import train
from src.utils.logger import create_logger, get_logger


@click.command()
@click.option('--job-dir')
@click.option('--bucket-name')
def main(job_dir, bucket_name):
    project_dir = Path(__file__).resolve().parents[1]
    Path(os.path.join(project_dir, 'data', 'raw')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'processed')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'models')).mkdir(exist_ok=True)
    CloudStorage(bucket_name).download('data/raw/recipes.small.json', os.path.join(project_dir, 'data', 'raw', 'recipes.json'))
    CloudStorage(bucket_name).download('data/processed/listwise.small.train.pkl',
                                       os.path.join(project_dir, 'data', 'processed', 'listwise.small.train.pkl'))
    CloudStorage(bucket_name).download('data/processed/listwise.small.train.pkl',
                                       os.path.join(project_dir, 'data', 'processed', 'listwise.small.train.pkl'))
    CloudStorage(bucket_name).download('data/processed/listwise.small.test.pkl',
                                       os.path.join(project_dir, 'data', 'processed', '/istwise.small.test.pkl'))
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.ConcatDataProcessor(),
        'data_processor_filename': 'concat_data_processor',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 1,
    }
    train(config)


if __name__ == '__main__':
    create_logger()
    get_logger()
    main()
