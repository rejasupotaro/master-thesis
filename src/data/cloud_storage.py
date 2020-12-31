import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Bucket
from loguru import logger

load_dotenv()
project_dir = Path(__file__).resolve().parents[2]


class CloudStorage:
    def __init__(self, bucket_name: str = None):
        if not bucket_name:
            bucket_name = os.getenv('BUCKET_NAME')
        self.bucket: Bucket = storage.Client().bucket(bucket_name)

    def upload(self, source: str, destination: str) -> None:
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(source)

    def download(self, source: str, destination: str) -> None:
        blob = self.bucket.blob(source)
        blob.download_to_filename(destination)


if __name__ == '__main__':
    cloud_storage = CloudStorage()

    dataset = 'cookpad'
    filepaths = []
    for i in range(10):
        filepaths.append(f'data/processed/listwise.{dataset}.{i}.train.pkl')
        filepaths.append(f'data/processed/listwise.{dataset}.{i}.val.pkl')

    for filepath in filepaths:
        source = f'{project_dir}/{filepath}'
        destination = filepath
        logger.info(f'Upload {source} to {destination}')
        cloud_storage.upload(source, destination)
    logger.info('Done')
