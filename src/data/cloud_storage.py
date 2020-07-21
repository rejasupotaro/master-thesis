import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Bucket

from src.utils.logger import create_logger, get_logger

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
    create_logger()
    cloud_storage = CloudStorage()

    for filepath in [
        'data/processed/listwise.small.train.pkl',
        'data/processed/listwise.small.val.pkl',
        'data/processed/listwise.medium.train.pkl',
        'data/processed/listwise.medium.val.pkl',
        'data/processed/recipes.large.pkl',
        'data/processed/recipes.medium.pkl',
        'data/processed/recipes.small.pkl',
    ]:
        source = f'{project_dir}/{filepath}'
        destination = filepath
        get_logger().info(f'Upload {source} to {destination}')
        cloud_storage.upload(source, destination)
    get_logger().info('Done')
