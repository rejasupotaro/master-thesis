import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage

from src.utils.logger import create_logger, get_logger

load_dotenv()


class CloudStorage:
    def __init__(self, bucket_name=None):
        if not bucket_name:
            bucket_name = os.getenv('BUCKET_NAME')
        self.bucket = storage.Client().bucket(bucket_name)

    def upload(self, source, destination):
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(source)

    def download(self, source, destination):
        blob = self.bucket.blob(source)
        blob.download_to_filename(destination)


if __name__ == '__main__':
    create_logger()
    project_dir = Path(__file__).resolve().parents[2]
    cloud_storage = CloudStorage()

    for filename in [
        'listwise.small.train.pkl',
        'listwise.small.test.pkl',
        'listwise.medium.train.pkl',
        'listwise.medium.test.pkl',
        'recipes.large.pkl',
        'recipes.medium.pkl',
        'recipes.small.pkl',
    ]:
        source = os.path.join(project_dir, 'data', 'processed', filename)
        destination = f'data/processed/{filename}'
        get_logger().info(f'Upload {source} to {destination}')
        cloud_storage.upload(source, destination)
    get_logger().info('Done')
