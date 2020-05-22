from src.utils.seed import set_seed
from src.utils.logger import create_logger, get_logger


def generate_triples():
    set_seed()
    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    generate_triples()
