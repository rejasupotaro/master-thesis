from src.data import data_processors
from src.models import naive, nrmf
from src.models import train_model, evaluate_model
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

if __name__ == '__main__':
    create_logger()
    set_seed()

    get_logger().info('Train naive model')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.ConcatDataProcessor(),
        'build_model_fn': naive.build_model,
        'model_filename': 'naive.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate naive model')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.ConcatDataProcessor(),
        'model_filename': 'naive.h5',
    }
    evaluate_model.evaluate(config)

    get_logger().info('Train NRM-F')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.MultiInstanceDataProcessor(),
        'build_model_fn': nrmf.build_model,
        'model_filename': 'nrmf.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate NRM-F')
    config = {
        'dataset': 'listwise.small',
        'data_processor': data_processors.MultiInstanceDataProcessor(),
        'model_filename': 'nrmf.h5',
    }
    evaluate_model.evaluate(config)
