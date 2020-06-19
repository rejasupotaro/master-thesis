from src.data import data_processors
from src.models import naive, nrmf, nrmf_concat
import evaluate_model
import train_model
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed


def naive():
    get_logger().info('Train naive model')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.ConcatDataProcessor(),
        'data_processor_filename': 'concat_data_processor',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate naive model')
    config = {
        'dataset': 'listwise.medium',
        'data_processor_filename': 'concat_data_processor',
        'model_filename': 'naive.h5',
    }
    evaluate_model.evaluate(config)


def nrmf():
    get_logger().info('Train NRM-F')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.MultiInstanceDataProcessor(),
        'data_processor_filename': 'multi_instance_data_processor',
        'model': nrmf.NRMF,
        'model_filename': 'nrmf.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate NRM-F')
    config = {
        'dataset': 'listwise.medium',
        'data_processor_filename': 'multi_instance_data_processor',
        'model_filename': 'nrmf.h5',
    }
    evaluate_model.evaluate(config)


def nrmf_concat():
    get_logger().info('Train NRM-F (Concat)')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.ConcatDataProcessor(),
        'data_processor_filename': 'concat_data_processor',
        'model': nrmf_concat.NRMFConcat,
        'model_filename': 'nrmf_concat.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate NRM-F (Concat)')
    config = {
        'dataset': 'listwise.medium',
        'data_processor_filename': 'concat_data_processor',
        'model_filename': 'nrmf_concat.h5',
    }
    evaluate_model.evaluate(config)


if __name__ == '__main__':
    create_logger()
    set_seed()
    # naive()
    nrmf()
    # nrmf_concat()
