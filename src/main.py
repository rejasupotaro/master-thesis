from src.data import data_processors
from src.models import naive, nrmf, nrmf_concat
import evaluate_model
import train_model
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed


def run_naive():
    get_logger().info('Train naive model')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size='medium'),
        'data_processor_filename': 'concat_data_processor.medium',
        'model': naive.Naive,
        'model_filename': 'naive.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate naive model')
    config = {
        'dataset': 'listwise.medium',
        'data_processor_filename': 'concat_data_processor.medium',
        'model_filename': 'naive.h5',
    }
    evaluate_model.evaluate(config)


def run_nrmf():
    get_logger().info('Train NRM-F')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.MultiInstanceDataProcessor(dataset_size='medium'),
        'data_processor_filename': 'multi_instance_data_processor.medium',
        'model': nrmf.NRMF,
        'model_filename': 'nrmf.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate NRM-F')
    config = {
        'dataset': 'listwise.medium',
        'data_processor_filename': 'multi_instance_data_processor.medium',
        'model_filename': 'nrmf.h5',
    }
    evaluate_model.evaluate(config)


def run_nrmf_concat():
    get_logger().info('Train NRM-F (Concat)')
    config = {
        'dataset': 'listwise.medium',
        'data_processor': data_processors.ConcatDataProcessor(dataset_size='medium'),
        'data_processor_filename': 'concat_data_processor.medium',
        'model': nrmf_concat.NRMFConcat,
        'model_filename': 'nrmf_concat.h5',
        'epochs': 10,
    }
    train_model.train(config)

    get_logger().info('Evaluate NRM-F (Concat)')
    config = {
        'dataset': 'listwise.medium',
        'data_processor_filename': 'concat_data_processor.medium',
        'model_filename': 'nrmf_concat.h5',
    }
    evaluate_model.evaluate(config)


if __name__ == '__main__':
    create_logger()
    set_seed()
    run_naive()
    # run_nrmf()
    # run_nrmf_concat()
