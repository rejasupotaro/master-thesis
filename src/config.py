from dataclasses import dataclass
from typing import Type, Tuple

from src.data.preprocessors import ConcatDataProcessor, MultiInstanceDataProcessor
from src.data.preprocessors import DataProcessor
from src.models import attention, fm, naive, nrmf, representation
from src.models.base_model import BaseModel


@dataclass
class TrainConfig:
    dataset: str
    data_processor: DataProcessor
    data_processor_filename: str
    model: Type[BaseModel]
    epochs: int
    verbose: int = 1


@dataclass
class EvalConfig:
    dataset: str
    data_processor_filename: str
    model_name: str
    verbose: int = 1


def ebr_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=representation.EBR,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='ebr',
        verbose=0,
    )
    return train_config, eval_config


def naive_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=naive.Naive,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='naive',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=MultiInstanceDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'multi_instance_data_processor.{dataset_size}',
        model=nrmf.NRMF,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'multi_instance_data_processor.{dataset_size}',
        model_name='nrmf',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_simple_query_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=nrmf.NRMFSimpleQuery,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='nrmf_simple_query',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_simple_all_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=nrmf.NRMFSimpleAll,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='nrmf_simple_all',
        verbose=0,
    )
    return train_config, eval_config


def fm_query_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=fm.FMQuery,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='fm_query',
        verbose=0,
    )
    return train_config, eval_config


def fm_all_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=fm.FMAll,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='fm_all',
        verbose=0,
    )
    return train_config, eval_config


def autoint_simple_config(dataset_size: str, epochs: int) -> Tuple[TrainConfig, EvalConfig]:
    train_config = TrainConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor=ConcatDataProcessor(dataset_size=dataset_size),
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model=attention.Attention,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset=f'listwise.{dataset_size}',
        data_processor_filename=f'concat_data_processor.{dataset_size}',
        model_name='autoint_simple',
        verbose=0,
    )
    return train_config, eval_config
