from typing import Tuple

from src.config.base_configs import TrainConfig, EvalConfig
from src.data.msmarco.preprocessors import ConcatDataProcessor
from src.data.msmarco.preprocessors import DataProcessor
from src.models.msmarco import fm, naive, nrmf, representation


def ebr_config(dataset_id: str, epochs: int, data_processor: DataProcessor) -> Tuple[
    TrainConfig, EvalConfig]:
    if not data_processor:
        data_processor = ConcatDataProcessor()
    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=representation.EBR,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='ebr',
        verbose=0,
    )
    return train_config, eval_config


def naive_config(dataset_id: str, epochs: int, data_processor: DataProcessor) -> Tuple[
    TrainConfig, EvalConfig]:
    if not data_processor:
        data_processor = ConcatDataProcessor()
    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=naive.Naive,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='naive',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_simple_query_config(dataset_id: str, epochs: int, data_processor: DataProcessor) -> Tuple[
    TrainConfig, EvalConfig]:
    if not data_processor:
        data_processor = ConcatDataProcessor()
    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=nrmf.NRMFSimpleQuery,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='nrmf_simple_query',
        verbose=0,
    )
    return train_config, eval_config


def nrmf_simple_all_config(dataset_id: str, epochs: int, data_processor) -> Tuple[
    TrainConfig, EvalConfig]:
    if not data_processor:
        data_processor = ConcatDataProcessor()
    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=nrmf.NRMFSimpleAll,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='nrmf_simple_all',
        verbose=0,
    )
    return train_config, eval_config


def fwfm_query_config(dataset_id: str, epochs: int, data_processor: DataProcessor) -> Tuple[
    TrainConfig, EvalConfig]:
    if not data_processor:
        data_processor = ConcatDataProcessor()
    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=fm.FwFMQuery,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='fwfm_query',
        verbose=0,
    )
    return train_config, eval_config


def fwfm_all_config(dataset_id: str, epochs: int, data_processor: DataProcessor) -> Tuple[
    TrainConfig, EvalConfig]:
    if not data_processor:
        data_processor = ConcatDataProcessor()
    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=fm.FwFMAll,
        epochs=epochs,
        verbose=2,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='fwfm_all',
        verbose=0,
    )
    return train_config, eval_config
