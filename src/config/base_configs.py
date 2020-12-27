from dataclasses import dataclass
from typing import Type

from src.data.cookpad.preprocessors import DataProcessor
from src.models.base_model import BaseModel


@dataclass
class TrainConfig:
    dataset_id: str
    data_processor: DataProcessor
    model: Type[BaseModel]
    epochs: int
    dataset: str = ''
    data_processor_filename: str = ''
    verbose: int = 1

    def __post_init__(self):
        self.dataset = f'listwise.{self.dataset_id}'
        self.dataset_processor_filename = f'concat_data_processor.{self.dataset_id}'


@dataclass
class EvalConfig:
    dataset_id: str
    model_name: str
    dataset: str = ''
    data_processor_filename: str = ''
    verbose: int = 1

    def __post_init__(self):
        self.dataset = f'listwise.{self.dataset_id}'
        self.dataset_processor_filename = f'concat_data_processor.{self.dataset_id}'
