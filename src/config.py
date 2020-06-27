from dataclasses import dataclass
from typing import Type
from src.data.data_processors import DataProcessor
from src.models.base_model import BaseModel


@dataclass
class TrainConfig:
    dataset: str
    data_processor: DataProcessor
    data_processor_filename: str
    model: Type[BaseModel]
    model_filename: str
    epochs: int
    verbose: int = 1


@dataclass
class EvalConfig:
    dataset: str
    data_processor_filename: str
    model_filename: str
    verbose: int = 1
