from typing import Tuple, Dict, List

from pandas import DataFrame
from src.data.data_generator import DataGenerator
from src.data.preprocessors import DataProcessor


class EmptyDataProcessor(DataProcessor):
    def __init__(self, dataset_size: str = ''):
        pass

    def process_df(self, df: DataFrame) -> None:
        pass

    def fit(self, df: DataFrame) -> None:
        pass

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        return df.to_dict('records'), [0] * len(df)


def test_getitem():
    df = DataFrame([
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4},
        {'a': 5, 'b': 6},
    ])
    processor = EmptyDataProcessor()
    generator = DataGenerator(df, processor, 2)
    assert len(generator) == 2
    print(generator[0])
    batch = generator[0]
    x, y = batch
    assert x == [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    assert y == [0, 0]
