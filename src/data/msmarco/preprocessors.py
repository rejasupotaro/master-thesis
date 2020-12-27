import abc
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from src.data.cookpad.recipes import load_recipes, load_raw_recipes

project_dir = Path(__file__).resolve().parents[3]


class DataProcessor(abc.ABC):
    def __init__(self, docs: Dict = None, dataset_size: str = None, num_words: int = 2000000,
                 max_negatives: int = 10):
        if not docs:
            if dataset_size:
                self.docs = load_recipes(dataset_size)
            else:
                self.docs = load_raw_recipes()
        else:
            self.docs = docs
        self.num_words: int = num_words
        self.tokenizer: Optional[Tokenizer] = None
        self.encoder: Dict[str, LabelEncoder] = {}
        self.max_negatives: int = max_negatives

    @property
    def doc_id_encoder(self) -> LabelEncoder:
        return self.encoder['doc_id']

    @property
    def total_words(self) -> int:
        return len(self.tokenizer.word_index) + 1

    def listwise_to_pairs(self, listwise_filename: str) -> DataFrame:
        with open(f'{project_dir}/data/processed/{listwise_filename}', 'rb') as file:
            dataset = pickle.load(file)

        rows = []
        for example in tqdm(dataset):
            query = example['query']
            positives = []
            negatives = []
            for doc in example['docs']:
                data = {
                    'query': query,
                    'doc_id': doc['doc_id'],
                    'label': doc['label']
                }
                if doc['label'] == 1:
                    positives.append(data)
                else:
                    negatives.append(data)

            for positive in positives:
                for negative in negatives[:self.max_negatives]:
                    rows.append(positive)
                    rows.append(negative)
        return DataFrame(rows)

    def listwise_to_random_pairs(self, listwise_filename: str) -> DataFrame:
        project_dir = Path(__file__).resolve().parents[2]
        with open(f'{project_dir}/data/processed/{listwise_filename}', 'rb') as file:
            dataset = pickle.load(file)

        positives = []
        negatives = []
        for example in tqdm(dataset):
            query = example['query']
            for doc in example['docs']:
                data = {
                    'query': query,
                    'doc_id': doc['doc_id'],
                    'label': doc['label']
                }
                if doc['label'] == 1:
                    positives.append(data)
                negatives.append(data)

        rows = []
        import random

        for positive in tqdm(positives):
            for _ in range(self.max_negatives):
                rows.append(positive)
                rows.append(random.choice(negatives))
        return DataFrame(rows)

    @abc.abstractmethod
    def process_df(self, df: DataFrame) -> None:
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def fit(self, df: DataFrame) -> None:
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        raise NotImplementedError('Calling an abstract method.')


class ConcatDataProcessor(DataProcessor):
    def process_df(self, df: DataFrame) -> None:
        df['doc_id'] = df['doc_id'].astype(np.int64)
        df['query'] = df['query'].astype(str)
        df['url'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['url']).astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['title']).astype(str)
        df['body'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['body']).astype(str)
        df['label'] = df['label'].astype(float)

    def fit(self, df: DataFrame) -> None:
        self.process_df(df)

        self.encoder['doc_id'] = LabelEncoder()
        doc_ids = [doc_id for doc_id in self.docs] + [-1]
        self.encoder['doc_id'].fit(doc_ids)

        sentences = set()
        sentences |= set(df['query'])
        sentences |= set(df['url'])
        sentences |= set(df['title'])
        sentences |= set(df['body'])
        self.tokenizer = Tokenizer(
            oov_token='<OOV>',
            char_level=False,
            num_words=self.num_words,
        )
        self.tokenizer.fit_on_texts(sentences)
        del sentences

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        df = df.copy()
        self.process_df(df)

        df['doc_id'] = df['doc_id'].apply(lambda c: c if c in self.encoder['doc_id'].classes_ else -1)
        df['doc_id'] = self.encoder['doc_id'].transform(df['doc_id'])
        doc_id = df['doc_id'].to_numpy()

        df['query'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['url'] = self.tokenizer.texts_to_sequences(df['url'].tolist())
        df['title'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['body'] = self.tokenizer.texts_to_sequences(df['body'].tolist())

        query = df['query'].tolist()
        query = pad_sequences(
            query,
            padding='post',
            truncating='post',
            maxlen=20
        )
        url = df['url'].tolist()
        url = pad_sequences(
            url,
            padding='post',
            truncating='post',
            maxlen=20
        )
        title = df['title'].tolist()
        title = pad_sequences(
            title,
            padding='post',
            truncating='post',
            maxlen=20
        )
        body = df['body'].tolist()
        body = pad_sequences(
            body,
            padding='post',
            truncating='post',
            maxlen=12000
        )
        label = df['label'].to_numpy()

        return {
                   'doc_id': doc_id,
                   'query': query,
                   'url': url,
                   'title': title,
                   'body': body,
               }, label
