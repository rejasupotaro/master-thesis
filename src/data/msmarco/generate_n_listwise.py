import gc
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from loguru import logger
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.msmarco.queries import preprocess_query

project_dir = Path(__file__).resolve().parents[3]


def generate_listwise(i: int, top100_df: DataFrame, qrels_df: DataFrame, train: bool) -> int:
    top100_df = sklearn.utils.shuffle(top100_df)
    dataset = []
    doc_ids = []
    for query, group in tqdm(top100_df.groupby(['query'])):
        example = {
            'query': query,
            'docs': [],
        }
        for row in group.to_dict(orient='records'):
            doc_id = row['doc_id']
            label = len(qrels_df[(qrels_df['query_id'] == row['query_id']) & (qrels_df['doc_id'] == doc_id)])
            example['docs'].append({
                'doc_id': doc_id,
                'label': label,
            })
            doc_ids.append(doc_id)
        dataset.append(example)
    if train:
        filename = f'listwise.msmarco.{i}.train.pkl'
    else:
        filename = f'listwise.msmarco.{i}.val.pkl'
    with open(f'{project_dir}/data/processed/{filename}', 'wb') as file:
        pickle.dump(dataset, file)
    return len(dataset)


def generate(n_splits: int = 10, frac: float = 0.6, train_size: float = 0.75):
    """Generate listwise JSON from interctions.csv
    The JSON format is like below.
    [
        {'query': 'chicken': [{'doc_id': 1, 'label': 1}, {'doc_id': 2, 'label': 0}]},
        ...
    ]
    Each row represents how a user interacted with a list of search results and hence,
    it consists of a query with several documents with its ID and label (clicked=1, not clicked=0).
    """
    queries_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-doctrain-queries.tsv.gz',
        delimiter='\t',
        header=None,
        names=['query_id', 'query'],
    )
    qrels_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-doctrain-qrels.tsv.gz',
        delimiter=' ',
        header=None,
        names=['query_id', 'Q0', 'doc_id', 'relevancy'],
    )
    qrels_df['doc_id'] = qrels_df['doc_id'].str[1:].astype(int)
    qrels_df = qrels_df[['query_id', 'doc_id']]
    interactions_df = qrels_df.merge(queries_df, on='query_id')
    interactions_df = interactions_df[['query', 'doc_id']]
    interactions_df = sklearn.utils.shuffle(interactions_df)
    interactions_df['query'] = interactions_df['query'].apply(preprocess_query)

    top100_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-doctrain-top100.gz',
        delimiter=' ',
        header=None,
        names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'runstring'],
    )
    top100_df['doc_id'] = top100_df['doc_id'].str[1:].astype(int)
    top100_df = top100_df[['query_id', 'doc_id']]
    top100_df = top100_df.merge(queries_df, on='query_id')

    start = 0
    stop = len(top100_df)
    step = stop // n_splits
    rows = []
    for i, to in enumerate(range(start, stop, step)):
        df = top100_df[start:to]
        if len(df) == step:
            ith = i - 1
            logger.info(f'Genereate dataset ({ith})')
            train_df, val_df = train_test_split(df, train_size=train_size, shuffle=False)

            train_dataset_size = generate_listwise(ith, train_df, qrels_df, train=True)
            logger.info(f'listwise.msmarco.{ith}.train was created with {train_dataset_size} lists')

            val_dataset_size = generate_listwise(ith, val_df, qrels_df, train=False)
            logger.info(f'listwise.msmarco.{ith}.val was created with {val_dataset_size} lists')

            rows.append({
                'i': ith,
                'train': train_dataset_size,
                'val': val_dataset_size
            })

            gc.collect()
        start = to
    dataset_size_df = DataFrame(rows)
    logger.info(dataset_size_df)
    # i  train  val
    # 0  27526  9177
    # 1  27526  9177
    # 2  27527  9176
    # 3  27527  9176
    # 4  27527  9177
    # 5  27527  9176
    # 6  27527  9176
    # 7  27527  9176
    # 8  27527  9176
    # 9  27527  9176
    logger.info('Done')
