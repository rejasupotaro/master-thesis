import pickle
from pathlib import Path

import pandas as pd
import sklearn
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.msmarco.queries import preprocess_query
from src.data.msmarco.docs import load_raw_docs

project_dir = Path(__file__).resolve().parents[3]


def generate(train_size: float = 0.8):
    """Generate listwise JSON from interctions.csv
    The JSON format is like below.
    [
        {'query': 'chicken': [{'doc_id': 1, 'label': 1}, {'doc_id': 2, 'label': 0}]},
        ...
    ]
    Each row represents how a user interacted with a list of search results and hence,
    it consists of a query with several documents with its ID and label (clicked=1, not clicked=0).
    """
    logger.info('Load documents')
    docs = load_raw_docs()

    queries_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-docdev-queries.tsv.gz',
        delimiter='\t',
        header=None,
        names=['query_id', 'query'],
    )
    qrels_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-docdev-qrels.tsv.gz',
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
        f'{project_dir}/data/raw/msmarco-docdev-top100.gz',
        delimiter=' ',
        header=None,
        names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'runstring'],
    )
    top100_df['doc_id'] = top100_df['doc_id'].str[1:].astype(int)
    top100_df = top100_df[['query_id', 'doc_id']]
    top100_df = top100_df.merge(queries_df, on='query_id')

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
    logger.info(f'Large listwise dataset was created with {len(dataset)} lists')
    train_dataset, val_dataset = train_test_split(dataset, train_size=train_size, shuffle=True)
    with open(f'{project_dir}/data/processed/listwise.msmarco.train.pkl', 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(f'{project_dir}/data/processed/listwise.msmarco.val.pkl', 'wb') as file:
        pickle.dump(val_dataset, file)

    docs = {doc_id: docs[doc_id] for doc_id in set(doc_ids) if doc_id in docs}
    logger.info(f'Doc data was created with {len(docs)} docs')
    with open(f'{project_dir}/data/processed/docs.msmarco.pkl', 'wb') as file:
        pickle.dump(docs, file)

    logger.info('Done')
