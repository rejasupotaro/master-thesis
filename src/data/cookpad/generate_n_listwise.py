import gc
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd
import sklearn
from loguru import logger
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.cookpad.queries import preprocess_query
from src.data.cookpad.recipes import load_raw_recipes

project_dir = Path(__file__).resolve().parents[3]


def generate_listwise(i: int, interactions_df: DataFrame, recipes: Dict, train: bool,
                      max_positives_per_query: int = 100) -> int:
    if train:
        interactions_df = sklearn.utils.shuffle(interactions_df)
        query_count_df = interactions_df.groupby('query').size().reset_index(name='count')
        minor_queries = query_count_df[query_count_df['count'] <= 3]['query'].tolist()
        interactions_df = interactions_df[~interactions_df['query'].isin(minor_queries)]
    available_recipe_ids = set(recipes.keys())
    dataset = {}
    recipe_ids = []
    counter = defaultdict(int)

    interactions_df = sklearn.utils.shuffle(interactions_df)
    for key, group in tqdm(interactions_df.groupby(['session_id', 'query'])):
        example = {}
        query = ''
        for index, row in group.iterrows():
            query = row['query']
            if counter[query] > max_positives_per_query:
                break
            if not row['fetched_recipe_ids'] or row['fetched_recipe_ids'] == '[]':
                break
            example['query'] = query
            positive_doc_id = row['recipe_id']
            if 'docs' not in example:
                example['docs'] = []
            fetched_recipe_ids = str(row['fetched_recipe_ids'])
            fetched_recipe_ids = fetched_recipe_ids.replace('[', '').replace(']', '').split(',')
            new_doc_ids = [int(doc_id) for doc_id in fetched_recipe_ids]
            new_doc_ids = new_doc_ids[:row['position'] + 1]
            doc_ids = {doc['doc_id']: i for i, doc in enumerate(example['docs'])}
            recipe_ids.extend([doc['doc_id'] for doc in example['docs']])
            for doc_id in new_doc_ids:
                if doc_id not in available_recipe_ids:
                    continue
                if doc_id in doc_ids:
                    doc = example['docs'][doc_ids[doc_id]]
                    doc['label'] = 1 if doc['label'] == 1 or doc_ids == positive_doc_id else 0
                else:
                    example['docs'].append({
                        'doc_id': doc_id,
                        'label': 1 if doc_id == positive_doc_id else 0
                    })
            if len(example['docs']) > 0 and example['docs'][-1]['label'] == 0:
                example['docs'].pop()
        counter[query] += 1
        if counter[query] > max_positives_per_query:
            continue
        if len(example['docs']) > 2:
            dataset[key] = example
    dataset = list(dataset.values())
    if train:
        filename = f'listwise.cookpad.{i}.train.pkl'
    else:
        filename = f'listwise.cookpad.{i}.val.pkl'
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
    logger.info('Load recipes')
    recipes = load_raw_recipes()
    usecols = ['started_at', 'session_id', 'country', 'recipe_id', 'position', 'query', 'page', 'fetched_recipe_ids']
    interactions_df = pd.read_csv(f'{project_dir}/data/raw/interactions.csv', usecols=usecols)
    # Note that the original dataset contains invalid recipe IDs (-1)
    interactions_df.dropna(inplace=True)
    interactions_df = interactions_df[interactions_df['recipe_id'] != -1]
    interactions_df = interactions_df[interactions_df['page'] == 1]
    interactions_df = interactions_df[interactions_df['country'] == 'US']
    interactions_df = interactions_df.sort_values(by='started_at', ascending=False)

    interactions_df['query'] = interactions_df['query'].apply(preprocess_query)

    start = 0
    stop = len(interactions_df)
    step = stop // n_splits
    rows = []
    for i, to in enumerate(range(start, stop, step)):
        df = interactions_df[start:to]
        if len(df) == step:
            ith = i - 1
            logger.info(f'Genereate dataset ({ith})')
            df = df.sample(frac=frac)
            df = df.sort_values(by='started_at', ascending=True)
            train_df, val_df = train_test_split(df, train_size=train_size, shuffle=False)

            train_dataset_size = generate_listwise(ith, train_df, recipes, train=True)
            logger.info(f'listwise.{ith}.train was created with {train_dataset_size} lists')

            val_dataset_size = generate_listwise(ith, val_df, recipes, train=False)
            logger.info(f'listwise.{ith}.val was created with {val_dataset_size} lists')

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
    # 0  34030  15533
    # 1  33038  15441
    # 2  32410  15374
    # 3  34979  15927
    # 4  34519  15798
    # 5  33413  15573
    # 6  32124  15956
    # 7  32812  14153
    # 8  31042  12127
    # 9  34845  14748
    logger.info('Done')
