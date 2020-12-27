import gc
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import sklearn
from loguru import logger
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.cookpad.queries import preprocess_query, get_popular_queries
from src.data.cookpad.recipes import load_raw_recipes

project_dir = Path(__file__).resolve().parents[3]


def generate_large(recipes: Dict, interactions_df: DataFrame, train_size: float,
                   max_positives_per_query: int = 100) -> Iterable:
    available_recipe_ids = set(recipes.keys())
    # Note that the original dataset contains invalid recipe IDs (-1).
    large_dataset = {}
    large_recipe_ids = []
    counter = defaultdict(int)

    for key, group in tqdm(interactions_df.groupby(['session_id', 'query'])):
        example = {}
        query = ''
        for index, row in group.iterrows():
            query = row['query']
            if counter[query] > max_positives_per_query:
                break
            example['query'] = query
            positive_doc_id = row['recipe_id']
            if 'docs' not in example:
                example['docs'] = []
            new_doc_ids = [int(doc_id) for doc_id in row['fetched_recipe_ids'].split(',')]
            new_doc_ids = new_doc_ids[:row['position'] + 1]
            doc_ids = {doc['doc_id']: i for i, doc in enumerate(example['docs'])}
            large_recipe_ids.extend([doc['doc_id'] for doc in example['docs']])
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
            if example['docs'][-1]['label'] == 0:
                example['docs'].pop()
        counter[query] += 1
        if counter[query] > max_positives_per_query:
            continue
        if len(example['docs']) > 2:
            large_dataset[key] = example
    large_dataset = list(large_dataset.values())
    logger.info(f'Large listwise dataset was created with {len(large_dataset)} lists')
    train_dataset, val_dataset = train_test_split(large_dataset, train_size=train_size, shuffle=True)
    with open(f'{project_dir}/data/processed/listwise.cookpad.large.train.pkl', 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(f'{project_dir}/data/processed/listwise.cookpad.large.val.pkl', 'wb') as file:
        pickle.dump(val_dataset, file)

    large_recipes = {recipe_id: recipes[recipe_id] for recipe_id in set(large_recipe_ids)}
    logger.info(f'Large recipe data was created with {len(large_recipes)} recipes')
    with open(f'{project_dir}/data/processed/docs.cookpad.large.pkl', 'wb') as file:
        pickle.dump(large_recipes, file)

    return large_dataset


def generate_medium(recipes: Dict, large_dataset: Iterable, target_queries: Iterable[str],
                    train_size: float) -> Iterable:
    medium_dataset = [example for example in large_dataset if example['query'] in target_queries]
    logger.info(f'Medium listwise dataset was created with {len(medium_dataset)} lists')
    train_dataset, val_dataset = train_test_split(medium_dataset, train_size=train_size, shuffle=True)
    with open(f'{project_dir}/data/processed/listwise.cookpad.medium.train.pkl', 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(f'{project_dir}/data/processed/listwise.cookpad.medium.val.pkl', 'wb') as file:
        pickle.dump(val_dataset, file)

    medium_recipe_ids = []
    for example in medium_dataset:
        medium_recipe_ids.extend([doc['doc_id'] for doc in example['docs']])
    medium_recipes = {recipe_id: recipes[recipe_id] for recipe_id in set(medium_recipe_ids)}
    logger.info(f'Medium recipe data was created with {len(medium_recipes)} recipes')
    with open(f'{project_dir}/data/processed/docs.cookpad.medium.pkl', 'wb') as file:
        pickle.dump(medium_recipes, file)

    return medium_dataset


def generate_small(recipes: Dict, medium_dataset: Iterable, train_size: float):
    np.random.shuffle(medium_dataset)
    small_dataset, _ = train_test_split(medium_dataset, train_size=0.03, shuffle=True)
    logger.info(f'Small listwise dataset was created with {len(small_dataset)} lists')
    train_dataset, val_dataset = train_test_split(small_dataset, train_size=train_size, shuffle=True)
    with open(f'{project_dir}/data/processed/listwise.cookpad.small.train.pkl', 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(f'{project_dir}/data/processed/listwise.cookpad.small.val.pkl', 'wb') as file:
        pickle.dump(val_dataset, file)

    small_recipe_ids = []
    for example in small_dataset:
        small_recipe_ids.extend([doc['doc_id'] for doc in example['docs']])
    small_recipes = {recipe_id: recipes[recipe_id] for recipe_id in set(small_recipe_ids)}
    logger.info(f'Small recipe data was created with {len(small_recipes)} recipes')
    with open(f'{project_dir}/data/processed/docs.cookpad.small.pkl', 'wb') as file:
        pickle.dump(small_recipes, file)


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
    logger.info('Load available recipe IDs')
    recipes = load_raw_recipes()
    interactions_df = pd.read_csv(f'{project_dir}/data/raw/interactions.csv')
    interactions_df = interactions_df[interactions_df['recipe_id'] != -1]
    interactions_df = interactions_df[interactions_df['page'] == 1]
    interactions_df = interactions_df[~interactions_df['session_id'].isna()]
    interactions_df = sklearn.utils.shuffle(interactions_df)

    interactions_df['query'] = interactions_df['query'].apply(preprocess_query)
    popular_queries = get_popular_queries(interactions_df, top_n=3000)
    popular_queries = set(popular_queries)

    logger.info('Genereate large dataset')
    # 351495 lists, 136764 recipes
    dataset = generate_large(recipes, interactions_df, train_size)
    gc.collect()

    logger.info('Genereate medium dataset')
    # 191144 lists, 55117 recipes
    dataset = generate_medium(recipes, dataset, popular_queries, train_size)
    gc.collect()

    logger.info('Genereate small dataset')
    # 5734 lists, 24713 recipes
    generate_small(recipes, dataset, train_size)

    logger.info('Done')
