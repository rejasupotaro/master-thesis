import os
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed
from src.data.recipes import load_available_recipe_ids
from src.data.queries import get_popular_queries

project_dir = Path(__file__).resolve().parents[2]


def generate(train_size=0.8):
    """Generate listwise JSON from interctions.csv
    The JSON format is like below.
    [
        {'query': 'chicken': [{'doc_id': 1, 'label': 1}, {'doc_id': 2, 'label': 0}]},
        ...
    ]
    Each row represents how a user interacted with a list of search results and hence,
    it consists of a query with several documents with its ID and label (clicked=1, not clicked=0).
    """
    get_logger().info('Load available recipe IDs')
    available_recipe_ids = load_available_recipe_ids()

    get_logger().info('Genereate listwise datasets')
    # Note that the original dataset contains invalid recipe IDs (-1).
    interactions_df = pd.read_csv(os.path.join(project_dir, 'data', 'raw', 'interactions.csv'))
    interactions_df = interactions_df[interactions_df['recipe_id'] != -1]
    large_dataset = defaultdict(dict)
    for index, row in tqdm(interactions_df.iterrows(), total=len(interactions_df)):
        if row['page'] != 1:
            continue

        example = large_dataset[row['session_id']]
        example['query'] = row['query']
        positive_doc_id = row['recipe_id']
        if positive_doc_id not in available_recipe_ids:
            continue

        if 'docs' not in example:
            example['docs'] = []
        new_doc_ids = [int(doc_id) for doc_id in row['fetched_recipe_ids'].split(',')]
        new_doc_ids = new_doc_ids[:row['position'] + 1]
        doc_ids = {doc['doc_id']: i for i, doc in enumerate(example['docs']) if doc['doc_id']}
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

    large_dataset = list(large_dataset.values())
    get_logger().info(f'Large listwise dataset was created with {len(large_dataset)} groups')  # 1153947 groups
    train_dataset, test_dataset = train_test_split(large_dataset, train_size=train_size, shuffle=True)
    with open(os.path.join(project_dir, 'data', 'processed', f'listwise.large.train.pkl'), 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(os.path.join(project_dir, 'data', 'processed', f'listwise.large.test.pkl'), 'wb') as file:
        pickle.dump(test_dataset, file)

    top30_queries = get_popular_queries(interactions_df, 30)
    top10_queries = set(top30_queries[:10])
    top30_queries = set(top30_queries)
    medium_dataset = [example for example in large_dataset if example['query'] in top30_queries]
    get_logger().info(f'Medium listwise dataset was created with {len(medium_dataset)} groups')  # 154978 groups
    train_dataset, test_dataset = train_test_split(medium_dataset, train_size=train_size, shuffle=True)
    with open(os.path.join(project_dir, 'data', 'processed', f'listwise.medium.train.pkl'), 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(os.path.join(project_dir, 'data', 'processed', f'listwise.medium.test.pkl'), 'wb') as file:
        pickle.dump(test_dataset, file)

    small_dataset = [example for example in medium_dataset if example['query'] in top10_queries]
    get_logger().info(f'Small listwise dataset was created with {len(small_dataset)} groups')  # 77161 groups
    train_dataset, test_dataset = train_test_split(small_dataset, train_size=train_size, shuffle=True)
    with open(os.path.join(project_dir, 'data', 'processed', f'listwise.small.train.pkl'), 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(os.path.join(project_dir, 'data', 'processed', f'listwise.small.test.pkl'), 'wb') as file:
        pickle.dump(test_dataset, file)

    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    set_seed()
    generate()
