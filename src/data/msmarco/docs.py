from pathlib import Path
from typing import Set, Dict

import pandas as pd

project_dir = Path(__file__).resolve().parents[3]


def load_raw_docs() -> Dict:
    docs_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-docs.tsv.gz',
        delimiter='\t',
        header=None,
        names=['doc_id', 'url', 'title', 'body'],
    )
    docs_df['doc_id'] = docs_df['doc_id'].str[1:].astype(int)
    return {doc['doc_id']: doc for doc in docs_df.to_dict(orient='records')}


def load_available_doc_ids() -> Set[str]:
    docs_df = pd.read_csv(
        f'{project_dir}/data/raw/msmarco-docs.tsv.gz',
        delimiter='\t',
        header=None,
        names=['doc_id', 'url', 'title', 'body'],
    )
    docs_df['doc_id'] = docs_df['doc_id'].str[1:].astype(int)
    return set(docs_df['doc_id'].tolist())
