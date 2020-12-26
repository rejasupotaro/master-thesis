from typing import List

from pandas import DataFrame


def preprocess_query(query: str) -> str:
    return str(query).replace('"', '')


def get_popular_queries(interactions_df: DataFrame, top_n: int) -> List[str]:
    queries_df = interactions_df[['query']].groupby('query').size().reset_index(name='count')
    queries_df = queries_df.sort_values('count', ascending=False)
    return queries_df.head(top_n)['query'].tolist()
