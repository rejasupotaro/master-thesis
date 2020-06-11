def preprocess_query(query):
    return str(query).replace('"', '')


def get_popular_queries(interactions_df, top_n):
    interactions_df['processed_query'] = interactions_df['query'].apply(preprocess_query)
    queries_df = interactions_df[['processed_query']].groupby('processed_query').size().reset_index(name='count')
    queries_df = queries_df.sort_values('count', ascending=False)
    return queries_df.head(top_n)['processed_query'].tolist()