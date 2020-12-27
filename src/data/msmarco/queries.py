def preprocess_query(query: str) -> str:
    return str(query).replace('"', '')
