from src.data.queries import preprocess_query


def test_process_query():
    assert preprocess_query('"pancake mix" cake') == "pancake mix cake"
