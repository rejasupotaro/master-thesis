import numpy as np
from src.metrics.metrics import sort_and_couple
from src.metrics.metrics import mean_average_precision
from src.metrics.metrics import discount_cumulative_gain, normalized_discount_cumulative_gain


def test_sort_and_couple():
    labels = [0, 1, 0]
    scores = [0.1, 0.6, 0.3]
    actual = sort_and_couple(labels, scores)
    expected = [[1, 0.6], [0, 0.3], [0, 0.1]]
    assert np.array_equal(actual, expected)


def test_mean_average_precision():
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.6, 0.4]
    actual = mean_average_precision(y_true, y_pred)
    expected = 1.0
    assert actual == expected


def test_discount_cumulative_gain():
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.6, 0.4]
    actual = discount_cumulative_gain(y_true, y_pred)
    assert actual == 1.0


def test_normalized_dicount_cumulative_gain():
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.6, 0.4]
    actual = normalized_discount_cumulative_gain(y_true, y_pred)
    expected = 1.0
    assert actual == expected
