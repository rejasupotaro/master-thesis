from typing import List

import numpy as np


def sort_and_couple(labels: List[int], scores: List[float]):
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


def mean_average_precision(y_true: List[int], y_pred: List[float], threshold: float = 0.5):
    result = 0.0
    pos = 0
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, score) in enumerate(coupled_pair):
        if label > threshold:
            pos += 1
            result += pos / (idx + 1.0)
    if pos == 0:
        return 0.0
    else:
        return result / pos


def discount_cumulative_gain(y_true: List[int], y_pred: List[float], k: int = 20, threshold: float = 0.5):
    coupled_pair = sort_and_couple(y_true, y_pred)
    result = 0.0
    for i, (label, score) in enumerate(coupled_pair):
        if i >= k:
            break
        if label > threshold:
            result += (np.power(2.0, label) - 1.0) / np.log2(2.0 + i)
    return result


def normalized_discount_cumulative_gain(y_true: List[int], y_pred: List[float], k: int = 20, threshold: float = 0.5):
    idcg_val = discount_cumulative_gain(y_true, y_true, k, threshold)
    dcg_val = discount_cumulative_gain(y_true, y_pred, k, threshold)
    return dcg_val / idcg_val if idcg_val != 0 else 0
