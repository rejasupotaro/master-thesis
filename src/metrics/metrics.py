import numpy as np


def sort_and_couple(labels, scores):
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


def mean_average_precision(y_true, y_pred, threshold=0.5):
    result = 0.
    pos = 0
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, score) in enumerate(coupled_pair):
        if label > threshold:
            pos += 1.
            result += pos / (idx + 1.)
    if pos == 0:
        return 0.
    else:
        return result / pos


def discount_cumulative_gain(y_true, y_pred, k=10, threshold=0.5):
    coupled_pair = sort_and_couple(y_true, y_pred)
    result = 0.
    for i, (label, score) in enumerate(coupled_pair):
        if i >= k:
            break
        if label > threshold:
            result += (np.power(2., label) - 1.) / np.log(2. + i)
    return result


def normalized_discount_cumulative_gain(y_true, y_pred, k=10, threshold=0.5):
    idcg_val = discount_cumulative_gain(y_true, y_true, k, threshold)
    dcg_val = discount_cumulative_gain(y_true, y_pred, k, threshold)
    return dcg_val / idcg_val if idcg_val != 0 else 0


if __name__ == '__main__':
    y_true = [0, 1, 0, 0]
    y_pred = [0.6, 0.1, 0.2, 0.3]
    map_score = mean_average_precision(y_true, y_pred)
    ndcg_score = normalized_discount_cumulative_gain(y_true, y_pred)
    print(f'MAP: {map_score}, NDCG: {ndcg_score}')
