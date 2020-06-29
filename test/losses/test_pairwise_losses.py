import numpy as np

from src.losses.pairwise_losses import cross_entropy_loss, hinge_loss


def test_cross_entropy_loss():
    y_true = [1, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.1, 0.9]
    actual = cross_entropy_loss(y_true, y_pred)
    assert actual[0] < actual[1]


def test_hinge_loss():
    y_true = [1, 0, 1, 0]
    y_pred = [0.0, 0.0, 1.0, 0.0]
    actual = hinge_loss(y_true, y_pred)
    expected = [0.5, 0.5]
    assert np.array_equal(actual, expected)
