import numpy as np

from src.losses.pairwise_losses import hinge_loss


def test_hinge_loss():
    y_true = [0, 1, 1, 0]
    y_pred = [0.0, 0.0, 1.0, 0.0]
    actual = hinge_loss(y_true, y_pred)
    expected = [0.5, 0.5]
    assert np.array_equal(actual, expected)
