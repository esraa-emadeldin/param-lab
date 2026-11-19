"""Tests for src/ml.py: parameters (gradient descent) and hyperparameters (tree depth)."""

import math

from src.ml import best_depth, depth_curve, gradient_descent, load_pass_fail, load_scores

# ------------------------------------------------ PARAMETERS (learned)


def test_scores_data():
    hours, score = load_scores()
    assert len(hours) == len(score) == 100
    assert 0 <= hours.min() and hours.max() <= 10


def test_history_length():
    hours, score = load_scores()
    assert len(gradient_descent(hours, score, lr=0.1, steps=30)) == 31  # step 0 + 30 steps


def test_good_lr_finds_true_line():
    # The data was made with score = 5 * hours + 40 (+ noise), so the model
    # should LEARN w ≈ 5 and b ≈ 40 by itself.
    hours, score = load_scores()
    w, b, loss = gradient_descent(hours, score, lr=0.1, steps=100)[-1]
    assert abs(w - 5) < 0.6
    assert abs(b - 40) < 3
    assert loss < gradient_descent(hours, score, lr=0.1, steps=100)[0][2]


def test_tiny_lr_is_too_slow():
    hours, score = load_scores()
    slow = gradient_descent(hours, score, lr=0.001, steps=100)[-1][2]
    good = gradient_descent(hours, score, lr=0.1, steps=100)[-1][2]
    assert slow > 5 * good


def test_huge_lr_explodes():
    hours, score = load_scores()
    history = gradient_descent(hours, score, lr=1.5, steps=100)
    assert history[-1][2] > 1000 * history[0][2] or not math.isfinite(history[-1][2])


def test_deterministic():
    hours, score = load_scores()
    assert gradient_descent(hours, score, 0.01, 50) == gradient_descent(hours, score, 0.01, 50)


# ------------------------------------------- HYPERPARAMETERS (chosen)


def test_deep_tree_overfits():
    X, y = load_pass_fail()
    depths, train, test = depth_curve(X, y)
    assert depths == list(range(1, 16))
    assert train[-1] >= 0.99  # depth 15 memorises the training data...
    assert train[-1] - test[-1] >= 0.08  # ...but does clearly worse on new data
    assert max(test) - test[-1] >= 0.05  # overfitting: test accuracy falls for deep trees
    assert max(test) - test[0] >= 0.05  # underfitting: depth 1 is too simple


def test_grid_search_finds_a_good_depth():
    X, y = load_pass_fail()
    depth, acc = best_depth(X, y)
    _, _, test = depth_curve(X, y)
    assert 2 <= depth <= 8
    assert acc >= max(test) - 0.04
