"""All the machine-learning logic for Param Lab.

PARAMETERS      = numbers the model LEARNS from the data (here: w and b of a line).
HYPERPARAMETERS = settings WE choose before training (here: learning rate, tree depth).

Every function is "data in -> numbers out", so it is easy to test and explain.
app.py only draws the screen and calls these.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42  # same seed everywhere -> same numbers on every run (good for recording)
LEARNING_RATES = (0.001, 0.01, 0.1, 1.5)  # the choices shown in the app
DEPTHS = list(range(1, 16))  # tree depths we try


# ================================================= PART 1: PARAMETERS
def load_scores():
    """Generated data: 100 students, study hours -> exam score.

    We make it with the rule  score = 5 * hours + 40  (+ some noise),
    so we KNOW the right answer is w = 5, b = 40. Can the model learn it?
    """
    rng = np.random.default_rng(SEED)
    hours = rng.uniform(0, 10, 100)
    score = np.clip(5 * hours + 40 + rng.normal(0, 5, 100), 0, 100)
    return hours, score


def gradient_descent(hours, score, lr: float, steps: int):
    """Learn the line  score = w * hours + b  with gradient descent.

    lr (learning rate) and steps are HYPERPARAMETERS: we choose them.
    w and b are PARAMETERS: the loop finds them by itself.
    Returns a list of (w, b, loss) for step 0, 1, ..., steps.
    """
    # Feature scaling: work with x = (hours - mean) / std, which keeps
    # gradient descent stable. We convert w and b back to normal units below.
    mean, std = hours.mean(), hours.std()
    x = (hours - mean) / std

    w_s, b_s = 0.0, 0.0  # start from a flat line at 0: the model knows nothing yet
    history = []
    with np.errstate(over="ignore", invalid="ignore"):  # a huge lr may overflow; that's the point
        for step in range(steps + 1):
            error = (w_s * x + b_s) - score
            loss = float(np.mean(error**2))  # mean squared error
            # back to "points per study hour" and "score with 0 hours"
            history.append((float(w_s / std), float(b_s - w_s * mean / std), loss))
            if step == steps:
                break
            # The gradient says which way to move w and b to reduce the loss.
            w_s -= lr * 2 * np.mean(error * x)
            b_s -= lr * 2 * np.mean(error)
    return history


# ============================================ PART 2: HYPERPARAMETERS
def load_pass_fail():
    """Generated data: 200 students, 4 features -> pass (1) or fail (0).

    10% of the labels are flipped ('unlucky' students), so a very deep tree
    that memorises every student will do worse on new students: overfitting.
    """
    rng = np.random.default_rng(SEED)
    n = 200
    study_hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(40, 100, n)
    past_grade = rng.uniform(30, 100, n)
    sleep_hours = rng.uniform(4, 9, n)
    score = 0.5 * study_hours + 0.05 * attendance + 0.05 * past_grade + 0.5 * sleep_hours
    y = (score > np.median(score)).astype(int)
    unlucky = rng.random(n) < 0.10  # bad luck on exam day: result flipped
    y[unlucky] = 1 - y[unlucky]
    X = np.column_stack([study_hours, attendance, past_grade, sleep_hours])
    return X, y


def _split(X, y):
    """Same train/test split everywhere (30% test, class balance kept)."""
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


def _tree(depth):
    return DecisionTreeClassifier(max_depth=depth, random_state=SEED)


def depth_curve(X, y):
    """Train one tree per max_depth; return (depths, train accuracies, test accuracies)."""
    X_tr, X_te, y_tr, y_te = _split(X, y)
    train, test = [], []
    for depth in DEPTHS:
        tree = _tree(depth).fit(X_tr, y_tr)
        train.append(tree.score(X_tr, y_tr))
        test.append(tree.score(X_te, y_te))
    return DEPTHS, train, test


def best_depth(X, y):
    """Let GridSearchCV pick max_depth with 5-fold cross-validation.

    It only looks at the TRAINING data (split into 5 parts), so the test set
    stays a fair final check. Returns (best depth, its test accuracy).
    """
    X_tr, X_te, y_tr, y_te = _split(X, y)
    search = GridSearchCV(_tree(None), {"max_depth": DEPTHS}, cv=5).fit(X_tr, y_tr)
    return search.best_params_["max_depth"], search.best_estimator_.score(X_te, y_te)


# ============================================ PART 3: REAL DATA
# Both datasets ship INSIDE scikit-learn, so nothing is downloaded and the app
# works offline. The functions above work on them unchanged: that's the point,
# the ideas are the same on real data.


def load_diabetes_bmi():
    """Real data: 442 diabetes patients, BMI -> disease progression one year later.

    scaled=False gives the real units (BMI 18-42, progression 25-346).
    """
    data = load_diabetes(scaled=False)
    bmi = data.data[:, 2]  # column 2 is BMI
    return bmi, data.target


def exact_line(x, y):
    """The best possible w and b, computed exactly by scikit-learn (no gradient descent).

    With real data nobody knows the "true" w and b, so we compare our
    gradient descent with this answer instead.
    """
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def load_tumours():
    """Real data: 569 breast tumours, 30 measurements -> malignant (0) or benign (1)."""
    return load_breast_cancer(return_X_y=True)
