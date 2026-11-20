"""Param Lab: parameters (learned by the model) vs hyperparameters (chosen by us).

Run it with:  streamlit run app.py
All the ML lives in src/ml.py; this file only draws the screen.

Speed tricks (so it runs smoothly on a laptop):
- @st.cache_data   : every training result is computed once, then reused.
- @st.fragment     : each section reruns on its own. Moving a slider in
                     section 3 does NOT redraw the charts of sections 1 and 2.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import ml

st.set_page_config(page_title="Param Lab", page_icon="🎛️", layout="centered")

# The datasets, by name. Passing a short name (not the arrays) to the cached
# functions below keeps the cache keys small and fast.
LINE_DATA = {"students": ml.load_scores, "diabetes": ml.load_diabetes_bmi}
TREE_DATA = {"students": ml.load_pass_fail, "tumours": ml.load_tumours}


# --- cached helpers: each result is computed once, then every click is instant
@st.cache_data
def line_data(name):
    return LINE_DATA[name]()


@st.cache_data
def training(name, lr, steps):
    x, y = line_data(name)
    return ml.gradient_descent(x, y, lr, steps)


@st.cache_data
def exact(name):
    return ml.exact_line(*line_data(name))


@st.cache_data
def tree_data(name):
    return TREE_DATA[name]()


@st.cache_data
def curve(name):
    return ml.depth_curve(*tree_data(name))


@st.cache_data
def grid_search(name):
    return ml.best_depth(*tree_data(name))


def exploded(value):
    return not math.isfinite(value) or abs(value) > 1e6


# --- drawing helpers (used by section 1 AND section 3)
def show_learned(w, b, loss):
    c1, c2, c3 = st.columns(3)
    c1.metric("w (learned)", "💥" if exploded(w) else f"{w:.2f}")
    c2.metric("b (learned)", "💥" if exploded(b) else f"{b:.2f}")
    c3.metric("Error (MSE)", "💥 exploded" if exploded(loss) else f"{loss:.1f}")


def line_chart(x, y, w, b, steps, x_range, y_range, reference, labels):
    """Dots = data, dashed line = the reference answer, blue line = our model."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(x, y, s=14, color="#9A9DA6", label=labels["dots"])
    xs = np.array(x_range)
    ref_w, ref_b = reference
    ax.plot(xs, ref_w * xs + ref_b, "--", color="#5A5D66", lw=1, label=labels["reference"])
    if not exploded(w) and not exploded(b):
        ax.plot(xs, w * xs + b, color="#2F5FC4", lw=3, label=f"model after {steps} steps")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)  # an exploding line simply flies off the chart
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)  # free the memory: otherwise every rerun keeps an old figure


def depth_chart(depths, train, test, depth, best, y_min, label_y, legend_loc="center right"):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(depths, train, "o-", color="#9A9DA6", label="train")
    ax.plot(depths, test, "o-", color="#B5530F", lw=3, label="test (new data)")
    ax.axvline(depth, color="#16181D", lw=1, ls=":")
    ax.text(1.2, label_y, "underfitting", fontsize=9)
    ax.text(11, label_y, "overfitting", fontsize=9)
    if best:
        ax.plot(best, test[best - 1], "*", ms=20, color="#2F5FC4", label="best (GridSearchCV)")
    ax.set_ylim(y_min, 1.02)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.legend(loc=legend_loc, fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


st.title("🎛️ Param Lab")
st.caption("**Parameters** are learned by the model. **Hyperparameters** are chosen by us before training.")


# ======================================================== 1) PARAMETERS
@st.fragment
def parameters_section():
    st.header("1) Parameters: the model learns them")
    st.write(
        "Predict a student's **exam score** from **study hours** with a line: `score = w × hours + b`. "
        "**w** and **b** are the parameters. Nobody types them in; gradient descent finds them."
    )
    lr = st.radio("Learning rate (a hyperparameter: we choose it)", ml.LEARNING_RATES, index=2, horizontal=True)
    steps = st.slider("Training steps", 0, 100, 0)

    w, b, loss = training("students", lr, steps)[-1]
    show_learned(w, b, loss)
    hours, score = line_data("students")
    line_chart(
        hours, score, w, b, steps, x_range=(0, 10), y_range=(0, 110), reference=(5, 40),
        labels={"dots": "students", "reference": "true rule (5 × hours + 40)",
                "x": "study hours per week", "y": "exam score"},
    )


# =================================================== 2) HYPERPARAMETERS
@st.fragment
def hyperparameters_section():
    st.header("2) Hyperparameters: we choose them")
    st.write(
        "Predict **pass / fail** for 200 students with a **Decision Tree**. "
        "**max_depth** (how many questions the tree may ask) is a hyperparameter: the tree can't learn it, we must pick it."
    )
    if "best" not in st.session_state:
        st.session_state.best = None  # remembered after the button is clicked

    depth = st.slider("max_depth", 1, 15, 1)
    depths, train, test = curve("students")
    c1, c2 = st.columns(2)
    c1.metric("Train accuracy (data it has seen)", f"{train[depth - 1]:.1%}")
    c2.metric("Test accuracy (new students)", f"{test[depth - 1]:.1%}")

    if st.button("Find the best max_depth (GridSearchCV)", type="primary"):
        st.session_state.best = grid_search("students")

    best = st.session_state.best[0] if st.session_state.best else None
    depth_chart(depths, train, test, depth, best, y_min=0.6, label_y=0.62)

    if st.session_state.best:
        best, acc = st.session_state.best
        st.success(f"Best max_depth: {best} → test accuracy {acc:.1%} (chosen with 5-fold cross-validation)")


# ===================================================== 3) REAL DATA
@st.fragment
def real_data_section():
    st.header("3) Real data: same ideas, real patients")
    st.write(
        "Sections 1 and 2 used data we generated, so we knew the right answer. "
        "Here are two **real** datasets that ship with scikit-learn. The ML code is exactly the same."
    )

    # ---- 3a) parameters on real data
    st.subheader("3a) Parameters: diabetes (442 patients)")
    st.write(
        "Predict **disease progression after one year** from **BMI**: `progression = w × BMI + b`. "
        "With real data nobody knows the true w and b, so the dashed line is scikit-learn's "
        "**exact** answer. Can our gradient descent reach it?"
    )
    lr = st.radio("Learning rate", ml.LEARNING_RATES, index=2, horizontal=True, key="real_lr")
    steps = st.slider("Training steps", 0, 100, 0, key="real_steps")

    w, b, loss = training("diabetes", lr, steps)[-1]
    show_learned(w, b, loss)
    ex_w, ex_b = exact("diabetes")
    st.caption(f"scikit-learn's exact answer: w = {ex_w:.2f}, b = {ex_b:.2f}")
    bmi, progression = line_data("diabetes")
    line_chart(
        bmi, progression, w, b, steps, x_range=(17, 43), y_range=(0, 360), reference=(ex_w, ex_b),
        labels={"dots": "patients", "reference": "exact answer (LinearRegression)",
                "x": "BMI (body mass index)", "y": "disease progression"},
    )

    # ---- 3b) hyperparameters on real data
    st.subheader("3b) Hyperparameters: breast cancer (569 tumours)")
    st.write(
        "Predict whether a tumour is **malignant or benign** from 30 measurements. "
        "Same Decision Tree, same **max_depth** choice."
    )
    if "real_best" not in st.session_state:
        st.session_state.real_best = None

    depth = st.slider("max_depth", 1, 15, 1, key="real_depth")
    depths, train, test = curve("tumours")
    c1, c2 = st.columns(2)
    c1.metric("Train accuracy (seen tumours)", f"{train[depth - 1]:.1%}")
    c2.metric("Test accuracy (new tumours)", f"{test[depth - 1]:.1%}")

    if st.button("Find the best max_depth (GridSearchCV)", key="real_grid"):
        st.session_state.real_best = grid_search("tumours")

    best = st.session_state.real_best[0] if st.session_state.real_best else None
    # lower center: on this chart the test line sits on the right, behind a "center right" legend
    depth_chart(depths, train, test, depth, best, y_min=0.85, label_y=0.86, legend_loc="lower center")

    if st.session_state.real_best:
        best, acc = st.session_state.real_best
        st.success(f"Best max_depth: {best} → test accuracy {acc:.1%} (chosen with 5-fold cross-validation)")
    st.info(
        "Real data is cleaner than our fake students, so overfitting is **smaller** here: "
        "a deep tree still scores 100% on tumours it has seen but only ~92% on new ones. "
        "GridSearchCV never looks at the test set, so its pick is not always the test-set winner, "
        "and that's on purpose: it keeps the final check fair."
    )


parameters_section()
st.divider()
hyperparameters_section()
st.divider()
real_data_section()
