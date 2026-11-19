"""Param Lab: parameters (learned by the model) vs hyperparameters (chosen by us).

Run it with:  streamlit run app.py
All the ML lives in src/ml.py; this file only draws the screen.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import ml

st.set_page_config(page_title="Param Lab", page_icon="🎛️", layout="centered")


# --- cached helpers: each result is computed once, then every click is instant
@st.cache_data
def scores():
    return ml.load_scores()


@st.cache_data
def training(lr, steps):
    hours, score = scores()
    return ml.gradient_descent(hours, score, lr, steps)


@st.cache_data
def pass_fail():
    return ml.load_pass_fail()


@st.cache_data
def curve():
    return ml.depth_curve(*pass_fail())


@st.cache_data
def grid_search():
    return ml.best_depth(*pass_fail())


def exploded(value):
    return not math.isfinite(value) or abs(value) > 1e6


st.title("🎛️ Param Lab")
st.caption("**Parameters** are learned by the model. **Hyperparameters** are chosen by us before training.")

# ======================================================== 1) PARAMETERS
st.header("1) Parameters: the model learns them")
st.write(
    "Predict a student's **exam score** from **study hours** with a line: `score = w × hours + b`. "
    "**w** and **b** are the parameters. Nobody types them in; gradient descent finds them."
)

lr = st.radio("Learning rate (a hyperparameter: we choose it)", ml.LEARNING_RATES, index=2, horizontal=True)
steps = st.slider("Training steps", 0, 100, 0)

w, b, loss = training(lr, steps)[-1]
c1, c2, c3 = st.columns(3)
c1.metric("w (learned)", "💥" if exploded(w) else f"{w:.2f}")
c2.metric("b (learned)", "💥" if exploded(b) else f"{b:.2f}")
c3.metric("Error (MSE)", "💥 exploded" if exploded(loss) else f"{loss:.1f}")

hours, score = scores()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.scatter(hours, score, s=14, color="#9A9DA6", label="students")
xs = np.array([0, 10])
ax.plot(xs, 5 * xs + 40, "--", color="#5A5D66", lw=1, label="true rule (5 × hours + 40)")
if not exploded(w) and not exploded(b):
    ax.plot(xs, w * xs + b, color="#2F5FC4", lw=3, label=f"model after {steps} steps")
ax.set_ylim(0, 110)  # an exploding line simply flies off the chart
ax.set_xlabel("study hours per week")
ax.set_ylabel("exam score")
ax.legend(loc="lower right", fontsize=8)
st.pyplot(fig)
plt.close(fig)

st.divider()

# =================================================== 2) HYPERPARAMETERS
st.header("2) Hyperparameters: we choose them")
st.write(
    "Predict **pass / fail** for 200 students with a **Decision Tree**. "
    "**max_depth** (how many questions the tree may ask) is a hyperparameter: the tree can't learn it, we must pick it."
)

if "best" not in st.session_state:
    st.session_state.best = None  # remembered after the button is clicked

depth = st.slider("max_depth", 1, 15, 1)
depths, train, test = curve()
c1, c2 = st.columns(2)
c1.metric("Train accuracy (data it has seen)", f"{train[depth - 1]:.1%}")
c2.metric("Test accuracy (new students)", f"{test[depth - 1]:.1%}")

if st.button("Find the best max_depth (GridSearchCV)", type="primary"):
    st.session_state.best = grid_search()

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(depths, train, "o-", color="#9A9DA6", label="train")
ax.plot(depths, test, "o-", color="#B5530F", lw=3, label="test (new students)")
ax.axvline(depth, color="#16181D", lw=1, ls=":")
ax.text(1.2, 0.62, "underfitting", fontsize=9)
ax.text(11, 0.62, "overfitting", fontsize=9)
if st.session_state.best:
    best, acc = st.session_state.best
    ax.plot(best, test[best - 1], "*", ms=20, color="#2F5FC4", label="best (GridSearchCV)")
ax.set_ylim(0.6, 1.02)
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.legend(loc="center right", fontsize=8)
st.pyplot(fig)
plt.close(fig)

if st.session_state.best:
    best, acc = st.session_state.best
    st.success(f"Best max_depth: {best} → test accuracy {acc:.1%} (chosen with 5-fold cross-validation)")
