"""The SAME mini demo as flask_version.py, written with Streamlit.

Run from the project folder:  streamlit run examples/streamlit-vs-flask/streamlit_version.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # so "from src import ml" works from here

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from src import ml  # noqa: E402

st.title("Learning rate demo (Streamlit)")

# One line = a widget on the page AND its current value in a Python variable.
lr = st.radio("Learning rate", ml.LEARNING_RATES, index=2, horizontal=True)
steps = st.slider("Training steps", 0, 100, 0)

# Every time a widget changes, Streamlit simply reruns this file from the top.
hours, score = ml.load_scores()
w, b, loss = ml.gradient_descent(hours, score, lr, steps)[-1]
st.write(f"w = {w:.2f}, b = {b:.2f}, error = {loss:.1f}")

fig, ax = plt.subplots()
ax.scatter(hours, score, s=10)
ax.plot([0, 10], [b, w * 10 + b], lw=3)
ax.set_ylim(0, 110)
st.pyplot(fig)  # Streamlit turns the figure into an image for us
