"""The SAME mini demo as streamlit_version.py, written with Flask.

Run from the project folder:  python examples/streamlit-vs-flask/flask_version.py
Then open http://127.0.0.1:5000 in your browser.
"""

import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # so "from src import ml" works from here

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw charts in memory: a web server has no screen
import matplotlib.pyplot as plt  # noqa: E402
from flask import Flask, render_template_string, request  # noqa: E402

from src import ml  # noqa: E402

app = Flask(__name__)

# In Flask WE write the page: HTML for every widget, plus a little JavaScript
# (onchange) so the page reloads when a widget moves. Streamlit did all this for us.
PAGE = """
<!doctype html>
<title>Learning rate demo (Flask)</title>
<h1>Learning rate demo (Flask)</h1>
<form method="get">
  Learning rate:
  {% for value in rates %}
    <label><input type="radio" name="lr" value="{{ value }}"
      {% if value == lr %}checked{% endif %} onchange="this.form.submit()"> {{ value }}</label>
  {% endfor %}
  <br>Training steps: <b>{{ steps }}</b>
  <input type="range" name="steps" min="0" max="100" value="{{ steps }}" onchange="this.form.submit()">
</form>
<p>w = {{ "%.2f"|format(w) }}, b = {{ "%.2f"|format(b) }}, error = {{ "%.1f"|format(loss) }}</p>
<img src="data:image/png;base64,{{ chart }}">
"""


def chart_as_png(hours, score, w, b):
    """Flask can't show a matplotlib figure: we must save it as a PNG ourselves."""
    fig, ax = plt.subplots()
    ax.scatter(hours, score, s=10)
    ax.plot([0, 10], [b, w * 10 + b], lw=3)
    ax.set_ylim(0, 110)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


@app.route("/")
def home():
    # Widget values arrive as TEXT in the URL (?lr=0.1&steps=50): we must read
    # and convert them ourselves, and pick defaults when they are missing.
    lr = float(request.args.get("lr", 0.1))
    steps = int(request.args.get("steps", 0))
    if lr not in ml.LEARNING_RATES:
        lr = 0.1  # never trust input from the browser
    steps = max(0, min(steps, 100))

    hours, score = ml.load_scores()
    w, b, loss = ml.gradient_descent(hours, score, lr, steps)[-1]
    return render_template_string(
        PAGE, rates=ml.LEARNING_RATES, lr=lr, steps=steps, w=w, b=b, loss=loss,
        chart=chart_as_png(hours, score, w, b),
    )


if __name__ == "__main__":
    app.run(debug=False)
