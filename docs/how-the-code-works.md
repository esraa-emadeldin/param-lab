# How Param Lab works (explained for absolute beginners)

No ML or web experience needed. Read it top to bottom in about 10 minutes.

---

## 1. The big idea in one minute

Every machine-learning model has **two kinds of numbers**:

| | **Parameters** | **Hyperparameters** |
|---|---|---|
| Who picks them? | The model, by itself, while training | **You**, before training starts |
| Everyday example | What a student *learns* while studying | *How* the student studies: how many hours, how fast they read |
| In our app | `w` and `b` of a line | learning rate, training steps, `max_depth` of a tree |

> **Rule of thumb:** if the computer finds it from the data → parameter.
> If you have to type it in → hyperparameter.

Param Lab is a small website where you move sliders (hyperparameters) and watch what the model learns (parameters).

---

## 2. The files: who does what

```
param-lab/
├── app.py         ← the SCREEN: sliders, buttons, charts (Streamlit)
├── src/ml.py      ← the BRAIN: all the machine learning
├── tests/         ← robots that check everything still works (pytest)
└── examples/streamlit-vs-flask/  ← the same mini app built two ways
```

**Why split screen and brain?** You can test the brain without opening a browser, and you could swap the
screen (e.g. Flask) without touching the ML. Each file has one job.

---

## 3. The brain: `src/ml.py`

### Part 1: Parameters → `gradient_descent()`

**Goal:** draw a straight line through the dots: `score = w × hours + b`.
- `w` = how many extra points one more study hour gives (the slope)
- `b` = the score with 0 hours of study (where the line starts)

**How the computer finds `w` and `b`: like walking down a hill blindfolded.**
1. Start with a flat line: `w = 0`, `b = 0`. (The model knows nothing.)
2. Measure how wrong it is: the **error** (MSE = average of the squared mistakes).
3. Feel which way is "downhill" (that's the *gradient*) and take a small step that way.
4. Repeat. Each step, the line fits the dots a bit better.

**The learning rate = the size of each step.** It's a hyperparameter, so *you* choose it:

| Learning rate | What happens |
|---|---|
| 0.001 | Baby steps. After 100 steps it has barely moved. 🐢 |
| 0.1 | Good steps. Finds the right line. ✅ |
| 1.5 | Giant leaps. It jumps *over* the valley, further every time. 💥 |

**How do we know it works?** We *made* the fake data with `w = 5`, `b = 40`. The model finds
**w = 5.04, b = 39.74** by itself. Nobody told it the answer.

### Part 2: Hyperparameters → `depth_curve()` and `best_depth()`

A **decision tree** predicts pass/fail by asking yes/no questions, like the game "20 Questions":
*"Studied more than 5 hours?" → "Slept more than 7 hours?" → … → PASS.*

**`max_depth` = how many questions in a row it may ask.** The tree can't choose this itself; you do.

| max_depth | What happens | Like a student who… |
|---|---|---|
| 1 (too small) | **Underfitting**: too simple, bad everywhere | studied only the chapter titles |
| 4 (just right) | Best on **new** students: 75% | understood the ideas |
| 15 (too big) | **Overfitting**: 100% on students it saw, 65% on new ones | memorised last year's exam answers |

**Train vs test:** we hide 30% of the students during training (the *test set*). The test score tells us
how the model does on people it has **never seen**, and that's the score that matters.

**`GridSearchCV` = let the computer try every depth for you.** It splits the training data into 5 parts,
trains on 4 and checks on the 5th, five times (**5-fold cross-validation**, like 5 practice exams), and picks the
depth with the best average. It never peeks at the test set, so the final check stays fair.

### Part 3: Real data

The same functions run on two **real** datasets that come inside scikit-learn (no download):
- **Diabetes** (442 patients, BMI → disease progression). With real data, nobody knows the "true" line,
  so `exact_line()` asks scikit-learn for the exact answer. Our gradient descent reaches it:
  **w = 10.23, b = −117.77**.
- **Breast cancer** (569 tumours). Same tree, same `max_depth` lesson. Overfitting is smaller here because real
  medical measurements are cleaner than our fake students.

> 💡 **Same code, new data, same lesson.** That's the point of Part 3.

---

## 4. The screen: `app.py` and Streamlit

### How Streamlit thinks
Streamlit has **one simple rule: whenever you touch a widget, it runs `app.py` again from the top.**

```python
steps = st.slider("Training steps", 0, 100, 0)   # draws a slider AND gives you its value
w, b, loss = training("students", lr, steps)[-1] # use it like any Python variable
st.metric("w (learned)", f"{w:.2f}")              # show the result
```
No HTML, no JavaScript, no "send the value to the server". Just Python.

### Three speed tricks (so it's smooth on a laptop)

| Trick | What it does | Everyday version |
|---|---|---|
| `@st.cache_data` | Remembers each result. Same inputs → instant answer. | Writing answers in a notebook instead of solving the same sum again |
| `@st.fragment` | Moving a slider reruns **only its own section** | Cleaning only your room, not the whole house |
| `plt.close(fig)` | Throws away each chart after showing it | Clearing the table after dinner, so plates don't pile up |

`st.session_state` is Streamlit's memory between reruns. We use it to remember the GridSearchCV result
after you click the button.

---

## 5. Why Streamlit and not Flask?

We built the **same** mini page twice in `examples/streamlit-vs-flask/`:

| | Streamlit | Flask |
|---|---|---|
| Lines of real code | **18** | **52** |
| You write | Python | Python + HTML + Jinja + JavaScript |
| A chart | `st.pyplot(fig)` | save to PNG → base64 → `<img>` tag |
| Updates when you move a slider | automatically | only because we added `onchange="this.form.submit()"` |

**Streamlit** is great for ML demos, dashboards and class projects.
**Flask** is for real websites (logins, many pages, many users). Our app is a demo, so Streamlit wins.

---

## 6. Tests: robots that check your work

`pytest` runs **21 small checks**, for example:
- "With lr 0.1, does the model learn w ≈ 5?"
- "With lr 1.5, does it explode?"
- "Does depth 15 overfit?"
- "Does the app show `Best max_depth: 3` after clicking the button?"

If you change something and break it, a test turns red and tells you **what** broke.
GitHub runs the same tests on every push (that's **CI**).

---

## 7. How to run it (step by step)

Open PowerShell in the `param-lab` folder:

```powershell
# 1) First time only: make a private Python box (.venv) and install the libraries
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# 2) Every time: switch on the box, then start the app
.\.venv\Scripts\Activate.ps1
streamlit run app.py            # opens http://localhost:8501 in your browser

# 3) Check that everything still works
pytest
```

Stop the app with **Ctrl + C** in the terminal.

---

## 8. Mini dictionary

| Word | Simple meaning |
|---|---|
| **Model** | A formula that turns inputs (hours) into a prediction (score) |
| **Training** | Letting the model adjust its parameters to fit the data |
| **Parameter** | A number the model learns (`w`, `b`) |
| **Hyperparameter** | A setting you choose before training (learning rate, `max_depth`) |
| **Loss / error (MSE)** | How wrong the model is. Lower = better. |
| **Gradient descent** | Walking downhill step by step to make the error smaller |
| **Learning rate** | How big each downhill step is |
| **Train / test set** | Data the model learns from / data kept hidden to check it fairly |
| **Overfitting** | Memorising instead of understanding: great on old data, bad on new |
| **Underfitting** | Too simple to learn the pattern at all |
| **Cross-validation** | Practice exams on the training data to pick hyperparameters fairly |
| **Cache** | Saved answers so you don't compute the same thing twice |
