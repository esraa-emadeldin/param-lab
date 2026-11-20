# 🎛️ Param Lab

A small Streamlit app that shows the difference between **parameters** (learned by the model) and **hyperparameters** (chosen by us), made for a 5-minute presentation video.

- **Part 1, Parameters:** predict an exam score from study hours with a line `score = w × hours + b`.
  - Gradient descent **learns** `w` and `b` by itself. We generated the data with `w = 5`, `b = 40`, and after 100 steps the model finds **w = 5.04, b = 39.74**.
  - The **learning rate** is a hyperparameter we choose. With 0.001 the line barely moves, with 0.1 it fits quickly, and with 1.5 it 💥 explodes.
- **Part 2, Hyperparameters:** a Decision Tree predicting pass/fail. The **max_depth** slider shows underfitting and overfitting:

  | max_depth | Train | Test |
  |---|---|---|
  | 1 | 77.1% | 65.0% |
  | 4 | 93.6% | **75.0%** |
  | 15 | 100% | 65.0% |

  **GridSearchCV** with 5-fold cross-validation picks **depth 4** by itself.
- **Part 3, Real data:** the same code on two real datasets that ship with scikit-learn (no download):
  - *Diabetes* (442 patients, BMI → disease progression): our gradient descent learns **w = 10.23, b = −117.77**, exactly scikit-learn's answer.
  - *Breast cancer* (569 tumours): depth 15 gets 100% train but ~92% test. GridSearchCV picks **depth 3**.

## Files
| Path | What |
|---|---|
| `src/ml.py` | All the ML (gradient descent written in plain numpy, decision trees, GridSearchCV) |
| `app.py` | The screen only (widgets + charts) |
| `tests/` | pytest tests (ML + the app itself) |
| `slides/slides.md` | The presentation ([Marp](https://marp.app) Markdown) |
| `docs/video-script.md` | The 5-minute video script |

## Run the app
```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
streamlit run app.py
```
> On the very first run Streamlit asks for your email. Just press **Enter**.

## Run the tests
```powershell
pytest
```

## Edit the slides
1. Install the **Marp for VS Code** extension (VS Code suggests it when you open this folder).
2. Open `slides/slides.md` and click the preview icon (top right).
3. To export, open the command palette (`Ctrl+Shift+P`), choose **Marp: Export slide deck…**, then pick PDF or PPTX.

Speaker notes are the `<!-- ... -->` comments under each slide.

## Working as a team
1. `git pull` to get the latest version.
2. `git checkout -b my-change` to work on a branch.
3. Run `pytest` before every commit.
4. Push the branch and open a Pull Request. Merge only when CI is green.

> ⚠️ Don't upgrade the libraries in `requirements.txt` before the video is recorded, or the numbers may no longer match the script and slides.
