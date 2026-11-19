# param-lab — instructions for Claude Code

Python 3.12 project. Code in `src/`, tests in `tests/` (pytest).

## Commands

- Setup: `uv venv --python 3.12 .venv; .\.venv\Scripts\Activate.ps1; uv pip install -r requirements.txt`
- Run: `streamlit run app.py` (ML logic in `src/ml.py`, screen in `app.py`)
- Slides: `slides/slides.md` (Marp). Video script: `docs/video-script.md`. Keep numbers in both in sync with the app.
- Test: `pytest` (CI runs the same on every push)

## Rules

- Always use the project `.venv` — never install into the global Python or `ai-env` without asking.
- Keep code simple and commented (the owner is learning). Add a test when behavior changes; run `pytest` before saying it's done.
- Commits: short, simple one-line messages. No Co-Authored-By or any attribution trailer.
- Never commit `.env` or secrets.
