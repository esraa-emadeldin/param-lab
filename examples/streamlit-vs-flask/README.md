# Why Streamlit? The same demo, built twice

Both files show the **same** page: pick a learning rate, move the steps slider, and see the
learned `w`, `b` and the line. Both use the same ML code (`src/ml.py`).

| | `streamlit_version.py` | `flask_version.py` |
|---|---|---|
| Lines of real code | **18** | **52** |
| Languages | Python | Python + HTML + Jinja + a bit of JavaScript |
| A slider | `st.slider(...)`, one line, value is already an `int` | HTML `<input type="range">` + read `request.args` + convert text → `int` + check the range |
| A chart | `st.pyplot(fig)` | save the figure to PNG in memory → base64 → `<img>` tag |
| Updates when you move a slider | automatic (the script reruns) | only because we added `onchange="this.form.submit()"`, which reloads the **whole page** |
| Good for | ML demos, dashboards, class projects | real websites: many pages, logins, APIs, many users |

**Conclusion:** for Param Lab, the screen is not the lesson; the ML is. Streamlit lets almost every
line of `app.py` be about the ML. Flask gives full control, but most of the extra code is web plumbing.

## Run them
From the project folder, with the `.venv` activated:

```powershell
uv pip install -r examples\streamlit-vs-flask\requirements.txt   # adds Flask
streamlit run examples\streamlit-vs-flask\streamlit_version.py
python examples\streamlit-vs-flask\flask_version.py               # then open http://127.0.0.1:5000
```
