"""The Streamlit vs Flask example: both versions must work and show the same numbers."""

import importlib.util
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FOLDER = Path(__file__).parent.parent / "examples" / "streamlit-vs-flask"


def test_streamlit_version():
    at = AppTest.from_file(str(FOLDER / "streamlit_version.py"), default_timeout=60).run()
    at.slider[0].set_value(100).run()
    assert "w = 5.04, b = 39.74" in at.markdown[0].value


def test_flask_version():
    pytest.importorskip("flask")  # Flask is optional: CI only installs the main requirements
    spec = importlib.util.spec_from_file_location("flask_version", FOLDER / "flask_version.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    page = module.app.test_client().get("/?lr=0.1&steps=100").get_data(as_text=True)
    assert "w = 5.04, b = 39.74" in page  # same numbers as the Streamlit version
    assert "data:image/png;base64," in page  # the chart is there

    bad = module.app.test_client().get("/?lr=999&steps=-5").get_data(as_text=True)
    assert "w = 0.00, b = 0.00" in bad  # bad input is fixed, not a crash
