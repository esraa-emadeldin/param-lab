"""Tests that drive the Streamlit app like a user would."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest looks for relative paths next to THIS test file, so give it a full path.
APP = str(Path(__file__).parent.parent / "app.py")


def start():
    return AppTest.from_file(APP, default_timeout=120).run()


def values(at):
    return [m.value for m in at.metric]


def test_app_loads():
    at = start()
    assert not at.exception
    assert len(at.radio) == 1  # learning rate
    assert len(at.slider) == 2  # training steps + max depth
    assert len(at.button) == 1  # find the best depth


def test_good_learning_rate_learns_the_line():
    at = start()
    at.radio[0].set_value(0.1).run()
    at.slider[0].set_value(100).run()
    assert "5.04" in values(at)  # w, learned (true value: 5)
    assert "39.74" in values(at)  # b, learned (true value: 40)


def test_huge_learning_rate_explodes():
    at = start()
    at.radio[0].set_value(1.5).run()
    at.slider[0].set_value(100).run()
    assert "💥 exploded" in values(at)


def test_grid_search_button():
    at = start()
    at.button[0].click().run()
    assert any("Best max_depth: 4" in m.value for m in at.success)


def test_depth_slider_shows_overfitting():
    at = start()
    at.slider[1].set_value(15).run()
    assert "100.0%" in values(at)  # train accuracy
    assert "65.0%" in values(at)  # test accuracy
