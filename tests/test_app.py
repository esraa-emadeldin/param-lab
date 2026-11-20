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
    assert len(at.radio) == 2  # learning rate (sections 1 and 3a)
    assert len(at.slider) == 4  # steps + max depth, for fake AND real data
    assert len(at.button) == 2  # find the best depth (sections 2 and 3b)


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


# ------------------------------------------------ section 3: real data


def test_real_data_gradient_descent_matches_sklearn():
    at = start()
    at.radio(key="real_lr").set_value(0.1).run()
    at.slider(key="real_steps").set_value(100).run()
    assert "10.23" in values(at)  # w, learned by OUR gradient descent
    assert "-117.77" in values(at)  # b, same as scikit-learn's exact answer


def test_real_data_huge_learning_rate_explodes():
    at = start()
    at.radio(key="real_lr").set_value(1.5).run()
    at.slider(key="real_steps").set_value(100).run()
    assert "💥 exploded" in values(at)


def test_real_data_grid_search():
    at = start()
    at.button(key="real_grid").click().run()
    assert any("Best max_depth: 3" in m.value for m in at.success)
