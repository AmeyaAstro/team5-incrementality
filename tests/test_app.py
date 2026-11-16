"""Smoke-test every page with Streamlit's headless AppTest."""

import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent

PAGES = ["app_pages/verdict.py", "app_pages/segments.py", "app_pages/analyst.py", "app_pages/model.py", "app_pages/method.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_assumptions_flow_through():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.number_input(key="aov").set_value(300.0).run()
    assert not at.exception
    assert any("Profitable" in m.value or "break-even" in m.value for m in at.markdown)
