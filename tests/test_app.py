from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_review_does_not_leak_to_another_sample():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    next(t for t in app.text_input if t.label == "Member Id").set_value("SYN02000001")
    next(b for b in app.button if b.label == "Save reviewed record").click().run()
    assert not app.exception
    assert app.session_state["review_saved"]["reviewed_fields"]["member_id"] == "SYN02000001"
    sample = next(s for s in app.selectbox if s.label == "Sample")
    sample.select_index(0).run(timeout=60)
    assert not app.exception
    assert "review_saved" not in app.session_state
    assert any("conflicting" in warning.value for warning in app.warning)
