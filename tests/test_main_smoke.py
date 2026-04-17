from pathlib import Path


def test_main_file_exists():
    assert Path("main.py").exists()


def test_main_contains_expected_keywords():
    content = Path("main.py").read_text(encoding="utf-8")

    assert "Config" in content
    assert "DataManager" in content
    assert "ExpandingWindowScheme" in content