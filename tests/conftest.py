import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def in_tmp(tmp_path, monkeypatch):
    """Run a test inside an empty directory (CSV logs and .client_cache land there)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
