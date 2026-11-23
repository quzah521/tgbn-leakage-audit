"""pytest fixtures; the test files also run as plain scripts (python tests/test_audit.py)."""
import pytest


@pytest.fixture
def out(tmp_path):
    """Output directory for the heuristic runs of one test."""
    return str(tmp_path)
