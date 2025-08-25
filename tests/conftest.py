import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path so imports work correctly
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def src_path():
    """Return the src directory path."""
    return SRC_PATH


# Configure pytest to show more detailed output for failed assertions
def pytest_configure(config):
    """Configure pytest settings."""
    config.option.verbose = True
