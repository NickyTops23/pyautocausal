import os
import sys
import pytest
from pathlib import Path

# Add the project root directory to the Python path so that
# the app package can be imported in the tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup test-specific configurations if needed
def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and conftest file after command line options have been parsed.
    """
    # Set up test-specific environment variables if needed
    os.environ.setdefault("TESTING", "True") 