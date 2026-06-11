"""
Root conftest.py — shared pytest fixtures for all tests.
"""

import sys
from pathlib import Path

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
