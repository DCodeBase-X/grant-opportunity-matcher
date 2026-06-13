"""
Pytest configuration — adds the project root to sys.path so all
test files can import `src.*` without sys.path.insert hacks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
