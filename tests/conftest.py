"""Make repository-only experiment scripts importable with either pytest entrypoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
