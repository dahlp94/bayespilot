# Deprecated after Stage 1.
# Replaced by training/train.py using a unified sklearn pipeline artifact.
#
# Prefer from the project root:
#   python -m training.train

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.train import main  # noqa: E402

if __name__ == "__main__":
    main()
