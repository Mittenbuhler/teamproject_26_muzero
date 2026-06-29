import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALPHAZERO = ROOT / "alphazero"

# The AlphaZero scripts currently use both package-style imports
# ("alphazero.module") and direct sibling imports ("from training import ...").
# Adding both paths keeps pytest collection stable without changing source code.
for path in (ROOT, ALPHAZERO):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
