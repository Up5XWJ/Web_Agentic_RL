"""Run this checkout even when a copied virtualenv still points to OpenApps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from open_apps.agentic_rl.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
