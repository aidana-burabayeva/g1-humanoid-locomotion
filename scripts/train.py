"""Start the registered 10 cm stair task with mjlab's training CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import g1_locomotion.tasks  # noqa: E402,F401
from g1_locomotion.tasks import stairs10 as stairs10_tasks  # noqa: E402
from mjlab.scripts.train import main  # noqa: E402

stairs10_tasks.register()

if __name__ == "__main__":
  main()
