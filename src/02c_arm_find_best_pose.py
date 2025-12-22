"""
Compatibilità: questo file è stato rinominato in `src/02_exploration_and_best_arm_pose.py`.
Usa direttamente quello per lo step "02".
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    runpy.run_path(str(here / "02_exploration_and_best_arm_pose.py"), run_name="__main__")
