"""
Same recipe, different weights: a reproducible experiment on the weights of small neural networks.

Command-line entry point. Prints the results as JSON (same format as results.json).
The experiment itself lives in core.py; the GUI is app.py.

    python run_experiment.py > results.json
"""
import json, sys
from core import run_all

out, _ = run_all()
sys.stdout.reconfigure(encoding="utf-8")
json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
print()
