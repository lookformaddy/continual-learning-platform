"""Standalone demo: replay vs no-replay across an abrupt drift.

Run with PYTHONPATH=src from the project root:
    PYTHONPATH=src python examples/run_demo.py
"""

from continual.data import StreamConfig
from continual.eval import format_comparison, run_comparison

if __name__ == "__main__":
    print(format_comparison(run_comparison(StreamConfig())))
