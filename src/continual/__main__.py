"""Allow ``python -m continual`` (requires src/ on PYTHONPATH)."""

from continual.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
