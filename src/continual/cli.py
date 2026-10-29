"""CLI: ``python -m continual demo`` or ``python -m continual eval``."""

from __future__ import annotations

import argparse

from continual.data import StreamConfig, generate_stream
from continual.eval import format_comparison, run_comparison
from continual.pipeline import ContinualPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="continual")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run one stream and print per-batch reports")
    demo.add_argument("--batches", type=int, default=40)
    demo.add_argument("--drift-at", type=int, default=20)
    demo.add_argument("--no-replay", action="store_true")
    sub.add_parser("eval", help="compare replay vs no-replay across a drift")
    args = parser.parse_args(argv)

    if args.command == "eval":
        print(format_comparison(run_comparison()))
        return 0

    cfg = StreamConfig(n_batches=args.batches, drift_at=args.drift_at)
    pipeline = ContinualPipeline()
    pipeline.config.use_replay = not args.no_replay
    for report in pipeline.run(generate_stream(cfg)):
        fired = [s.detector for s in report.signals if s.fired]
        print(
            f"batch {report.batch_id:>2} regime={report.regime} "
            f"acc={report.accuracy:.3f} action={report.action} drift={fired}"
        )
    print(pipeline.status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
