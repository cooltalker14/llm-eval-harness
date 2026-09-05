"""CLI: generate outputs for the golden dataset against a served model."""

import argparse
import asyncio
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dataset import load as load_dataset
from .runner import run, save


def main():
    ap = argparse.ArgumentParser(description="Generate eval outputs")
    ap.add_argument("--dataset", type=Path, default=Path("data/incidents.jsonl"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    args = ap.parse_args()

    items = load_dataset(args.dataset)
    result = asyncio.run(
        run(items, args.base_url, args.model, args.tag,
            temperature=args.temperature, max_tokens=args.max_tokens,
            concurrency=args.concurrency, cache_dir=args.cache_dir)
    )
    result.env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "dataset": str(args.dataset),
        "n_items": len(items),
    }

    out = args.out_dir / f"{args.tag}.json"
    save(result, out)
    print(f"[run:{args.tag}] {len(items)} items, {result.n_errors} errors, "
          f"{result.wall_time_s:.1f}s, {result.total_completion_tokens} completion tokens")
    print(f"[run:{args.tag}] wrote {out}")


if __name__ == "__main__":
    main()
