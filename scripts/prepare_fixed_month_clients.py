#!/usr/bin/env python3
"""Create a CSV containing the first N complete calendar months per client."""

import argparse
import calendar
import csv
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first complete calendar months for selected clients."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--clients", nargs="+", required=True)
    parser.add_argument("--identifier", default="District")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.months <= 0:
        raise SystemExit("--months must be positive.")
    if args.interval_minutes <= 0:
        raise SystemExit("--interval-minutes must be positive.")
    if not args.source.is_file():
        raise SystemExit(f"Source dataset not found: {args.source}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output}. Use --overwrite to replace it.")

    requested_clients = list(dict.fromkeys(args.clients))
    requested_set = set(requested_clients)
    starts: Dict[str, datetime] = {}

    with args.source.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        required = {args.identifier, args.time_column}
        missing_columns = sorted(required - set(fieldnames))
        if missing_columns:
            raise SystemExit(f"Source dataset is missing columns: {missing_columns}")
        for row in reader:
            cid = row[args.identifier]
            if cid not in requested_set:
                continue
            value = datetime.strptime(row[args.time_column], TIMESTAMP_FORMAT)
            if cid not in starts or value < starts[cid]:
                starts[cid] = value

    missing_clients = sorted(requested_set - set(starts))
    if missing_clients:
        raise SystemExit(f"Source dataset is missing clients: {missing_clients}")

    cutoffs = {cid: add_months(start, args.months) for cid, start in starts.items()}
    timestamps: Dict[str, List[datetime]] = defaultdict(list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            delete=False,
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
            with args.source.open(newline="") as source:
                reader = csv.DictReader(source)
                writer = csv.DictWriter(temporary, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    cid = row[args.identifier]
                    if cid not in requested_set:
                        continue
                    value = datetime.strptime(row[args.time_column], TIMESTAMP_FORMAT)
                    if starts[cid] <= value < cutoffs[cid]:
                        writer.writerow(row)
                        timestamps[cid].append(value)

        expected_delta = timedelta(minutes=args.interval_minutes)
        expected_counts = {}
        for cid in requested_clients:
            values = sorted(timestamps[cid])
            expected_count = int((cutoffs[cid] - starts[cid]) / expected_delta)
            expected_counts[cid] = expected_count
            if len(values) != expected_count:
                raise SystemExit(
                    f"Client {cid} has {len(values)} records; expected {expected_count} "
                    f"for {args.months} continuous calendar months."
                )
            if values[0] != starts[cid] or values[-1] != cutoffs[cid] - expected_delta:
                raise SystemExit(f"Client {cid} does not cover the complete requested window.")
            for previous, current in zip(values, values[1:]):
                if current - previous != expected_delta:
                    raise SystemExit(
                        f"Client {cid} has a non-{args.interval_minutes}-minute interval "
                        f"between {previous} and {current}."
                    )

        if len(set(expected_counts.values())) != 1:
            raise SystemExit(f"Selected clients have unequal expected record counts: {expected_counts}")

        os.replace(temporary_path, args.output)
        os.chmod(args.output, 0o644)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Prepared dataset: {args.output}")
    print(f"Clients: {len(requested_clients)}")
    for cid in requested_clients:
        values = sorted(timestamps[cid])
        print(
            f"  {cid}: {len(values)} records, "
            f"{values[0].strftime(TIMESTAMP_FORMAT)} to {values[-1].strftime(TIMESTAMP_FORMAT)}"
        )


if __name__ == "__main__":
    main()
