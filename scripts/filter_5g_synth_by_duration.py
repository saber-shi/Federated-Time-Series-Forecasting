#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


ROWS_PER_DAY = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a synthetic 5G CSV to mixed per-station history lengths.")
    parser.add_argument("--input_path", type=str, default="dataset/5G-2y-firstcell-10stations.csv")
    parser.add_argument("--output_path", type=str, default="dataset/5G-2y-firstcell-10stations-mixed-duration.csv")
    parser.add_argument(
        "--duration_spec",
        type=str,
        default="182x2,365x2,547x3,730x3",
        help="Comma-separated day-count groups, e.g. '7x2,30x2,365x2' means 2 stations at 7 days, 2 at 30, 2 at 365.",
    )
    parser.add_argument(
        "--district_limit",
        type=int,
        default=0,
        help="Optional number of sorted districts to include before assigning durations. 0 means all districts.",
    )
    return parser.parse_args()


def parse_duration_spec(spec: str) -> List[int]:
    durations: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" not in chunk:
            raise ValueError(f"Invalid duration chunk '{chunk}'. Expected format '<days>x<count>'.")
        days_str, count_str = chunk.split("x", 1)
        days = int(days_str)
        count = int(count_str)
        if days <= 0 or count <= 0:
            raise ValueError(f"Invalid non-positive duration chunk '{chunk}'.")
        durations.extend([days] * count)
    return durations


def build_duration_map(districts: List[str], duration_days: List[int]) -> Dict[str, int]:
    if len(districts) != len(duration_days):
        raise ValueError(
            f"District count ({len(districts)}) does not match duration assignments ({len(duration_days)})."
        )
    return {district: days for district, days in zip(districts, duration_days)}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Input CSV is missing a header.")

        rows_by_district = defaultdict(list)
        for row in reader:
            rows_by_district[row["District"]].append(row)

    districts = sorted(rows_by_district)
    if args.district_limit > 0:
        districts = districts[: args.district_limit]

    duration_days_list = parse_duration_spec(args.duration_spec)
    duration_days = build_duration_map(districts, duration_days_list)
    duration_rows = {district: days * ROWS_PER_DAY for district, days in duration_days.items()}

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for district in districts:
            rows = rows_by_district[district]
            keep_rows = duration_rows[district]
            if len(rows) < keep_rows:
                raise ValueError(
                    f"District {district} has only {len(rows)} rows, but {keep_rows} rows are required."
                )
            writer.writerows(rows[:keep_rows])

    kept_counts = Counter()
    for district in districts:
        kept_counts[district] = duration_rows[district]

    print(f"Saved filtered dataset to: {output_path}")
    print("Kept rows per district:")
    for district in districts:
        print(f"  {district}: {kept_counts[district]} rows ({duration_days[district]} days)")


if __name__ == "__main__":
    main()
