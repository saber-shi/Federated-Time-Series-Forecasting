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
    return parser.parse_args()


def build_duration_map(districts: List[str]) -> Dict[str, int]:
    if len(districts) != 10:
        raise ValueError(f"Expected exactly 10 districts, found {len(districts)}: {districts}")

    day_map: Dict[str, int] = {}
    for district in districts[:2]:
        day_map[district] = 182
    for district in districts[2:4]:
        day_map[district] = 365
    for district in districts[4:7]:
        day_map[district] = 547
    for district in districts[7:]:
        day_map[district] = 730
    return day_map


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
    duration_days = build_duration_map(districts)
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
