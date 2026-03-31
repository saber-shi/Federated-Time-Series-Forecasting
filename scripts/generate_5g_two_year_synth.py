#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


OUTPUT_COLUMNS = [
    "Base Station ID",
    "District",
    "time",
    "Timestamp",
    "PRB Usage Ratio (%)",
    "Traffic Volume (KByte)",
    "Number of Users",
    "BBU Energy (W)",
    "RRU Energy (W)",
    "Channel Shutdown Time (Millisecond)",
    "Channel Shutdown Time (Millisecond).1",
    "Deep Sleep Time (Millisecond)",
]

INPUT_METRIC_COLUMNS = [
    "PRB Usage Ratio (%)",
    "Traffic Volume (KByte)",
    "Number of Users",
    "BBU Energy (W)",
    "RRU Energy (W)",
    "Channel Shutdown Time (Millisecond)",
    "Channel Shutdown Time (Millisecond).1",
    "Deep Sleep Time (Millisecond)",
]

TRAFFIC_COLUMNS = [
    "PRB Usage Ratio (%)",
    "Traffic Volume (KByte)",
    "Number of Users",
    "RRU Energy (W)",
]

POWER_COLUMNS = [
    "BBU Energy (W)",
]


@dataclass
class DailyProfile:
    timestamps: List[str]
    metrics: Dict[str, List[float]]
    stats: Dict[str, Tuple[float, float]]
    zero_like_columns: Dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2-year synthetic 5G data from 1-day weekday/weekend snapshots.")
    parser.add_argument("--weekday_path", type=str, default="dataset/5G-weekday-1d.csv")
    parser.add_argument("--weekend_path", type=str, default="dataset/5G-weekend-1d.csv")
    parser.add_argument("--output_path", type=str, default="dataset/5G-2y-firstcell-synth.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_date", type=str, default="2024-01-01")
    parser.add_argument("--num_days", type=int, default=730)
    parser.add_argument("--first_cell_policy", type=str, default="lexicographic")
    parser.add_argument("--noise_level", type=float, default=0.08)
    parser.add_argument("--drift_strength", type=float, default=0.06)
    parser.add_argument("--station_limit", type=int, default=0, help="Optional limit for generated stations, 0 means all.")
    return parser.parse_args()


def normalize_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    normalized = []
    for header in headers:
        count = seen.get(header, 0)
        if count == 0:
            normalized.append(header)
        else:
            normalized.append(f"{header}.{count}")
        seen[header] = count + 1
    return normalized


def parse_timestamp_slot(timestamp: str) -> int:
    hour_str, minute_str = timestamp.split(":")
    return int(hour_str) * 2 + int(minute_str) // 30


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: List[float], mean_value: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def clip(value: float, minimum: float) -> float:
    return value if value >= minimum else minimum


def read_profiles(path: str) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        headers = normalize_headers(raw_headers)
        for row in reader:
            if not row:
                continue
            record = dict(zip(headers, row))
            grouped[record["Base Station ID"]][record["Cell ID"]].append(record)
    return grouped


def build_daily_profile(rows: List[Dict[str, str]]) -> DailyProfile:
    ordered_rows = sorted(rows, key=lambda row: parse_timestamp_slot(row["Timestamp"]))
    timestamps = [row["Timestamp"] for row in ordered_rows]
    metrics: Dict[str, List[float]] = {}
    stats: Dict[str, Tuple[float, float]] = {}
    zero_like_columns: Dict[str, bool] = {}

    for column in INPUT_METRIC_COLUMNS:
        values = [float(row.get(column, "0") or 0.0) for row in ordered_rows]
        mean_value = safe_mean(values)
        std_value = safe_std(values, mean_value)
        metrics[column] = values
        stats[column] = (mean_value, std_value)
        zero_like_columns[column] = all(abs(value) < 1e-12 for value in values)

    return DailyProfile(
        timestamps=timestamps,
        metrics=metrics,
        stats=stats,
        zero_like_columns=zero_like_columns,
    )


def select_first_cell_profiles(
    weekday_profiles: Dict[str, Dict[str, List[Dict[str, str]]]],
    weekend_profiles: Dict[str, Dict[str, List[Dict[str, str]]]],
    first_cell_policy: str,
    station_limit: int,
) -> Dict[str, Dict[str, object]]:
    if first_cell_policy != "lexicographic":
        raise ValueError("Only first_cell_policy=lexicographic is currently supported.")

    stations = sorted(set(weekday_profiles) & set(weekend_profiles))
    if station_limit > 0:
        stations = stations[:station_limit]

    selected: Dict[str, Dict[str, object]] = {}
    for station_id in stations:
        shared_cells = sorted(set(weekday_profiles[station_id]) & set(weekend_profiles[station_id]))
        if not shared_cells:
            continue
        first_cell = shared_cells[0]
        weekday_profile = build_daily_profile(weekday_profiles[station_id][first_cell])
        weekend_profile = build_daily_profile(weekend_profiles[station_id][first_cell])
        if len(weekday_profile.timestamps) != 48 or len(weekend_profile.timestamps) != 48:
            raise ValueError(f"Expected 48 half-hour slots for station {station_id}, cell {first_cell}.")
        selected[station_id] = {
            "district": first_cell,
            "weekday": weekday_profile,
            "weekend": weekend_profile,
        }
    return selected


def synthesize_metric_value(
    column: str,
    base_value: float,
    activity_factor: float,
    day_noise_factor: float,
    slot_std: float,
    noise_level: float,
    rng: random.Random,
) -> float:
    if column in TRAFFIC_COLUMNS:
        slot_noise = rng.gauss(0.0, slot_std * noise_level * 0.15)
        value = base_value * activity_factor * day_noise_factor + slot_noise
        return clip(value, 0.0)

    if column in POWER_COLUMNS:
        additive_scale = max(slot_std, max(base_value * 0.01, 0.1))
        value = base_value * (1.0 + 0.18 * (activity_factor - 1.0)) + rng.gauss(0.0, additive_scale * noise_level * 0.35)
        return clip(value, 0.0)

    slot_noise = rng.gauss(0.0, slot_std * noise_level * 0.1)
    value = base_value * (1.0 + 0.08 * (activity_factor - 1.0)) + slot_noise
    return clip(value, 0.0)


def synthesize_day_rows(
    station_id: str,
    district: str,
    day: datetime,
    profile: DailyProfile,
    rng: random.Random,
    noise_level: float,
    drift_state: float,
    annual_phase: float,
) -> Iterable[List[object]]:
    day_of_year = day.timetuple().tm_yday
    annual_factor = 1.0 + 0.04 * math.sin((2.0 * math.pi * day_of_year / 365.25) + annual_phase)
    activity_factor = clip(drift_state * annual_factor * math.exp(rng.gauss(0.0, noise_level * 0.22)), 0.05)

    metric_day_noise: Dict[str, float] = {}
    for column in INPUT_METRIC_COLUMNS:
        if profile.zero_like_columns[column]:
            metric_day_noise[column] = 1.0
            continue
        if column == "Traffic Volume (KByte)":
            sigma = noise_level * 0.10
        elif column == "Number of Users":
            sigma = noise_level * 0.08
        elif column == "PRB Usage Ratio (%)":
            sigma = noise_level * 0.07
        elif column == "RRU Energy (W)":
            sigma = noise_level * 0.04
        elif column == "BBU Energy (W)":
            sigma = noise_level * 0.02
        else:
            sigma = noise_level * 0.01
        metric_day_noise[column] = math.exp(rng.gauss(0.0, sigma))

    for slot, timestamp_label in enumerate(profile.timestamps):
        current_time = day + timedelta(minutes=30 * slot)
        row = [
            station_id,
            district,
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            timestamp_label,
        ]
        for column in INPUT_METRIC_COLUMNS:
            base_value = profile.metrics[column][slot]
            _, slot_std = profile.stats[column]
            if profile.zero_like_columns[column]:
                synthesized = 0.0
            else:
                synthesized = synthesize_metric_value(
                    column=column,
                    base_value=base_value,
                    activity_factor=activity_factor,
                    day_noise_factor=metric_day_noise[column],
                    slot_std=slot_std,
                    noise_level=noise_level,
                    rng=rng,
                )
            row.append(synthesized)
        yield row


def generate_dataset(args: argparse.Namespace) -> None:
    weekday_raw = read_profiles(args.weekday_path)
    weekend_raw = read_profiles(args.weekend_path)
    selected_profiles = select_first_cell_profiles(
        weekday_profiles=weekday_raw,
        weekend_profiles=weekend_raw,
        first_cell_policy=args.first_cell_policy,
        station_limit=args.station_limit,
    )

    start_day = datetime.strptime(args.start_date, "%Y-%m-%d")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    station_states: Dict[str, Dict[str, float]] = {}
    for station_id in selected_profiles:
        station_states[station_id] = {
            "drift": 1.0 + rng.gauss(0.0, args.drift_strength * 0.15),
            "annual_phase": rng.uniform(0.0, 2.0 * math.pi),
        }

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        for station_index, station_id in enumerate(sorted(selected_profiles), start=1):
            station_profile = selected_profiles[station_id]
            district = station_profile["district"]
            weekday_profile = station_profile["weekday"]
            weekend_profile = station_profile["weekend"]
            station_rng = random.Random((args.seed + 1) * 1_000_003 + station_index)

            drift_state = station_states[station_id]["drift"]
            annual_phase = station_states[station_id]["annual_phase"]

            for day_offset in range(args.num_days):
                day = start_day + timedelta(days=day_offset)
                daily_drift_step = station_rng.gauss(0.0, args.drift_strength / 45.0)
                drift_state = clip(0.985 * drift_state + 0.015 * 1.0 + daily_drift_step, 0.25)
                drift_state = min(drift_state, 4.0)

                profile = weekday_profile if day.weekday() < 5 else weekend_profile
                for row in synthesize_day_rows(
                    station_id=station_id,
                    district=district,
                    day=day,
                    profile=profile,
                    rng=station_rng,
                    noise_level=args.noise_level,
                    drift_state=drift_state,
                    annual_phase=annual_phase,
                ):
                    writer.writerow(row)

    total_rows = len(selected_profiles) * args.num_days * 48
    print(f"Generated synthetic dataset for {len(selected_profiles)} stations.")
    print(f"Output path: {output_path}")
    print(f"Total rows written: {total_rows}")


def main() -> None:
    args = parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
