from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def _find_prediction_files(input_glob: str) -> list[Path]:
	files = sorted(Path().glob(input_glob))
	if not files:
		raise FileNotFoundError(
			f"No prediction files matched pattern: {input_glob}. "
			"Example: './saved_predictions/*_last_*.csv'"
		)
	return files


def _infer_station_from_filename(file_path: Path, station_regex: str | None) -> str:
	stem = file_path.stem
	if station_regex is None:
		# Default for files like: <cid>_<model_name>_last_<num_lags>.csv
		match = re.match(r"(.+?)_[^_]+_last_\d+$", stem)
		if match:
			return match.group(1)
		# Fallback: take whole stem if pattern does not match
		return stem

	match = re.search(station_regex, stem)
	if not match:
		raise ValueError(
			f"Could not extract station id from filename '{file_path.name}' using regex '{station_regex}'."
		)
	if match.groups():
		return match.group(1)
	return match.group(0)


def _select_predicted_column(df: pd.DataFrame, pred_col: str | None) -> str:
	if pred_col is not None:
		if pred_col not in df.columns:
			raise ValueError(f"Requested predicted column '{pred_col}' not found. Available: {list(df.columns)}")
		return pred_col

	pred_cols = [col for col in df.columns if str(col).startswith("pred_")]
	if not pred_cols:
		raise ValueError(
			"No predicted columns found. Expected columns like 'pred_<target_name>' in client prediction CSV."
		)
	if len(pred_cols) > 1:
		raise ValueError(
			f"Multiple predicted columns found: {pred_cols}. "
			"Specify one with --pred_col."
		)
	return pred_cols[0]


def merge_prediction_files(
	input_glob: str,
	output_csv: str,
	pred_col: str | None,
	station_regex: str | None,
	time_col: str,
	allow_time_union: bool,
) -> Path:
	files = _find_prediction_files(input_glob)

	merged_df: pd.DataFrame | None = None

	for file_path in files:
		df = pd.read_csv(file_path)
		if time_col not in df.columns:
			raise ValueError(f"File '{file_path}' does not contain required time column '{time_col}'.")

		column_name = _select_predicted_column(df, pred_col)
		station_id = _infer_station_from_filename(file_path, station_regex)

		station_df = df[[time_col, column_name]].copy()
		station_df = station_df.rename(columns={column_name: station_id})

		if merged_df is None:
			merged_df = station_df
		else:
			how = "outer" if allow_time_union else "inner"
			merged_df = merged_df.merge(station_df, on=time_col, how=how)

	if merged_df is None:
		raise RuntimeError("No prediction files were processed.")

	merged_df = merged_df.sort_values(time_col).reset_index(drop=True)

	# batterycase expects rows=time slots and columns=stations
	loads_df = merged_df.drop(columns=[time_col])

	if loads_df.empty:
		raise RuntimeError("Merged output has no station columns.")

	if loads_df.isna().any().any():
		raise ValueError(
			"Merged load data contains missing values after alignment. "
			"Use matching horizons across clients or run with --allow_time_union and post-process NaNs."
		)

	output_path = Path(output_csv).expanduser()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	loads_df.to_csv(output_path, index=False)

	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Merge per-client prediction CSV files into one batterycase-compatible loads CSV."
	)
	parser.add_argument(
		"--input_glob",
		type=str,
		default="./saved_predictions/*_last_*.csv",
		help="Glob pattern for client prediction CSVs.",
	)
	parser.add_argument(
		"--output_csv",
		type=str,
		default="./merged_predicted_loads.csv",
		help="Path for merged loads CSV output.",
	)
	parser.add_argument(
		"--pred_col",
		type=str,
		default=None,
		help="Predicted column to merge (e.g., 'pred_Traffic Volume (KByte)').",
	)
	parser.add_argument(
		"--station_regex",
		type=str,
		default=None,
		help=(
			"Optional regex to extract station id from filename stem. "
			"If it has a capture group, group(1) is used."
		),
	)
	parser.add_argument(
		"--time_col",
		type=str,
		default="time",
		help="Time column name in client prediction CSVs.",
	)
	parser.add_argument(
		"--allow_time_union",
		action="store_true",
		help="Use outer-join on timestamps across clients (otherwise inner-join).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_path = merge_prediction_files(
		input_glob=args.input_glob,
		output_csv=args.output_csv,
		pred_col=args.pred_col,
		station_regex=args.station_regex,
		time_col=args.time_col,
		allow_time_union=args.allow_time_union,
	)
	print(f"Merged predicted loads saved to: {output_path}")


if __name__ == "__main__":
	main()

