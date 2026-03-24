from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

params = {
"WLSACCESSID": '402bd59f-47b2-40f2-915d-8822c1886180',
"WLSSECRET": '325a9d0e-64ee-4d62-b685-a9895e5c4351',
"LICENSEID": 2715331,
}
env = gp.Env(params=params)


@dataclass(frozen=True)
class BatterySpec:
	e_max: float
	e_min: float
	p_ch_max: float
	p_dis_max: float
	eta_ch: float
	eta_dis: float
	e0: float


def _as_2d_loads(loads: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> tuple[np.ndarray, list[str]]:
	if isinstance(loads, pd.DataFrame):
		station_ids = [str(col) for col in loads.columns]
		values = loads.to_numpy(dtype=float).T
		return values, station_ids

	values = np.asarray(loads, dtype=float)
	if values.ndim != 2:
		raise ValueError("`loads` must be 2D with shape (n_stations, n_slots) or a DataFrame with stations as columns.")
	station_ids = [f"station_{idx}" for idx in range(values.shape[0])]
	return values, station_ids


def _as_price_array(prices: pd.Series | np.ndarray | Sequence[float], n_slots: int) -> np.ndarray:
	values = np.asarray(prices, dtype=float).reshape(-1)
	if values.shape[0] != n_slots:
		raise ValueError(f"`prices` length ({values.shape[0]}) must match number of time slots ({n_slots}).")
	return values


def _extract_price_series(
	prices_df: pd.DataFrame,
	price_col: str | None,
	price_day: str | None = None,
	datetime_col: str = "Datetime (Local)",
) -> np.ndarray:
	filtered_df = prices_df
	if price_day is not None:
		if datetime_col not in prices_df.columns:
			raise ValueError(f"--price_day requires '{datetime_col}' column in prices CSV.")

		local_ts = pd.to_datetime(prices_df[datetime_col], errors="coerce")
		if local_ts.isna().all():
			raise ValueError(f"Could not parse timestamps from '{datetime_col}' column.")

		try:
			selected_day = pd.to_datetime(price_day).date()
		except Exception as exc:
			raise ValueError(f"Invalid --price_day '{price_day}'. Use YYYY-MM-DD.") from exc

		mask = local_ts.dt.date == selected_day
		filtered_df = prices_df.loc[mask]
		if filtered_df.empty:
			raise ValueError(f"No price rows found for local day {selected_day} in '{datetime_col}'.")

	if price_col is not None:
		if price_col not in filtered_df.columns:
			raise ValueError(f"Price column '{price_col}' not found in prices CSV.")
		return filtered_df[price_col].to_numpy(dtype=float)

	if filtered_df.shape[1] == 1:
		return filtered_df.iloc[:, 0].to_numpy(dtype=float)

	if "Price (EUR/MWhe)" in filtered_df.columns:
		return filtered_df["Price (EUR/MWhe)"].to_numpy(dtype=float)

	price_like_cols = [col for col in filtered_df.columns if "price" in str(col).lower()]
	if len(price_like_cols) == 1:
		return filtered_df[price_like_cols[0]].to_numpy(dtype=float)

	raise ValueError(
		"prices_csv has multiple columns. Specify --price_col, or provide a file with a single price column."
	)


def _prepare_prices_for_slots(prices: np.ndarray, n_slots: int, slot_hours: float) -> np.ndarray:
	prices = np.asarray(prices, dtype=float).reshape(-1)
	if prices.shape[0] == n_slots:
		return prices

	if slot_hours <= 0:
		raise ValueError("`slot_hours` must be positive.")

	slots_per_hour_float = 1.0 / slot_hours
	slots_per_hour = int(round(slots_per_hour_float))
	if not np.isclose(slots_per_hour_float, slots_per_hour):
		raise ValueError(
			"Unsupported slot_hours for hourly-price expansion. Use a divisor of 1 hour (e.g., 1.0, 0.5, 0.25)."
		)

	if prices.shape[0] * slots_per_hour == n_slots:
		return np.repeat(prices, slots_per_hour)

	raise ValueError(
		f"Unable to align price series length ({prices.shape[0]}) to load slots ({n_slots}) with slot_hours={slot_hours}. "
		"Provide prices already matching load slots, or hourly prices compatible with the selected slot_hours."
	)


def _align_specs(
	specs: Sequence[BatterySpec] | Mapping[str, BatterySpec],
	station_ids: list[str],
) -> list[BatterySpec]:
	if isinstance(specs, Mapping):
		missing = [station_id for station_id in station_ids if station_id not in specs]
		if missing:
			raise ValueError(f"Missing battery specs for stations: {missing}")
		ordered = [specs[station_id] for station_id in station_ids]
	else:
		ordered = list(specs)
		if len(ordered) != len(station_ids):
			raise ValueError(
				f"`specs` size ({len(ordered)}) must equal number of stations ({len(station_ids)})."
			)

	for idx, spec in enumerate(ordered):
		if not (0 < spec.eta_ch <= 1 and 0 < spec.eta_dis <= 1):
			raise ValueError(f"Invalid efficiencies for station {station_ids[idx]}: eta_ch and eta_dis must be in (0, 1].")
		if not (spec.e_min <= spec.e0 <= spec.e_max):
			raise ValueError(f"Invalid e0 bounds for station {station_ids[idx]}: expected e_min <= e0 <= e_max.")
		if spec.e_min < 0 or spec.e_max <= 0:
			raise ValueError(f"Invalid energy bounds for station {station_ids[idx]}: e_min >= 0 and e_max > 0 required.")
		if spec.p_ch_max < 0 or spec.p_dis_max < 0:
			raise ValueError(f"Invalid power bounds for station {station_ids[idx]}: max powers must be non-negative.")

	return ordered


def optimize_battery_schedule(
	loads: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
	prices: pd.Series | np.ndarray | Sequence[float],
	specs: Sequence[BatterySpec] | Mapping[str, BatterySpec],
	slot_hours: float = 0.5,
	mip_gap: float | None = None,
	time_limit_sec: float | None = None,
	verbose: bool = False,
) -> dict[str, float | str | pd.DataFrame]:

	if slot_hours <= 0:
		raise ValueError("`slot_hours` must be positive.")

	loads_arr, station_ids = _as_2d_loads(loads)
	n_stations, n_slots = loads_arr.shape
	prices_arr = _as_price_array(prices, n_slots)
	specs_arr = _align_specs(specs, station_ids)

	model = gp.Model("battery_arbitrage", env=env)
	if not verbose:
		model.Params.OutputFlag = 0
	if mip_gap is not None:
		model.Params.MIPGap = float(mip_gap)
	if time_limit_sec is not None:
		model.Params.TimeLimit = float(time_limit_sec)

	g = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="g")
	c = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
	d = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="d")
	E = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, name="E")
	u = model.addVars(n_stations, n_slots, vtype=GRB.BINARY, name="u")

	for i in range(n_stations):
		spec = specs_arr[i]
		for t in range(n_slots):
			E[i, t].LB = spec.e_min
			E[i, t].UB = spec.e_max

	model.setObjective(
		gp.quicksum(prices_arr[t] * g[i, t] for i in range(n_stations) for t in range(n_slots)),
		GRB.MINIMIZE,
	)

	for i in range(n_stations):
		spec = specs_arr[i]
		max_charge_energy = spec.p_ch_max * slot_hours
		max_discharge_energy = spec.p_dis_max * slot_hours

		for t in range(n_slots):
			model.addConstr(g[i, t] + d[i, t] == loads_arr[i, t] + c[i, t], name=f"balance[{i},{t}]")

			if t == 0:
				model.addConstr(
					E[i, t] == spec.e0 + spec.eta_ch * c[i, t] - d[i, t] / spec.eta_dis,
					name=f"soc_dyn[{i},{t}]",
				)
			else:
				model.addConstr(
					E[i, t] == E[i, t - 1] + spec.eta_ch * c[i, t] - d[i, t] / spec.eta_dis,
					name=f"soc_dyn[{i},{t}]",
				)

			model.addConstr(c[i, t] <= max_charge_energy, name=f"charge_lim[{i},{t}]")
			model.addConstr(d[i, t] <= max_discharge_energy, name=f"discharge_lim[{i},{t}]")
			model.addConstr(c[i, t] <= u[i, t] * max_charge_energy, name=f"mode_charge[{i},{t}]")
			model.addConstr(d[i, t] <= (1 - u[i, t]) * max_discharge_energy, name=f"mode_discharge[{i},{t}]")

		model.addConstr(E[i, n_slots - 1] == spec.e0, name=f"terminal_soc[{i}]")

	model.optimize()

	if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
		if model.status == GRB.INFEASIBLE:
			raise RuntimeError("Optimization infeasible. Check battery specs, load profile, and SoC constraints.")
		raise RuntimeError(f"Optimization failed with Gurobi status code: {model.status}")

	records: list[dict[str, float | str | int]] = []
	for i, station_id in enumerate(station_ids):
		for t in range(n_slots):
			grid_energy = g[i, t].X
			charge = c[i, t].X
			discharge = d[i, t].X
			soc = E[i, t].X
			mode = int(round(u[i, t].X))
			records.append(
				{
					"station": station_id,
					"t": t,
					"price": prices_arr[t],
					"load": loads_arr[i, t],
					"grid_energy": grid_energy,
					"charge": charge,
					"discharge": discharge,
					"soc_end": soc,
					"mode": mode,
					"slot_cost": prices_arr[t] * grid_energy,
				}
			)

	schedule = pd.DataFrame.from_records(records)
	station_summary = (
		schedule.groupby("station", as_index=False)
		.agg(total_grid_energy=("grid_energy", "sum"), total_cost=("slot_cost", "sum"))
		.sort_values("station")
		.reset_index(drop=True)
	)

	return {
		"status": model.Status,
		"objective_value": float(model.ObjVal),
		"mip_gap": float(model.MIPGap) if model.SolCount > 0 else np.nan,
		"schedule": schedule,
		"station_summary": station_summary,
	}


def _demo_data(n_stations: int = 3, n_slots: int = 48) -> tuple[pd.DataFrame, np.ndarray, list[BatterySpec]]:
	rng = np.random.default_rng(7)

	base_load = 15 + 5 * np.sin(np.linspace(0, 2 * np.pi, n_slots, endpoint=False))
	loads = []
	for i in range(n_stations):
		loads.append(np.maximum(2.0, base_load + rng.normal(0, 1.2, size=n_slots) + i))
	loads_df = pd.DataFrame(np.vstack(loads).T, columns=[f"bs_{i}" for i in range(n_stations)])

	hourly_prices = np.array([0.10] * 7 + [0.18] * 5 + [0.30] * 6 + [0.16] * 6, dtype=float)
	if n_slots % hourly_prices.shape[0] != 0:
		raise ValueError("Demo `n_slots` must be a multiple of 24.")
	prices = np.repeat(hourly_prices, n_slots // hourly_prices.shape[0])

	specs = [
		BatterySpec(e_max=45.0, e_min=8.0, p_ch_max=10.0, p_dis_max=10.0, eta_ch=0.95, eta_dis=0.95, e0=20.0)
		for _ in range(n_stations)
	]
	return loads_df, prices, specs


def main(argv: Iterable[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Battery charging/discharging optimization with Gurobi.")
	parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data.")
	parser.add_argument(
		"--loads_csv",
		type=str,
		default=None,
		help="Path to CSV containing per-station load (columns=stations, rows=time slots).",
	)
	parser.add_argument(
		"--prices_csv",
		type=str,
		default=None,
		help="Path to CSV containing price data. Supports direct slot-level prices or hourly prices (auto-expanded).",
	)
	parser.add_argument(
		"--price_col",
		type=str,
		default=None,
		help="Price column name when using --prices_csv (optional for common formats like 'Price (EUR/MWhe)').",
	)
	parser.add_argument(
		"--price_day",
		type=str,
		default="2016-06-30",
		help="Local day to select from multi-day price CSV, using 'Datetime (Local)' (format: YYYY-MM-DD).",
	)
	parser.add_argument(
		"--slot_hours",
		type=float,
		default=0.5,
		help="Duration of each prediction/load slot in hours (default: 0.5 for half-hour granularity).",
	)
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args(list(argv) if argv is not None else None)

	if args.demo:
		loads_df, prices_arr, specs_arr = _demo_data()
	else:
		if args.loads_csv is None or args.prices_csv is None:
			raise ValueError("Provide --demo or both --loads_csv and --prices_csv.")

		loads_df = pd.read_csv(args.loads_csv)
		prices_df = pd.read_csv(args.prices_csv)
		raw_prices = _extract_price_series(prices_df, args.price_col, price_day=args.price_day)
		prices_arr = _prepare_prices_for_slots(raw_prices, n_slots=len(loads_df), slot_hours=args.slot_hours)

		specs_arr = [
			BatterySpec(e_max=45.0, e_min=8.0, p_ch_max=10.0, p_dis_max=10.0, eta_ch=0.95, eta_dis=0.95, e0=20.0)
			for _ in loads_df.columns
		]

	result = optimize_battery_schedule(
		loads=loads_df,
		prices=prices_arr,
		specs=specs_arr,
		slot_hours=args.slot_hours,
		verbose=args.verbose,
	)

	print(f"Optimization status: {result['status']}")
	print(f"Objective value (total electricity cost): {result['objective_value']:.6f}")
	print("\nPer-station summary:")
	print(result["station_summary"].to_string(index=False))


if __name__ == "__main__":
	main()
