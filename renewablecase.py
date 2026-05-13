from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

params = {
    "WLSACCESSID": "1c23747a-c60d-4a23-aad9-4f1ad990da0c",
    "WLSSECRET": "251bbc74-0263-4665-b382-3c329957c321",
    "LICENSEID": 2715331,
}
env = gp.Env(params=params)


PREDICTION_METHOD_ALIASES = {
	"plain": "plain_heterofl",
	"plain_heterofl": "plain_heterofl",
	"pwrh": "pwrh",
	"spa": "spa_hfl",
	"spa_hfl": "spa_hfl",
}


@dataclass(frozen=True)
class BatterySpec:
	e_max: float
	e_min: float
	p_ch_max: float
	p_dis_max: float
	eta_ch: float
	eta_dis: float
	e0: float


def _as_2d_values(values: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], name: str) -> tuple[np.ndarray, list[str]]:
	if isinstance(values, pd.DataFrame):
		station_ids = [str(col) for col in values.columns]
		arr = values.to_numpy(dtype=float).T
		return arr, station_ids

	arr = np.asarray(values, dtype=float)
	if arr.ndim != 2:
		raise ValueError(f"`{name}` must be 2D with shape (n_stations, n_slots) or DataFrame with stations as columns.")
	station_ids = [f"station_{idx}" for idx in range(arr.shape[0])]
	return arr, station_ids


def _as_1d(values: pd.Series | np.ndarray | Sequence[float], n_slots: int, name: str) -> np.ndarray:
	arr = np.asarray(values, dtype=float).reshape(-1)
	if arr.shape[0] != n_slots:
		raise ValueError(f"`{name}` length ({arr.shape[0]}) must match number of time slots ({n_slots}).")
	return arr


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


def _filter_df_by_local_day(
	df: pd.DataFrame,
	day: str | None,
	datetime_col: str,
	arg_name: str,
) -> pd.DataFrame:
	if day is None:
		if datetime_col in df.columns:
			ts = pd.to_datetime(df[datetime_col], errors="coerce")
			unique_days = ts.dt.date.dropna().nunique()
			if unique_days > 1:
				raise ValueError(
					f"Input has multiple days in '{datetime_col}'. Specify --{arg_name} YYYY-MM-DD."
				)
		return df

	if datetime_col not in df.columns:
		raise ValueError(f"--{arg_name} requires '{datetime_col}' column in CSV.")

	ts = pd.to_datetime(df[datetime_col], errors="coerce")
	if ts.isna().all():
		raise ValueError(f"Could not parse timestamps from '{datetime_col}' column.")

	try:
		selected_day = pd.to_datetime(day).date()
	except Exception as exc:
		raise ValueError(f"Invalid --{arg_name} '{day}'. Use YYYY-MM-DD.") from exc

	filtered = df.loc[ts.dt.date == selected_day].copy()
	if filtered.empty:
		raise ValueError(f"No rows found for local day {selected_day} in '{datetime_col}'.")

	return filtered


def _prepare_renewables_for_loads(
	renewables_df: pd.DataFrame,
	load_columns: Sequence[str],
	n_slots: int,
	slot_hours: float,
	renewable_col: str,
	renewable_day: str | None,
	renewable_time_col: str,
) -> pd.DataFrame:
	filtered_df = _filter_df_by_local_day(
		df=renewables_df,
		day=renewable_day,
		datetime_col=renewable_time_col,
		arg_name="renewable_day",
	)

	if renewable_col not in filtered_df.columns:
		raise ValueError(
			f"Renewable column '{renewable_col}' not found in renewables_csv."
		)

	raw_renewables = filtered_df[renewable_col].to_numpy(dtype=float)
	ren_series = _prepare_prices_for_slots(raw_renewables, n_slots=n_slots, slot_hours=slot_hours)

	# Same renewable availability is used for every station at each time step.
	return pd.DataFrame({col: ren_series for col in load_columns})


def _canonical_prediction_method(method: str) -> str:
	key = method.strip().lower()
	if key not in PREDICTION_METHOD_ALIASES:
		raise ValueError(
			f"Unknown prediction method '{method}'. Supported methods: {sorted(PREDICTION_METHOD_ALIASES)}"
		)
	return PREDICTION_METHOD_ALIASES[key]


def _extract_station_id_from_prediction_file(path: Path, method: str, model_name: str) -> str:
	prefix = f"{method}_"
	suffix = f"_{model_name}_last_"
	name = path.name
	if not name.startswith(prefix) or suffix not in name:
		raise ValueError(f"Could not parse station id from prediction filename: {path.name}")
	return name[len(prefix): name.index(suffix)]


def _prediction_step_columns(df: pd.DataFrame, target: str, value_prefix: str) -> list[tuple[int, str]]:
	pattern = re.compile(rf"^{re.escape(value_prefix)}_{re.escape(target)}_step\+(\d+)$")
	columns: list[tuple[int, str]] = []
	for col in df.columns:
		match = pattern.match(str(col))
		if match:
			columns.append((int(match.group(1)), str(col)))
	columns.sort(key=lambda item: item[0])
	if not columns:
		raise ValueError(f"No '{value_prefix}_{target}_step+N' columns found in prediction CSV.")
	return columns


def _sample_prediction_csv_loads(
	path: Path,
	target: str,
	sample_every: int,
	n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
	if sample_every <= 0:
		raise ValueError("--sample_every must be positive.")
	if n_points <= 0:
		raise ValueError("--n_points must be positive.")

	df = pd.read_csv(path)
	pred_cols = _prediction_step_columns(df, target, "pred")
	true_cols = _prediction_step_columns(df, target, "true")
	if [step for step, _ in pred_cols] != [step for step, _ in true_cols]:
		raise ValueError(f"Prediction and ground-truth horizons do not match in {path}.")

	pred_values: list[float] = []
	true_values: list[float] = []
	for row_idx in range(0, len(df), sample_every):
		row = df.iloc[row_idx]
		for (_, pred_col), (_, true_col) in zip(pred_cols, true_cols):
			pred_values.append(float(row[pred_col]))
			true_values.append(float(row[true_col]))
			if len(pred_values) == n_points:
				return np.asarray(pred_values, dtype=float), np.asarray(true_values, dtype=float)

	raise ValueError(
		f"Could only sample {len(pred_values)} points from {path}; need {n_points}. "
		"Use a smaller --sample_every or --n_points."
	)


def load_prediction_loads(
	predictions_dir: str | Path,
	method: str,
	target: str = "BBU Energy (W)",
	sample_every: int = 4,
	n_points: int = 48,
	model_name: str = "lstm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
	predictions_path = Path(predictions_dir)
	canonical_method = _canonical_prediction_method(method)
	files = sorted(predictions_path.glob(f"{canonical_method}_*_{model_name}_last_*.csv"))
	if not files:
		raise FileNotFoundError(
			f"No prediction files found for method '{canonical_method}' in {predictions_path}. "
			f"Expected files like '{canonical_method}_<station>_{model_name}_last_48.csv'."
		)

	pred_columns: dict[str, np.ndarray] = {}
	true_columns: dict[str, np.ndarray] = {}
	for path in files:
		station_id = _extract_station_id_from_prediction_file(path, canonical_method, model_name)
		pred_load, true_load = _sample_prediction_csv_loads(
			path=path,
			target=target,
			sample_every=sample_every,
			n_points=n_points,
		)
		pred_columns[station_id] = pred_load
		true_columns[station_id] = true_load

	station_ids = sorted(pred_columns)
	pred_loads = pd.DataFrame({station_id: pred_columns[station_id] for station_id in station_ids})
	true_loads = pd.DataFrame({station_id: true_columns[station_id] for station_id in station_ids})
	return pred_loads, true_loads


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
			raise ValueError(f"`specs` size ({len(ordered)}) must equal number of stations ({len(station_ids)}).")

	for idx, spec in enumerate(ordered):
		if not (0 < spec.eta_ch <= 1 and 0 < spec.eta_dis <= 1):
			raise ValueError(f"Invalid efficiencies for station {station_ids[idx]}: eta_ch and eta_dis must be in (0,1].")
		if spec.e_min < 0 or spec.e_max <= 0 or spec.e_min > spec.e_max:
			raise ValueError(f"Invalid energy bounds for station {station_ids[idx]}.")
		if not (spec.e_min <= spec.e0 <= spec.e_max):
			raise ValueError(f"Invalid initial energy for station {station_ids[idx]}: require e_min <= e0 <= e_max.")
		if spec.p_ch_max < 0 or spec.p_dis_max < 0:
			raise ValueError(f"Invalid power bounds for station {station_ids[idx]}: max powers must be non-negative.")

	return ordered


def optimize_renewable_battery_schedule(
	loads: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
	renewables: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
	prices: pd.Series | np.ndarray | Sequence[float],
	specs: Sequence[BatterySpec] | Mapping[str, BatterySpec],
	curtailment_cost: float,
	slot_hours: float = 0.5,
	mip_gap: float | None = None,
	time_limit_sec: float | None = None,
	verbose: bool = False,
) -> dict[str, float | int | pd.DataFrame]:
	"""
	Solve the renewable-aware battery scheduling MILP.

	Decision variables per station i and time t:
	- r_u[i,t]: renewable used directly to serve load
	- r_c[i,t]: renewable used to charge battery
	- g[i,t]: grid purchase
	- c_g[i,t]: battery charging from grid
	- d[i,t]: battery discharging
	- E[i,t]: battery SoC at end of slot
	- u[i,t]: binary mode (1=charging mode, 0=discharging mode)
	- s[i,t]: curtailed renewable (linearization for max(0, R-r_u-r_c))
	"""

	if slot_hours <= 0:
		raise ValueError("`slot_hours` must be positive.")
	if curtailment_cost < 0:
		raise ValueError("`curtailment_cost` must be non-negative.")

	loads_arr, station_ids = _as_2d_values(loads, "loads")
	ren_arr, ren_station_ids = _as_2d_values(renewables, "renewables")
	if loads_arr.shape != ren_arr.shape:
		raise ValueError(
			f"`loads` shape {loads_arr.shape} must match `renewables` shape {ren_arr.shape}."
		)
	if ren_station_ids != station_ids:
		raise ValueError("When DataFrames are used, `loads` and `renewables` columns must match in the same order.")

	n_stations, n_slots = loads_arr.shape
	prices_arr = _as_1d(prices, n_slots, "prices")
	specs_arr = _align_specs(specs, station_ids)

	model = gp.Model("renewable_battery_arbitrage", env=env)
	if not verbose:
		model.Params.OutputFlag = 0
	if mip_gap is not None:
		model.Params.MIPGap = float(mip_gap)
	if time_limit_sec is not None:
		model.Params.TimeLimit = float(time_limit_sec)

	r_u = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="r_u")
	r_c = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="r_c")
	g = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="g")
	c_g = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="c_g")
	d = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="d")
	E = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, name="E")
	u = model.addVars(n_stations, n_slots, vtype=GRB.BINARY, name="u")
	s = model.addVars(n_stations, n_slots, vtype=GRB.CONTINUOUS, lb=0.0, name="curtail")

	for i in range(n_stations):
		spec = specs_arr[i]
		for t in range(n_slots):
			E[i, t].LB = spec.e_min
			E[i, t].UB = spec.e_max

	model.setObjective(
		gp.quicksum(
			prices_arr[t] * g[i, t] + curtailment_cost * s[i, t]
			for i in range(n_stations)
			for t in range(n_slots)
		),
		GRB.MINIMIZE,
	)

	for i in range(n_stations):
		spec = specs_arr[i]
		max_charge_energy = spec.p_ch_max * slot_hours
		max_discharge_energy = spec.p_dis_max * slot_hours

		for t in range(n_slots):
			model.addConstr(
				g[i, t] + d[i, t] + r_u[i, t] == loads_arr[i, t] + c_g[i, t] + r_c[i, t],
				name=f"balance[{i},{t}]",
			)

			if t == 0:
				model.addConstr(
					E[i, t] == spec.e0 + spec.eta_ch * (c_g[i, t] + r_c[i, t]) - d[i, t] / spec.eta_dis,
					name=f"soc_dyn[{i},{t}]",
				)
			else:
				model.addConstr(
					E[i, t] == E[i, t - 1] + spec.eta_ch * (c_g[i, t] + r_c[i, t]) - d[i, t] / spec.eta_dis,
					name=f"soc_dyn[{i},{t}]",
				)

			model.addConstr(c_g[i, t] + r_c[i, t] <= max_charge_energy, name=f"charge_lim[{i},{t}]")
			model.addConstr(d[i, t] <= max_discharge_energy, name=f"discharge_lim[{i},{t}]")
			model.addConstr(c_g[i, t] + r_c[i, t] <= u[i, t] * max_charge_energy, name=f"mode_charge[{i},{t}]")
			model.addConstr(d[i, t] <= (1 - u[i, t]) * max_discharge_energy, name=f"mode_discharge[{i},{t}]")

			model.addConstr(r_u[i, t] + r_c[i, t] <= ren_arr[i, t], name=f"ren_use_lim[{i},{t}]")
			model.addConstr(s[i, t] >= ren_arr[i, t] - r_u[i, t] - r_c[i, t], name=f"curtail_def[{i},{t}]")

		model.addConstr(E[i, n_slots - 1] == spec.e0, name=f"terminal_soc[{i}]")

	model.optimize()

	if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
		if model.status == GRB.INFEASIBLE:
			raise RuntimeError("Optimization infeasible. Check data consistency and battery constraints.")
		raise RuntimeError(f"Optimization failed with Gurobi status code: {model.status}")

	records: list[dict[str, float | str | int]] = []
	for i, station_id in enumerate(station_ids):
		for t in range(n_slots):
			grid_energy = g[i, t].X
			ren_load = r_u[i, t].X
			ren_charge = r_c[i, t].X
			grid_charge = c_g[i, t].X
			discharge = d[i, t].X
			soc = E[i, t].X
			curtailed = s[i, t].X

			records.append(
				{
					"station": station_id,
					"t": t,
					"price": prices_arr[t],
					"load": loads_arr[i, t],
					"renewable_available": ren_arr[i, t],
					"renewable_to_load": ren_load,
					"renewable_to_battery": ren_charge,
					"grid_energy": grid_energy,
					"grid_to_battery": grid_charge,
					"discharge": discharge,
					"soc_end": soc,
					"curtailed_renewable": curtailed,
					"mode": int(round(u[i, t].X)),
					"grid_cost": prices_arr[t] * grid_energy,
					"curtailment_cost": curtailment_cost * curtailed,
					"slot_cost": prices_arr[t] * grid_energy + curtailment_cost * curtailed,
				}
			)

	schedule = pd.DataFrame.from_records(records)
	station_summary = (
		schedule.groupby("station", as_index=False)
		.agg(
			total_grid_energy=("grid_energy", "sum"),
			total_grid_cost=("grid_cost", "sum"),
			total_curtailment=("curtailed_renewable", "sum"),
			total_curtailment_cost=("curtailment_cost", "sum"),
			total_cost=("slot_cost", "sum"),
		)
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


def add_groundtruth_penalty(
	optimization_result: dict[str, float | int | pd.DataFrame],
	groundtruth_loads: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
	prices: pd.Series | np.ndarray | Sequence[float],
	penalty_coef: float,
	curtailment_cost: float,
) -> dict[str, float | int | pd.DataFrame]:
	if penalty_coef < 0:
		raise ValueError("--penalty_coef must be non-negative.")
	if curtailment_cost < 0:
		raise ValueError("--curtailment_cost must be non-negative.")

	truth_arr, truth_station_ids = _as_2d_values(groundtruth_loads, "groundtruth_loads")
	schedule = optimization_result["schedule"].copy()
	if not isinstance(schedule, pd.DataFrame):
		raise TypeError("optimization_result['schedule'] must be a pandas DataFrame.")
	price_arr = _as_1d(prices, truth_arr.shape[1], "prices")

	schedule_station_ids = list(dict.fromkeys(schedule["station"].astype(str)))
	if schedule_station_ids != truth_station_ids:
		raise ValueError(
			"Ground-truth load stations do not match optimized schedule stations. "
			f"Schedule={schedule_station_ids}, groundtruth={truth_station_ids}."
		)

	for station_idx, station_id in enumerate(truth_station_ids):
		mask = schedule["station"].astype(str) == station_id
		station_slots = schedule.loc[mask, "t"].to_numpy(dtype=int)
		if len(station_slots) != truth_arr.shape[1]:
			raise ValueError(
				f"Ground-truth slot count for station {station_id} ({truth_arr.shape[1]}) "
				f"does not match schedule slot count ({len(station_slots)})."
			)
		schedule.loc[mask, "groundtruth_load"] = truth_arr[station_idx, station_slots]

	schedule["shortage_penalty_rate"] = schedule["t"].map(lambda t: penalty_coef * price_arr[int(t)])
	schedule["surplus_penalty_rate"] = float(curtailment_cost)
	schedule["diff"] = (
		schedule["grid_energy"]
		+ schedule["discharge"]
		+ schedule["renewable_to_load"]
		- (
			schedule["groundtruth_load"]
			+ schedule["grid_to_battery"]
			+ schedule["renewable_to_battery"]
		)
	)
	schedule["shortage_penalty_cost"] = np.where(
		schedule["diff"] < 0,
		-schedule["diff"] * schedule["shortage_penalty_rate"],
		0.0,
	)
	schedule["surplus_penalty_cost"] = np.where(
		schedule["diff"] > 0,
		schedule["diff"] * schedule["surplus_penalty_rate"],
		0.0,
	)
	schedule["penalty_cost"] = schedule["shortage_penalty_cost"] + schedule["surplus_penalty_cost"]
	schedule["realized_slot_cost"] = schedule["slot_cost"] + schedule["penalty_cost"]

	penalty_cost = float(schedule["penalty_cost"].sum())
	shortage_penalty_cost = float(schedule["shortage_penalty_cost"].sum())
	surplus_penalty_cost = float(schedule["surplus_penalty_cost"].sum())
	base_cost = float(optimization_result["objective_value"])
	penalized_cost = base_cost + penalty_cost
	station_summary = (
		schedule.groupby("station", as_index=False)
		.agg(
			total_grid_energy=("grid_energy", "sum"),
			total_grid_cost=("grid_cost", "sum"),
			total_curtailment=("curtailed_renewable", "sum"),
			total_curtailment_cost=("curtailment_cost", "sum"),
			total_cost=("slot_cost", "sum"),
			total_shortage_penalty_cost=("shortage_penalty_cost", "sum"),
			total_surplus_penalty_cost=("surplus_penalty_cost", "sum"),
			total_penalty_cost=("penalty_cost", "sum"),
			total_realized_cost=("realized_slot_cost", "sum"),
			min_diff=("diff", "min"),
			max_diff=("diff", "max"),
		)
		.sort_values("station")
		.reset_index(drop=True)
	)

	updated = dict(optimization_result)
	updated.update(
		{
			"base_objective_value": base_cost,
			"shortage_penalty_cost": shortage_penalty_cost,
			"surplus_penalty_cost": surplus_penalty_cost,
			"penalty_cost": penalty_cost,
			"penalized_objective_value": penalized_cost,
			"schedule": schedule,
			"station_summary": station_summary,
		}
	)
	return updated


def evaluate_prediction_method_cost(
	predicted_loads: pd.DataFrame,
	groundtruth_loads: pd.DataFrame,
	renewables: pd.DataFrame,
	prices: pd.Series | np.ndarray | Sequence[float],
	specs: Sequence[BatterySpec] | Mapping[str, BatterySpec],
	curtailment_cost: float,
	slot_hours: float = 0.5,
	penalty_coef: float = 1.5,
	mip_gap: float | None = None,
	time_limit_sec: float | None = None,
	verbose: bool = False,
) -> dict[str, float | int | pd.DataFrame]:
	result = optimize_renewable_battery_schedule(
		loads=predicted_loads,
		renewables=renewables,
		prices=prices,
		specs=specs,
		curtailment_cost=curtailment_cost,
		slot_hours=slot_hours,
		mip_gap=mip_gap,
		time_limit_sec=time_limit_sec,
		verbose=verbose,
	)
	return add_groundtruth_penalty(
		result,
		groundtruth_loads=groundtruth_loads,
		prices=prices,
		penalty_coef=penalty_coef,
		curtailment_cost=curtailment_cost,
	)


def _demo_data(n_stations: int = 3, n_slots: int = 48) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[BatterySpec]]:
	rng = np.random.default_rng(11)

	base_load = 15 + 4.5 * np.sin(np.linspace(0, 2 * np.pi, n_slots, endpoint=False) - 0.3)
	loads = []
	for i in range(n_stations):
		loads.append(np.maximum(2.0, base_load + rng.normal(0, 1.0, size=n_slots) + 0.8 * i))
	loads_df = pd.DataFrame(np.vstack(loads).T, columns=[f"bs_{i}" for i in range(n_stations)])

	solar_shape = np.maximum(0.0, np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n_slots, endpoint=False)))
	renewables = []
	for i in range(n_stations):
		renewables.append(np.maximum(0.0, 8.0 * solar_shape + rng.normal(0, 0.5, size=n_slots) + 0.5 * i))
	renewables_df = pd.DataFrame(np.vstack(renewables).T, columns=[f"bs_{i}" for i in range(n_stations)])

	hourly_prices = np.array([0.11] * 7 + [0.20] * 5 + [0.34] * 6 + [0.17] * 6, dtype=float)
	if n_slots % hourly_prices.shape[0] != 0:
		raise ValueError("Demo `n_slots` must be a multiple of 24.")
	prices = np.repeat(hourly_prices, n_slots // hourly_prices.shape[0])

	specs = [
		BatterySpec(e_max=45.0, e_min=8.0, p_ch_max=10.0, p_dis_max=10.0, eta_ch=0.95, eta_dis=0.95, e0=20.0)
		for _ in range(n_stations)
	]
	return loads_df, renewables_df, prices, specs


def main(argv: Iterable[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Renewable-aware battery scheduling optimization with Gurobi.")
	parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data.")
	parser.add_argument(
		"--predictions_dir",
		type=str,
		default=None,
		help="Directory containing saved prediction CSVs named by method, e.g. plain_heterofl_<cid>_lstm_last_48.csv.",
	)
	parser.add_argument(
		"--methods",
		nargs="+",
		default=["plain", "pwrh"],
		help="Prediction methods to evaluate from --predictions_dir. Aliases include plain and pwrh.",
	)
	parser.add_argument(
		"--prediction_target",
		type=str,
		default="BBU Energy (W)",
		help="Target column in saved prediction files to use as the load.",
	)
	parser.add_argument(
		"--prediction_model_name",
		type=str,
		default="lstm",
		help="Model-name segment used in saved prediction filenames.",
	)
	parser.add_argument(
		"--sample_every",
		type=int,
		default=4,
		help="Sample one rolling forecast row every N rows, then flatten its forecast horizons.",
	)
	parser.add_argument("--n_points", type=int, default=48, help="Number of sampled load points per station.")
	parser.add_argument("--loads_csv", type=str, default=None, help="CSV with base-station loads (columns=stations).")
	parser.add_argument(
		"--renewables_csv",
		type=str,
		default="ninja_pv_33.9647_-118.1510_uncorrected.csv",
		help="CSV with renewable energy time series (shared across all stations).",
	)
	parser.add_argument(
		"--renewable_col",
		type=str,
		default="electricity",
		help="Renewable energy column name (default: electricity).",
	)
	parser.add_argument(
		"--renewable_day",
		type=str,
		default="2019-01-01",
		help="Local day to select from renewables_csv (format: YYYY-MM-DD).",
	)
	parser.add_argument(
		"--renewable_time_col",
		type=str,
		default="local_time",
		help="Datetime column used for renewable day filtering (default: local_time).",
	)
	parser.add_argument(
		"--prices_csv",
		type=str,
		default="United Kingdom.csv",
		help="CSV containing price data. Supports direct slot-level prices or hourly prices (auto-expanded).",
	)
	parser.add_argument(
		"--price_col",
		type=str,
		default="Price (EUR/MWhe)",
		help="Price column name when using --prices_csv (optional for common formats like 'Price (EUR/MWhe)').",
	)
	parser.add_argument(
		"--price_day",
		type=str,
		default="2016-06-30",
		help="Local day to select from multi-day price CSV (format: YYYY-MM-DD).",
	)
	parser.add_argument(
		"--price_datetime_col",
		type=str,
		default="Datetime (Local)",
		help="Datetime column to use with --price_day.",
	)
	parser.add_argument("--curtailment_cost", type=float, default=50, help="Penalty per unit of curtailed renewable energy.")
	parser.add_argument(
		"--penalty_coef",
		type=float,
		default=1.5,
		help="Multiplier on slot price for shortage penalty: max(0, -diff) * penalty_coef * price[t].",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Optional directory to save prediction-based cost summaries and schedules.",
	)
	parser.add_argument(
		"--slot_hours",
		type=float,
		default=0.5,
		help="Duration of each prediction/load slot in hours (default: 0.5 for half-hour granularity).",
	)
	parser.add_argument("--mip_gap", type=float, default=None)
	parser.add_argument("--time_limit_sec", type=float, default=None)
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args(list(argv) if argv is not None else None)

	if args.demo:
		loads_df, renewables_df, prices_arr, specs_arr = _demo_data()
	elif args.predictions_dir is not None:
		renewables_raw_df = pd.read_csv(args.renewables_csv)
		prices_df = pd.read_csv(args.prices_csv)
		raw_prices = _extract_price_series(
			prices_df,
			args.price_col,
			price_day=args.price_day,
			datetime_col=args.price_datetime_col,
		)
		prices_arr = _prepare_prices_for_slots(raw_prices, n_slots=args.n_points, slot_hours=args.slot_hours)

		method_summaries: list[dict[str, float | str | int]] = []
		output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else None
		if output_dir is not None:
			output_dir.mkdir(parents=True, exist_ok=True)

		for method in args.methods:
			canonical_method = _canonical_prediction_method(method)
			predicted_loads, groundtruth_loads = load_prediction_loads(
				predictions_dir=args.predictions_dir,
				method=canonical_method,
				target=args.prediction_target,
				sample_every=args.sample_every,
				n_points=args.n_points,
				model_name=args.prediction_model_name,
			)
			renewables_df = _prepare_renewables_for_loads(
				renewables_df=renewables_raw_df,
				load_columns=list(predicted_loads.columns),
				n_slots=len(predicted_loads),
				slot_hours=args.slot_hours,
				renewable_col=args.renewable_col,
				renewable_day=args.renewable_day,
				renewable_time_col=args.renewable_time_col,
			)
			specs_arr = [
				BatterySpec(e_max=45.0, e_min=8.0, p_ch_max=10.0, p_dis_max=10.0, eta_ch=0.95, eta_dis=0.95, e0=20.0)
				for _ in predicted_loads.columns
			]
			result = evaluate_prediction_method_cost(
				predicted_loads=predicted_loads,
				groundtruth_loads=groundtruth_loads,
				renewables=renewables_df,
				prices=prices_arr,
				specs=specs_arr,
				curtailment_cost=args.curtailment_cost,
				slot_hours=args.slot_hours,
				penalty_coef=args.penalty_coef,
				mip_gap=args.mip_gap,
				time_limit_sec=args.time_limit_sec,
				verbose=args.verbose,
			)
			method_summaries.append(
				{
					"method": canonical_method,
					"status": int(result["status"]),
					"base_objective_value": float(result["base_objective_value"]),
					"shortage_penalty_cost": float(result["shortage_penalty_cost"]),
					"surplus_penalty_cost": float(result["surplus_penalty_cost"]),
					"penalty_cost": float(result["penalty_cost"]),
					"penalized_objective_value": float(result["penalized_objective_value"]),
					"mip_gap": float(result["mip_gap"]),
					"n_stations": len(predicted_loads.columns),
					"n_slots": len(predicted_loads),
				}
			)

			print(f"\nMethod: {canonical_method}")
			print(f"Optimization status: {result['status']}")
			print(f"Predicted-load optimal cost: {result['base_objective_value']:.6f}")
			print(f"Shortage penalty cost: {result['shortage_penalty_cost']:.6f}")
			print(f"Surplus penalty cost: {result['surplus_penalty_cost']:.6f}")
			print(f"Total penalty cost: {result['penalty_cost']:.6f}")
			print(f"Penalized realized cost: {result['penalized_objective_value']:.6f}")
			print("\nPer-station summary:")
			print(result["station_summary"].to_string(index=False))

			if output_dir is not None:
				predicted_loads.to_csv(output_dir / f"{canonical_method}_sampled_predicted_loads.csv", index=False)
				groundtruth_loads.to_csv(output_dir / f"{canonical_method}_sampled_groundtruth_loads.csv", index=False)
				renewables_df.to_csv(output_dir / f"{canonical_method}_sampled_renewables.csv", index=False)
				result["schedule"].to_csv(output_dir / f"{canonical_method}_renewable_schedule_with_penalty.csv", index=False)
				result["station_summary"].to_csv(output_dir / f"{canonical_method}_station_summary.csv", index=False)

		summary_df = pd.DataFrame(method_summaries)
		print("\nMethod comparison summary:")
		print(summary_df.to_string(index=False))
		if output_dir is not None:
			summary_df.to_csv(output_dir / "method_cost_summary.csv", index=False)
		return
	else:
		if args.loads_csv is None or args.renewables_csv is None or args.prices_csv is None:
			raise ValueError(
				"Provide --demo, --predictions_dir, or all of --loads_csv, --renewables_csv and --prices_csv."
			)

		loads_df = pd.read_csv(args.loads_csv)
		renewables_raw_df = pd.read_csv(args.renewables_csv)
		prices_df = pd.read_csv(args.prices_csv)
		raw_prices = _extract_price_series(
			prices_df,
			args.price_col,
			price_day=args.price_day,
			datetime_col=args.price_datetime_col,
		)
		prices_arr = _prepare_prices_for_slots(raw_prices, n_slots=len(loads_df), slot_hours=args.slot_hours)
		renewables_df = _prepare_renewables_for_loads(
			renewables_df=renewables_raw_df,
			load_columns=list(loads_df.columns),
			n_slots=len(loads_df),
			slot_hours=args.slot_hours,
			renewable_col=args.renewable_col,
			renewable_day=args.renewable_day,
			renewable_time_col=args.renewable_time_col,
		)

		specs_arr = [
			BatterySpec(e_max=45.0, e_min=8.0, p_ch_max=10.0, p_dis_max=10.0, eta_ch=0.95, eta_dis=0.95, e0=20.0)
			for _ in loads_df.columns
		]

	result = optimize_renewable_battery_schedule(
		loads=loads_df,
		renewables=renewables_df,
		prices=prices_arr,
		specs=specs_arr,
		curtailment_cost=args.curtailment_cost,
		slot_hours=args.slot_hours,
		mip_gap=args.mip_gap,
		time_limit_sec=args.time_limit_sec,
		verbose=args.verbose,
	)

	print(f"Optimization status: {result['status']}")
	print(f"Objective value (total cost): {result['objective_value']:.6f}")
	print("\nPer-station summary:")
	print(result["station_summary"].to_string(index=False))


if __name__ == "__main__":
	main()
