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


def _ensure_gurobi() -> None:
	if gp is None:
		raise ImportError(
			"gurobipy is required for renewable-aware battery optimization. Install with 'pip install gurobipy' "
			"and make sure a valid Gurobi license is available."
		) from _GUROBI_IMPORT_ERROR


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
	slot_hours: float = 1.0,
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
	_ensure_gurobi()

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


def _demo_data(n_stations: int = 3, n_slots: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[BatterySpec]]:
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

	prices = np.array([0.11] * 7 + [0.20] * 5 + [0.34] * 6 + [0.17] * 6, dtype=float)

	specs = [
		BatterySpec(e_max=45.0, e_min=8.0, p_ch_max=10.0, p_dis_max=10.0, eta_ch=0.95, eta_dis=0.95, e0=20.0)
		for _ in range(n_stations)
	]
	return loads_df, renewables_df, prices, specs


def main(argv: Iterable[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Renewable-aware battery scheduling optimization with Gurobi.")
	parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data.")
	parser.add_argument("--loads_csv", type=str, default=None, help="CSV with base-station loads (columns=stations).")
	parser.add_argument(
		"--renewables_csv",
		type=str,
		default=None,
		help="CSV with available renewable energy (columns=stations, same as loads_csv).",
	)
	parser.add_argument(
		"--prices_csv",
		type=str,
		default=None,
		help="CSV with electricity prices (single column or use --price_col).",
	)
	parser.add_argument("--price_col", type=str, default=None, help="Price column name when prices_csv has multiple columns.")
	parser.add_argument("--curtailment_cost", type=float, default=0.05, help="Penalty per unit of curtailed renewable energy.")
	parser.add_argument("--slot_hours", type=float, default=1.0)
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args(list(argv) if argv is not None else None)

	if args.demo:
		loads_df, renewables_df, prices_arr, specs_arr = _demo_data()
	else:
		if args.loads_csv is None or args.renewables_csv is None or args.prices_csv is None:
			raise ValueError("Provide --demo or all of --loads_csv, --renewables_csv and --prices_csv.")

		loads_df = pd.read_csv(args.loads_csv)
		renewables_df = pd.read_csv(args.renewables_csv)
		prices_df = pd.read_csv(args.prices_csv)

		if args.price_col is not None:
			if args.price_col not in prices_df.columns:
				raise ValueError(f"Price column '{args.price_col}' not found in {args.prices_csv}.")
			prices_arr = prices_df[args.price_col].to_numpy(dtype=float)
		elif prices_df.shape[1] == 1:
			prices_arr = prices_df.iloc[:, 0].to_numpy(dtype=float)
		else:
			raise ValueError("prices_csv has multiple columns. Specify --price_col.")

		if list(loads_df.columns) != list(renewables_df.columns):
			raise ValueError("loads_csv and renewables_csv must have identical station columns in the same order.")

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
		verbose=args.verbose,
	)

	print(f"Optimization status: {result['status']}")
	print(f"Objective value (total cost): {result['objective_value']:.6f}")
	print("\nPer-station summary:")
	print(result["station_summary"].to_string(index=False))


if __name__ == "__main__":
	main()

