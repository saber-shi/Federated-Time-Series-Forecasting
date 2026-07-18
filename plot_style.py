import re
from pathlib import Path
from typing import Dict, Union, List, Optional
import numpy as np
import matplotlib
# matplotlib.use('WebAgg')

import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font',family='Times New Roman')
matplotlib.rcParams['figure.figsize'] = [6.5, 4] # for square canvas
font_format = {'size':24,'weight':1.5}
line_style = ['--','--','-.',':','--','--','-', '-.']
marker = ["8",">","s","p","P","*","h","H"]
color = ['#8dd3c7','#fdb462','#bebada','#80b1d3','#fb8072','#ffffb3','#b3de69','#fccde5']


def build_log_path(
    method: str,
    dataset: str,
    budget: Union[str, float, int],
    round_: Union[str, int],
    weight: Union[str, float, int],
    root: Union[str, Path] = "./out/classification_log",
) -> Path:
    """
    Build the log file path:
      {root}/{method}_{dataset}_{budget}_{round}_{weight}.log
    """
    def fmt(x: Union[str, float, int]) -> str:
        s = str(x)
        return s
    
    fname = f"{fmt(method)}_{fmt(dataset)}_test{fmt(budget)}_{fmt(round_)}_{fmt(weight)}.log"
    return Path(root) / fname

def load_target_accuracy(log_path: Union[str, Path]) -> Dict[int, float]:
    """
    Return mapping: target_id -> accuracy.
    Parses lines like:
        'classifier: 9/16 target: 13 accuracy: 0.8517 time: 0.6103'
    If the path corresponds to a Non-Private run (filename contains 'Non-Private'),
    redirect to the real baseline log:
        ./out/classification_log/{dataset}_{round_}.log
    """
    p = Path(log_path)

    # Redirect Non-Private synthetic paths to the real baseline path
    name_lower = p.name.lower()
    if "non-private" in name_lower or "nonprivate" in name_lower:
        parts = p.stem.split("_")
        # Expected: method_dataset_test{eps}_{round}_{weight}
        if len(parts) >= 4:
            dataset = parts[1]
            round_token = parts[3]
            baseline_path = Path("./out/classification_log") / f"{dataset}_{round_token}.log"
            if baseline_path.exists():
                p = baseline_path
    acc = {}
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Normalize tabs
            line = line.replace("\t", " ")
            parts = line.split()
            # Find indices of keywords
            try:
                ti = parts.index("target:")
                ai = parts.index("accuracy:")
                target = int(parts[ti + 1])
                accuracy = float(parts[ai + 1])
                acc[target] = accuracy
                acc[target] = 1.0-accuracy
            except (ValueError, IndexError):
                continue
    return acc

def get_all_target_accuracy_from_paths(
    method_paths: List[Union[str, Path]],
) -> Dict[str, Dict[int, float]]:
    """
    Load per-target accuracies for each method log in method_paths.

    Returns:
        {
            method_name (basename pattern "<method>_<weight>")
                -> {target_id -> accuracy}
        }
    """
    accs: Dict[str, Dict[int, float]] = {}
    for mp in method_paths:
        p = Path(mp)
        # if not p.exists():
        #     continue

        parts = p.stem.split("_")
        method_name = f"{parts[0]}"

        accs[method_name] = load_target_accuracy(p)

    return accs

def compute_avg_var_max_accuracy_over_targets_from_paths(
    method_budget_path_list: List[Union[str, Path]],
    target_attrs: List[int],
):
    """
    Compute average, variance, and maximum accuracy on given target attributes
    for each (method, budget) pair in method_budget_path_list.

    Returns:
        avg_map, var_map, max_map with shape:
            {method_name -> {budget_str -> stat}}
    """
    targets = sorted({int(t) for t in target_attrs})
    per_key = defaultdict(lambda: {"avg": [], "var": [], "max": []})

    for mp in method_budget_path_list:
        p = Path(mp)
        # if not p.exists():
        #     continue

        parts = p.stem.split("_")
        if len(parts) < 3:
            continue

        method_name = f"{parts[0]}"
        budget_token = parts[2]
        m_budget = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", budget_token)
        budget_key = m_budget.group(0) if m_budget else budget_token

        acc_map = load_target_accuracy(p)
        if not acc_map:
            per_key[(method_name, budget_key)]["avg"].append(float("nan"))
            per_key[(method_name, budget_key)]["var"].append(float("nan"))
            per_key[(method_name, budget_key)]["max"].append(float("nan"))
            continue

        vals = [acc_map[t] for t in targets if t in acc_map]
        if vals:
            arr = np.asarray(vals, dtype=float)
            a_avg = float(np.mean(arr))
            a_var = float(np.var(arr))
            a_max = float(np.max(arr))
        else:
            a_avg = a_var = a_max = float("nan")

        per_key[(method_name, budget_key)]["avg"].append(a_avg)
        per_key[(method_name, budget_key)]["var"].append(a_var)
        per_key[(method_name, budget_key)]["max"].append(a_max)

    avg_map: Dict[str, Dict[str, float]] = {}
    var_map: Dict[str, Dict[str, float]] = {}
    max_map: Dict[str, Dict[str, float]] = {}

    for (m, b), stats in per_key.items():
        vals_avg = np.array(stats["avg"], dtype=float)
        vals_var = np.array(stats["var"], dtype=float)
        vals_max = np.array(stats["max"], dtype=float)

        r_avg = float(np.nanmean(vals_avg)) if not np.all(np.isnan(vals_avg)) else float("nan")
        r_var = float(np.nanmean(vals_var)) if not np.all(np.isnan(vals_var)) else float("nan")
        r_max = float(np.nanmax(vals_max)) if not np.all(np.isnan(vals_max)) else float("nan")

        avg_map.setdefault(m, {})[b] = r_avg
        var_map.setdefault(m, {})[b] = r_var
        max_map.setdefault(m, {})[b] = r_max

    return avg_map, var_map, max_map

def compute_avg_var_accuracy_all_targets_from_paths(
    method_budget_path_list: List[Union[str, Path]],
):
    """
    Compute average and variance of accuracy over ALL attributes
    for each (method, budget) pair in method_budget_path_list.

    Returns:
        avg_map, var_map with shape:
            {method_name -> {budget_str -> stat}}
    """
    per_key = defaultdict(lambda: {"avg": [], "var": []})

    for mp in method_budget_path_list:
        p = Path(mp)

        parts = p.stem.split("_")
        if len(parts) < 3:
            continue

        # method_dataset_test{eps}_{round}_{weight}.log
        method_name = f"{parts[0]}"
        budget_token = parts[2]
        m_budget = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", budget_token)
        budget_key = m_budget.group(0) if m_budget else budget_token

        acc_map = load_target_accuracy(p)
        if not acc_map:
            per_key[(method_name, budget_key)]["avg"].append(float("nan"))
            per_key[(method_name, budget_key)]["var"].append(float("nan"))
            continue

        vals = list(acc_map.values())
        if vals:
            arr = np.asarray(vals, dtype=float)
            a_avg = float(np.mean(arr))
            a_var = float(np.var(arr))
        else:
            a_avg = a_var = float("nan")

        per_key[(method_name, budget_key)]["avg"].append(a_avg)
        per_key[(method_name, budget_key)]["var"].append(a_var)

    avg_map: Dict[str, Dict[str, float]] = {}
    var_map: Dict[str, Dict[str, float]] = {}

    for (m, b), stats in per_key.items():
        vals_avg = np.array(stats["avg"], dtype=float)
        vals_var = np.array(stats["var"], dtype=float)

        r_avg = float(np.nanmean(vals_avg)) if not np.all(np.isnan(vals_avg)) else float("nan")
        r_var = float(np.nanmean(vals_var)) if not np.all(np.isnan(vals_var)) else float("nan")

        avg_map.setdefault(m, {})[b] = r_avg
        var_map.setdefault(m, {})[b] = r_var

    return avg_map, var_map

def plot_grouped_bars(
    accs_map: Dict[str, Dict[Union[int, str], float]],
    title: str,
    x_label: str,
    y_label: str,
    outdir: Union[str, Path] = "./out/plots",
    out_name: str = "grouped_bars.png",
    dpi: int = 300,
    x_order: Optional[List[Union[int, str]]] = None,
    sort_numeric: bool = True,
) -> Path:
    """
    Plot grouped bars for data shaped as:
      {method_name -> {x_id (attribute_id or budget_id) -> abs_diff}}

    - X axis: attribute_id or budget_id
    - Y axis: absolute difference
    - One bar per method for each X tick.

    Params:
      - x_order: optional explicit order for X ticks; if None, union of keys is used.
      - sort_numeric: if True, try to sort X by numeric value (useful for budgets).
    """
    methods = list(accs_map.keys())
    if not methods:
        raise ValueError("No methods provided")

    # Union of X keys across methods
    x_keys = set()
    for m in methods:
        x_keys.update(accs_map[m].keys())

    if not x_keys:
        raise ValueError("No X keys to plot")

    # Determine X order
    if x_order is not None:
        xs = [k for k in x_order if k in x_keys]
    else:
        xs = list(x_keys)
        if sort_numeric:
            def to_num(v):
                try:
                    return float(v)
                except Exception:
                    return float("inf")
            xs.sort(key=to_num)
        else:
            try:
                xs.sort()
            except Exception:
                pass

    M = len(methods)
    T = len(xs)
    data = np.full((M, T), np.nan)
    for i, m in enumerate(methods):
        for j, xk in enumerate(xs):
            data[i, j] = accs_map.get(m, {}).get(xk, np.nan)

    # Figure size scales with number of X ticks
    fig_w = max(8, min(22, 0.35 * T + 2))
    fig_h = 4 + 0.2 * min(T, 25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    x = np.arange(T)
    width = 0.8 / max(M, 1)
    for i, m in enumerate(methods):
        offsets = x - 0.4 + i * width
        ax.bar(offsets, data[i], width=width, label=m)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in xs], rotation=60, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / out_name
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close(fig)


def plot_grouped_lines(
    series_map: Dict[str, Dict[Union[int, str], float]],
    title: str,
    x_label: str,
    y_label: str,
    outdir: Union[str, Path] = "./out/plots",
    out_name: str = "grouped_lines.png",
    dpi: int = 300,
    x_order: Optional[List[Union[int, str]]] = None,
    sort_numeric: bool = True,
    markers: bool = True,
) -> Path:
    """
    Plot grouped LINES for data shaped as:
      {method_name -> {x_id (attribute_id or budget_id) -> value}}

    - X axis: attribute_id or budget_id
    - Y axis: value
    - One line per method.
    """
    methods = list(series_map.keys())
    if not methods:
        raise ValueError("No methods provided")

    # Union of X keys across methods
    x_keys = set()
    for m in methods:
        x_keys.update(series_map[m].keys())
    if not x_keys:
        raise ValueError("No X keys to plot")

    # Determine X order
    if x_order is not None:
        xs = [k for k in x_order if k in x_keys]
    else:
        xs = list(x_keys)
        if sort_numeric:
            def to_num(v):
                try:
                    return float(v)
                except Exception:
                    return float("inf")
            xs.sort(key=to_num)
        else:
            try:
                xs.sort()
            except Exception:
                pass

    # Prepare data arrays (nan where missing)
    T = len(xs)
    # fig_w = max(8, min(22, 0.45 * T + 2))
    # fig_h = 5
    fig, ax = plt.subplots(dpi=dpi)

    # Use numeric X positions for consistent spacing; label with xs
    x_pos = np.arange(T)
    for m in methods:
        ys = np.array([series_map.get(m, {}).get(xk, np.nan) for xk in xs], dtype=float)
        if np.all(np.isnan(ys)):
            continue
        ax.plot(
            x_pos,
            ys,
            label=m,
            linewidth=2,
            linestyle = line_style[methods.index(m) % len(line_style)],
            color = color[methods.index(m) % len(color)],
            marker = marker[methods.index(m) % len(marker)] if markers else None,
            markersize=4 if markers else 0,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k) for k in xs], rotation=60, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=14,ncol=4,loc='upper center', bbox_to_anchor=(0.5, 1.4), fancybox=True)
    ax.tick_params(axis='both',labelsize=20)
    # plt.subplots_adjust(left=0.14,right=0.99, bottom=0.2, top=0.77)
    plt.xlim(0)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / out_name
    # plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close(fig)
    return out_path

if __name__ == '__main__':
    # data_list = ['nltcs', 'acs', 'adult', 'br2000']
    data_list = ['nltcs']
    epsilon_list = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    # target_tasks = [3, 5, 8, 10]
    # target_tasks = [1,3,5,7,9,11]
    # Individual target tasks per dataset (edit these lists as needed)
    target_tasks_by_dataset = {
        "nltcs": [3, 4, 5, 7, 8, 11, 15],
        "acs":   [2, 4, 9, 11, 12, 14, 18],
        # "adult": [8, 9, 10, 12, 13],
        "adult": [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14],
        "br2000":[1, 6, 8, 11, 13],
    }
    default_target_tasks = [1, 3, 5, 7, 9, 11]
    # balance_weight_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    balance_weight_list = [0.7]
    balance_weight_default = [1.0]
    round_ = 1
    method_list = ['Non-Private', 'PrivBayes', 'PrivMRF', 'PrivBalance']
    # method_list = ['PrivMRF', 'PrivBalance']

    for dataset in data_list:
        # pick dataset-specific targets
        target_tasks = target_tasks_by_dataset.get(dataset, default_target_tasks)
        # baseline_path = Path("./out/classification_log") / f"{dataset}_{round_}.log"
        all_method_paths = []
        outdir = Path("./out/plots_acc_lines") / dataset
        for epsilon in epsilon_list:
            method_paths = []
            for method in method_list:
                # if method == 'Non-Private':
                #     path = baseline_path
                #     method_paths.append(path)
                #     continue
                if method != 'PrivBalance':
                    balance_weights = balance_weight_default.copy()
                else:
                    balance_weights = balance_weight_list.copy()
                for weight in balance_weights:
                    path = build_log_path(
                        method=method,
                        dataset=dataset,
                        budget=epsilon,
                        round_=round_,
                        weight=weight
                    )
                    method_paths.append(path)

            accs = get_all_target_accuracy_from_paths(
                    method_paths=method_paths
                )
            #Plot accuracy bars for each attribute under different methods
            # out_name = f"acc_bars_eps{epsilon}.png"
            # plot_grouped_bars(
            #     accs_map=accs,
            #     title=f"Misclassification rate on {dataset} (ε={epsilon})",
            #     x_label="Target attribute",
            #     y_label="Misclassification rate",
            #     outdir=outdir,
            #     out_name=out_name,
            #     )
            # out_name = f"acc_lines_eps{epsilon}.png"
            # plot_grouped_lines(
            #     series_map=accs,
            #     title=f"Accuracy on {dataset} (ε={epsilon})",
            #     x_label="Target attribute",
            #     y_label="Accuracy",
            #     outdir=outdir,
            #     out_name=out_name,
            #     )
            
            all_method_paths.extend(method_paths)
        avg_map, var_map, max_map = compute_avg_var_max_accuracy_over_targets_from_paths(
            method_budget_path_list=all_method_paths,
            target_attrs=target_tasks
        )

        # avg_map, var_map = compute_avg_var_accuracy_all_targets_from_paths(
        #     method_budget_path_list=all_method_paths
        # )

        # Plot Average absolute accuracy difference bars for each method under different budgets
        # out_name = f"avg_accs_bars.png"
        # plot_grouped_bars(
        #     accs_map=avg_map,
        #     title=f"Average misclassification rate on {dataset}",
        #     x_label="Privacy budget (ε)",
        #     y_label="Average misclassification rate",
        #     outdir=outdir,
        #     out_name=out_name,
        #     )
        
        out_name = f"avg_accs_lines.png"
        plot_grouped_lines(
            series_map=avg_map,
            title=f"Misclassification rate on {dataset}",
            x_label="Privacy budget (ε)",
            y_label="Average misclassification rate",
            outdir=outdir,
            out_name=out_name,
            )
        
        # Plot Variance of absolute accuracy difference bars for each method under different budgets
        # out_name = f"var_accs_bars.png"
        # plot_grouped_bars(
        #     accs_map=var_map,
        #     title=f"Variance of accuracy on {dataset}",
        #     x_label="Privacy budget (ε)",
        #     y_label="Variance of accuracy",
        #     outdir=outdir,
        #     out_name=out_name,
        #     )
        # out_name = f"var_accs_lines.png"
        # plot_grouped_lines(
        #     series_map=var_map,
        #     title=f"Variance of misclassification rate on {dataset}",
        #     x_label="Privacy budget (ε)",
        #     y_label="Variance of misclassification rate",
        #     outdir=outdir,
        #     out_name=out_name,
        #     )