# blox2/cli/experiment_with_known_values.py

import argparse
import csv
from dataclasses import dataclass
import datetime as _dt
import faulthandler
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

from blox2 import split_df_by_n_rows, load_features

@dataclass(frozen=True)
class ExperimentConfig:
    n_iters: int
    n_suggestions: int

    features_path: str
    values_path: str
    initial_n_obs: int

    predictor_class: str
    predictor_args: dict[str, Any]
    selector_class: str
    selector_args: dict[str, Any]

    report_interval: int
    sd_plot_cutoff: int
    scatter_plot_intervals: list[int]
    verbose_plot: bool
    output_dir: str = "results"

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def _parse_config(d: dict[str, Any]) -> ExperimentConfig:    
    cfg = ExperimentConfig(
        n_iters=int(d["n_iters"]),
        n_suggestions=int(d.get("n_suggestions", 1)),
        features_path=str(d["features_path"]),
        values_path=str(d["values_path"]),
        initial_n_obs=int(d["initial_n_obs"]),
        predictor_class=str(d["predictor_class"]),
        predictor_args=dict(d.get("predictor_args", {}) or {}),
        selector_class=str(d["selector_class"]),
        selector_args=dict(d.get("selector_args", {}) or {}),
        report_interval=int(d.get("report_interval", 100)), 
        sd_plot_cutoff=int(d.get("sd_plot_cutoff", 0)), 
        verbose_plot=bool(d.get("verbose_plot", False)),
        scatter_plot_intervals=list(d.get("scatter_plot_intervals", []) or []),
        output_dir=str(d.get("output_dir", "results")),
    )
    return cfg

def _resolve_predictor(class_name: str):
    import blox2.predictor as predictor_mod

    if not hasattr(predictor_mod, class_name):
        raise ValueError(f"Predictor class '{class_name}' not found in blox2.predictor.")
    return getattr(predictor_mod, class_name)

def _resolve_selector(class_name: str):
    import blox2 as blox2_mod

    if not hasattr(blox2_mod, class_name):
        raise ValueError(f"Selector class '{class_name}' not found in blox2.")
    return getattr(blox2_mod, class_name)

def _make_output_dir(output_dir: str, config_path: str) -> str:
    config_name = Path(config_path).stem
    ts = _dt.datetime.now().strftime("%m-%d_%H%M%S")
    out_dir = os.path.join(output_dir, f"{ts}_{config_name}")
    scatter_out_dir = os.path.join(out_dir, "scatter")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scatter_out_dir, exist_ok=True)
    return out_dir

def _copy_config(config_path: str, out_dir: str) -> None:
    dst = os.path.join(out_dir, "config.yaml")
    shutil.copyfile(config_path, dst)

def _write_candidate_history_csv(out_dir: str, selector) -> None:
    path = os.path.join(out_dir, "candidate_history.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for x in selector.candidate_id_history[selector.initial_n_obs:]:
            writer.writerow([x])

def _write_observation_csvs(out_dir: str, selector) -> np.ndarray:
    observation_history = selector.make_observation_history()
    np.savetxt(os.path.join(out_dir, "initial_observed_properties.csv"), observation_history[:selector.initial_n_obs], delimiter=",",)
    np.savetxt(os.path.join(out_dir, "observation_history.csv"), observation_history[selector.initial_n_obs:], delimiter=",",)
    return observation_history

def _write_time_consumption(out_dir: str, selector) -> None:
    df = pd.DataFrame({
        "Selection": selector.passed_times_selection,
        "Train": selector.passed_times_train,
        "Pred": selector.passed_times_pred,
        "Total": selector.passed_times_total
    })
    df["Misc"] = df["Total"] - df["Selection"] - df["Train"] - df["Pred"]
    df.to_csv(os.path.join(out_dir, "time_consumption.csv"), index=False)

    ax = df.drop(columns="Total").plot()
    ax.set_xlabel("Number of samplings")
    ax.set_ylabel("Time consumption (sec)")
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(out_dir, "time_consumption.png"))
    plt.close()

# def _plot_stein_discrepancy(out_dir: str, observation_history: np.ndarray, fixed_data_for_scaler: np.ndarray, sigma: float, initial_n_obs: int, cutoff: int=0):
#     sd_trajectory = stein_discrepancy_trajectory(observation_history, scale=fixed_data_for_scaler, sigma=sigma,)

#     y = sd_trajectory[initial_n_obs + cutoff:]
#     x = np.arange(cutoff + 1, cutoff + 1 + len(y))

#     plt.figure()
#     plt.plot(x, y)
#     plt.xscale("log", base=10)
#     plt.xlabel("Number of samplings")
#     plt.ylabel("Stein discrepancy")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"stein_discrepancy_s={sigma}.png"))
#     plt.close()

#     np.savetxt(os.path.join(out_dir, f"stein_discrepancy_history_s={sigma}.csv"), sd_trajectory[initial_n_obs:], delimiter=",",)

def _plot_scatter(out_dir: str, observation_history: np.ndarray, initial_n_obs: int, intervals: list[int], x_label: str, y_label: str):
    for n in intervals:
        if n <= 0:
            continue

        end = n + initial_n_obs
        if end > len(observation_history):
            continue
        data = observation_history[:end]

        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Number of sampling: {n}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scatter", f"scatter_{n}.png"))
        plt.close()

def run_experiment(config_path: str) -> str:
    cfg_raw = _load_yaml(config_path)
    cfg = _parse_config(cfg_raw)

    out_dir = _make_output_dir(cfg.output_dir, config_path)
    _copy_config(config_path, out_dir)

    # Load data
    print("Loading data...")
    features_df = load_features(cfg.features_path)
    values_df = pd.read_csv(cfg.values_path)

    observed_features, unchecked_features = split_df_by_n_rows(features_df, cfg.initial_n_obs)
    observed_values, unchecked_values = split_df_by_n_rows(values_df, cfg.initial_n_obs)

    def get_true_value(idx: int) -> np.ndarray:
        return np.asarray(unchecked_values.iloc[idx - len(observed_features)])

    # Initialization
    print("Initializing the selector...")
    predictor_class = _resolve_predictor(cfg.predictor_class)
    predictor = predictor_class(**cfg.predictor_args)

    selector_class = _resolve_selector(cfg.selector_class)
    if cfg.verbose_plot:
        verbose_plot_dir = os.path.join(out_dir, "verbose_plots")
    else:
        verbose_plot_dir = None
    selector = selector_class(observed_features, observed_values, unchecked_features, predictor, verbose_plot_dir=verbose_plot_dir, **cfg.selector_args)

    # Main loop
    print("Starting the main loop.")
    t0 = time.perf_counter()
    n_total = min(cfg.n_iters, len(unchecked_features))
    for i in range(n_total):
        ids = selector.next_candidates(n=cfg.n_suggestions)
        for cid in ids:
            selector.observe(cid, get_true_value(cid))

        if (i + 1) % cfg.report_interval == 0:
            print(f"{(i+1) * cfg.n_suggestions} candidates suggested. Passed time: {time.perf_counter() - t0:.3f} sec", flush=True)

    # Record results
    _write_candidate_history_csv(out_dir, selector)
    observation_history = _write_observation_csvs(out_dir, selector)

    # Fixed scaling (uses all property values)
    # fixed_data_for_scaler = np.vstack([observed_values.to_numpy(dtype=float, copy=False), unchecked_values.to_numpy(dtype=float, copy=False)])

    _write_time_consumption(out_dir, selector)
    # _plot_stein_discrepancy(out_dir=out_dir, observation_history=observation_history, fixed_data_for_scaler=fixed_data_for_scaler, sigma=cfg.sigma, initial_n_obs=selector.initial_n_obs, cutoff=cfg.sd_plot_cutoff)

    # Scatter
    x_label = observed_values.columns[0] if len(observed_values.columns) > 0 else "obj0"
    y_label = observed_values.columns[1] if len(observed_values.columns) > 1 else "obj1"
    _plot_scatter(out_dir=out_dir, observation_history=observation_history, initial_n_obs=selector.initial_n_obs, intervals=cfg.scatter_plot_intervals, x_label=x_label, y_label=y_label)

    print(f"Saved results to: {out_dir}")
    return out_dir

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="blox2")
    p.add_argument("-c", "--config", required=True, help="Path to YAML config, e.g. config/example.yaml",)
    return p

def iter_yaml_paths(p: Path, recursive: bool=False) -> list[Path]:
    """
    Collect YAML files from a file or directory, sorted by name.
    """
    if p.is_file():
        if p.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Config must be .yaml/.yml, got: {p}")
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"Config path not found: {p}")

    pattern = "**/*.y*ml" if recursive else "*.y*ml"
    paths = [x for x in p.glob(pattern) if x.is_file()]
    paths.sort(key=lambda x: str(x))
    return paths

def run_batch(config_path: str, recursive: bool=False) -> int:
    p = Path(config_path).expanduser()
    yamls = iter_yaml_paths(p, recursive=recursive)

    if not yamls:
        raise FileNotFoundError(f"No YAML files found in: {p}")

    ok: list[Path] = []
    ng: list[Path] = []

    for cfg in yamls:
        try:
            print("Running: ", cfg)
            run_experiment(str(cfg))
            ok.append(cfg)
        except Exception as e:
            ng.append(cfg)
            print(e)

    if (len(ok) + len(ng)) == 1:
        pass
    elif len(ng) == 0:
        print("All runs completed.")
    else:
        print("Succeeded: ")
        for p in ok:
            print(p)
        print("Failed: ")
        for p in ng:
            print(p)
            
    return 0 if not ng else 1 # exit code

def main() -> None:
    args = build_parser().parse_args()
    exit_code = run_batch(args.config)
    raise SystemExit(exit_code)

if __name__ == "__main__":
    faulthandler.enable()
    main()