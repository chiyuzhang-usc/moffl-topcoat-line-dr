from pathlib import Path
from typing import Dict
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def run_step08_des_plotter(
    paths: Path
) -> None:
    """Run Step-08 DES plotter.

    Args:
        paths: Step-08 paths.
    """
    if plt is None:
        print("matplotlib not available; skipping Step-08 DES plotting.")
        return
    
    # if path not given, set to default
    if (paths is None) or (not paths.exists()):
        print("Path not given, using default output/step08_plots directory.")
        plot_output_dir = "output/step08_plots"
        plot_output_dir = Path(plot_output_dir)
    else:
        plot_output_dir = paths.parent / "step08_plots"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics data
    metrics_path = paths
    if not metrics_path.exists():
        print(f"Metrics file {metrics_path} does not exist; skipping plotting.")
        return
    metrics_df = pd.read_csv(metrics_path)
    # check if there's data
    if metrics_df.empty:
        print("Metrics data is empty; skipping plotting.")
        return
    print(f"Loaded metrics data from {metrics_path}, shape: {metrics_df.shape}")
    # print out how many points to plot
    print(f"Number of data points to plot: {metrics_df.shape[0]}")

    method_colors = metrics_df["method"].map({
            "step05": "red",
            "step06_milp": "green",
            "step07_nsga2": "blue",
            "step07_nsga2_seeded": "cyan"
        })
    
    print("Generating Step-08 DES plots - Makespan vs Block Fraction and PB Blocked Times")
    # the plot data is from step08_des.py
    plt.figure(figsize=(10, 6))
    
    # plot block fraction max vs make span,
    # use different colors for sdsu algo and milp algo, nsga use another
    # map: method column, step05 to red, step06_milp to green, 
    # step07_nsga2 to blue, step07_nsga2_seeded to cyan
    plt.scatter(
        metrics_df["makespan"],
        metrics_df["pb_block_frac_max"],
        alpha=0.6,
        c=method_colors,
    )
    # mark the one on bottom left corner (best) with a star
    # best_idx = (metrics_df["makespan"] + metrics_df["pb_block_frac_max"]).idxmin()
    # plt.scatter(
    #     metrics_df.loc[best_idx, "makespan"],
    #     metrics_df.loc[best_idx, "pb_block_frac_max"],
    #     marker="*", s=200, c="gold", edgecolors="black", label="Best Solution"
    # )
    # write that best point's in text with the data row, method, makespan and block fraction
    # write it as label at the bottom of the figure
  
    
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Step 05 (SDSU)', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Step 06 (MILP)', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Step 07 (NSGA-II)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Step 07 Seeded (NSGA-II)', markerfacecolor='cyan', markersize=10),
    ])
    plt.xlabel("Makespan")
    plt.ylabel("Max Block Fraction")
    plt.title("Step 08 - Makespan vs Max Block Fraction")


    plt.savefig(plot_output_dir / "S8_makespan_vs_block_fraction.png")
    plt.close()

    print(f"Step-08 DES plots saved to: {plot_output_dir}")
    
    

    