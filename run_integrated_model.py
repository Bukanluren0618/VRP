# run_integrated_model.py

import sys
import os
import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt

# --- Core Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing import loader_final
from src.modeling import integrated_model
from src.common import config_final as config
from src.analysis import visualizations


def main():
    """
    Main execution function for the integrated optimization model.
    """
    # Fix for Chinese font display in plots
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    output_dir = "results/integrated_model_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- All outputs will be saved to: {output_dir} ---")

    # 1. Load Data
    print("Loading scenario data...")
    data_loader = loader_final.DataLoader(config)
    data = data_loader.load_all()

    # 2. Build the Integrated Model
    model = integrated_model.create_integrated_model(data, config)

    # 3. Solve the Model
    print("\n" + "=" * 50)
    print(">>> Starting to solve the Integrated Model with Gurobi <<<")
    print("This is a very complex model. Please be patient.")
    print("=" * 50)

    solver = SolverFactory('gurobi')
    # --- MODIFIED: Increased time limit and gap for the complex grid-aware model ---
    solver.options['TimeLimit'] = 7200  # e.g., 2 hours
    solver.options['MIPGap'] = 0.3      # e.g., 30% gap is acceptable initially

    results = solver.solve(model, tee=True)

    # 4. Analyze and Visualize Results
    if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
        print("\nSUCCESS: Integrated model found a feasible/optimal solution!")
        print(f"Final Objective (Total System Cost): {value(model.objective):,.2f}")

        # --- Extract and Print KPIs ---
        served_tasks = sum(
            value(model.y[data['tasks'][t]['delivery_to'], k]) for t in model.TASKS for k in model.VEHICLES)
        total_dist = sum(
            data['dist_matrix'].loc[i, j] * value(model.x[i, j, k]) for i in model.LOCATIONS for j in model.LOCATIONS
            for k in model.VEHICLES if i != j)
        total_energy_cost = sum(
            value(model.p_grid[s, t]) * config.TIME_STEP_HOURS * data['electricity_prices'][t] for s in model.STATIONS
            for t in model.T)

        print("\n--- KEY PERFORMANCE INDICATORS ---")
        print(f"  - Tasks Served: {served_tasks:.0f} / {len(model.TASKS)}")
        print(f"  - Total Distance Traveled: {total_dist:,.2f} km")
        print(f"  - Total Energy Cost: {total_energy_cost:,.2f} Yuan")
        print("------------------------------------")

        # --- Visualization ---
        grid_load = pd.Series(
            [sum(value(model.p_grid[s, t]) for s in model.STATIONS) for t in model.T],
            index=pd.to_timedelta(np.arange(len(model.T)) * config.TIME_STEP_HOURS, unit='h')
        )
        ev_load = pd.Series(
            [sum(data['ev_demand_timestep'][s][t] for s in model.STATIONS) for t in model.T],
            index=pd.to_timedelta(np.arange(len(model.T)) * config.TIME_STEP_HOURS, unit='h')
        )

        visualizations.plot_peak_shaving_comparison(
            scheduled_grid_load=grid_load,
            unscheduled_grid_load=ev_load,
            output_dir=output_dir
        )

        print(f"Result plots have been saved to '{output_dir}'")

    else:
        print("\nFAILURE: The integrated model could not be solved.")
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")


if __name__ == "__main__":
    main()