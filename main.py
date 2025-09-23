# main.py

import pandas as pd
from src.data_processing.loader_final import DataLoader
from src.modeling.model_final import create_operational_model
from src.analysis.post_analysis import analyze_solution, extract_routes
from pyomo.environ import SolverFactory
import src.common.config_final as config

# Import the new visualization module we created
from src.analysis.visualizations import (
    plot_road_network_with_routes,
    plot_pv_grid_comparison,
    plot_dispatch_comparison,
    plot_station_queue
)


def main():
    print("Starting the HDT Swapping Optimization Model (Single Stage VRP)...")
    print("NOTE: For advanced simulation, please run 'main_two_stage.py'")

    # 1. Load and preprocess data
    print("Loading data...")
    data_loader = DataLoader(config)
    data_loader.load_all()
    data = data_loader.get_data_dictionary()
    print("Data loaded successfully.")

    # --- FIX: Get vehicle and task IDs from the 'data' dictionary AFTER it's loaded ---
    all_vehicle_ids = list(data['vehicles'].keys())
    all_task_ids = list(data['tasks'].keys())

    vehicle_ids = all_vehicle_ids[:config.NUM_VEHICLES_TO_USE]
    # Use all tasks for the selected vehicles
    tasks_for_selected_vehicles = {tid: t for tid, t in data['tasks'].items() if
                                   any(v in t.get('depot', '') for v in vehicle_ids)}
    task_ids = list(tasks_for_selected_vehicles.keys())

    print(f"Running model for {len(vehicle_ids)} vehicles and {len(task_ids)} tasks.")

    # 2. Create the optimization model
    print("Creating the optimization model...")
    model = create_operational_model(data, vehicle_ids, task_ids, config)
    print("Model created.")

    # 3. Solve the model
    print("Solving the model... This may take some time.")
    solver = SolverFactory('cbc')
    solver.options['threads'] = config.SOLVER_THREADS
    results = solver.solve(model, tee=True)
    print("Solver finished.")

    # 4. Post-analysis and result visualization
    if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
        print("Optimal solution found!")
        solution_summary = analyze_solution(model, data, vehicle_ids, task_ids, config)

        # The new visualizations are more complex and require simulation data,
        # so they are best called from the new two-stage or simulation script.
        # We will call the original road network plot here.
        try:
            solution_routes = extract_routes(model, data, vehicle_ids)
            road_network = data_loader.road_network
            plot_road_network_with_routes(road_network, solution_routes)
        except Exception as e:
            print(f"Could not generate road network plot: {e}")

        print("\nAll tasks completed.")

    else:
        print("Could not find an optimal solution.")
        print("Solver Status:", results.solver.status)
        print("Termination Condition:", results.solver.termination_condition)


if __name__ == '__main__':
    main()