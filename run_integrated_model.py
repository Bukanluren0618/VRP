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


def _build_vehicle_location_sequences(model, vehicle_id, data):
    """Return every location sequence (including subtours) traversed by a vehicle."""
    depot = data['vehicles'][vehicle_id]['depot_id']
    if depot not in model.LOCATIONS:
        return []

    edges = [
        (i, j)
        for i in model.LOCATIONS
        for j in model.LOCATIONS
        if i != j and value(model.x[i, j, vehicle_id]) > 0.5
    ]

    if not edges:
        return []

    adjacency = {}
    for start, end in edges:
        adjacency.setdefault(start, []).append(end)

    route = [depot]
    current = depot
    visited = {depot}
    max_steps = len(model.LOCATIONS) + len(edges)

    for node, neighbours in adjacency.items():
        neighbours.sort(key=lambda item: str(item))

    sequences = []
    remaining_edges = sum(len(neighbours) for neighbours in adjacency.values())

    def _pick_start_node():
        available_nodes = [node for node, neighbours in adjacency.items() if neighbours]
        if not available_nodes:
            return None
        if depot in available_nodes:
            return depot
        return min(available_nodes, key=lambda item: str(item))

    while remaining_edges > 0 and adjacency:
        start_node = _pick_start_node()
        if start_node is None:
            break
        route = [start_node]
        current = start_node
        while True:
            neighbours = adjacency.get(current, [])
            if not neighbours:
                break

            next_node = neighbours.pop(0)
            remaining_edges -= 1
            if not neighbours:
                adjacency.pop(current, None)
            route.append(next_node)
            current = next_node

        if len(route) > 1:
            sequences.append(route)

    return sequences


def _expand_route_to_nodes(location_route, path_matrix):
    """Expand a location-level route into the detailed road-node trajectory."""
    if path_matrix is None or not location_route or len(location_route) < 2:
        return []

    expanded_path = []
    for start, end in zip(location_route[:-1], location_route[1:]):
        if start not in path_matrix.index or end not in path_matrix.columns:
            continue
        segment = path_matrix.loc[start, end]
        if not isinstance(segment, list) or not segment:
            continue
        if not expanded_path:
            expanded_path.extend(segment)
        else:
            expanded_path.extend(segment[1:])
    return expanded_path

def _build_vehicle_node_paths(location_sequences, path_matrix):
    """Expand every location sequence to node-level trajectories."""
    node_segments = []
    for sequence in location_sequences:
        expanded = _expand_route_to_nodes(sequence, path_matrix)
        node_segments.append({'locations': sequence, 'node_path': expanded})
    return node_segments


def _extract_vehicle_routes(model, data):
    """Collect location sequences and node-level paths for every vehicle."""
    vehicle_routes = {}
    path_matrix = data.get('path_matrix')
    for vehicle_id in model.VEHICLES:
        location_sequences = _build_vehicle_location_sequences(model, vehicle_id, data)
        node_segments = _build_vehicle_node_paths(location_sequences, path_matrix)
        vehicle_routes[vehicle_id] = {
            'segments': node_segments,
        }
    return vehicle_routes




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

        vehicle_routes = _extract_vehicle_routes(model, data)
        print("\n--- Vehicle Routes ---")
        for vehicle_id, info in vehicle_routes.items():
            segments = info.get('segments', []) if isinstance(info, dict) else []
            if not segments:
                print(f"  {vehicle_id}: 未形成有效的行驶路径")
                continue

            print(f"  {vehicle_id}:")
            for idx, segment in enumerate(segments, start=1):
                locations = segment.get('locations') or []
                node_path = segment.get('node_path') or []
                if locations:
                    location_str = ' -> '.join(str(loc) for loc in locations)
                else:
                    location_str = '无有效地点序列'
                if node_path and len(node_path) > 1:
                    node_str = ' -> '.join(str(node) for node in node_path)
                elif node_path:
                    node_str = str(node_path[0])
                else:
                    node_str = '无有效路网轨迹'

                print(f"    Segment {idx} Locations: {location_str}")
                print(f"    Segment {idx} Road Nodes: {node_str}")

        visualizations.plot_road_network_with_routes(
            road_network=data['traffic_graph'],
            solution_routes=vehicle_routes,
            data=data,
            output_dir=output_dir,
            title="Integrated Model Vehicle Trajectories",
        )

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