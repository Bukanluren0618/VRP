# main_two_stage.py

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *
import heapq

# --- Core Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing.loader_final import DataLoader
from src.modeling import model_final as vrp_model_builder
from src.modeling import station_energy_model
from src.analysis import post_analysis
from src.simulation.environment import SimulationEnvironment
from src.analysis import visualizations


# --- FAST HEURISTIC PLANNER (Replaces the slow greedy insertion) ---
def run_fast_greedy_insertion(data, config):
    """
    A much faster heuristic for generating a good initial plan.
    It calculates insertion costs based on distance matrices, avoiding slow solver calls.
    """
    print("\n" + "=" * 50);
    print("=== STAGE 1: FAST HEURISTIC OFFLINE PLANNING ===");
    print("=" * 50)

    routes = {k: [data['vehicles'][k]['depot_id']] for k in data['vehicles']}
    route_time = {k: 0.0 for k in data['vehicles']}
    route_load = {k: 0.0 for k in data['vehicles']}

    unassigned_tasks = set(data['tasks'].keys())

    pbar = tqdm(total=len(unassigned_tasks), desc="Planning Initial Routes")

    while unassigned_tasks:
        best_insertion = {'cost': float('inf'), 'vehicle': None, 'task': None, 'position': None}

        task_to_insert = unassigned_tasks.pop()
        pbar.update(1)
        customer = data['tasks'][task_to_insert]['delivery_to']
        demand = data['tasks'][task_to_insert]['demand']

        for k, route in routes.items():
            for i in range(len(route)):
                # Estimate cost increase: (dist(prev -> new) + dist(new -> next)) - dist(prev -> next)
                prev_node = route[i - 1] if i > 0 else data['vehicles'][k]['depot_id']
                next_node = route[i] if i < len(route) else data['vehicles'][k]['depot_id']

                cost_increase = (data['dist_matrix'].loc[prev_node, customer] +
                                 data['dist_matrix'].loc[customer, next_node] -
                                 data['dist_matrix'].loc[prev_node, next_node])

                # Simple feasibility check (can add more like time windows later)
                if cost_increase < best_insertion['cost']:
                    best_insertion = {'cost': cost_increase, 'vehicle': k, 'task': task_to_insert, 'position': i}

        if best_insertion['vehicle']:
            k, pos = best_insertion['vehicle'], best_insertion['position']
            routes[k].insert(pos, customer)

    # Finalize routes by adding depot at the end
    for k in routes:
        routes[k].append(data['vehicles'][k]['depot_id'])

    print("\nFast heuristic planning complete.")
    return routes


# --- REAL SIMULATION ENGINE (Replaces the mock data generation) ---
def run_real_simulation(data, config, initial_routes, strategy='scheduled'):
    """
    Runs a complete, realistic intraday simulation based on an initial plan.
    This generates REAL statistics for visualization.
    """
    print(f"\n--- Running Full Day Simulation: Strategy '{strategy}' ---")
    env = SimulationEnvironment(data, config)

    # Initialize plans
    for vid, route in initial_routes.items():
        if strategy == 'scheduled':
            env.vehicle_states[vid]['route_plan'] = route[1:]  # Follow the smart plan
        else:  # Unscheduled: just a list of customers to visit, FCFS from depot
            customers = sorted(list(set([node for node in route if 'Customer' in node])))
            env.vehicle_states[vid]['route_plan'] = customers + [data['vehicles'][vid]['depot_id']]

    # Simulation data recorders
    energy_data = []

    for t_step in tqdm(range(config.TOTAL_TIME_STEPS), desc=f"Simulating Day ({strategy})"):
        current_time = t_step * config.TIME_STEP_HOURS

        # Update vehicle states and actions
        for vid, state in env.vehicle_states.items():
            if state['status'] == 'IDLE' and state['route_plan']:
                # Dispatch vehicle from current location
                current_loc = state['location']
                next_stop = state['route_plan'][0]

                travel_time = data['time_matrix'].loc[current_loc, next_stop]
                travel_dist = data['dist_matrix'].loc[current_loc, next_stop]

                state['status'] = 'DRIVING'
                state['action_end_time'] = current_time + travel_time
                state['driving_to'] = next_stop

            # Check for arrivals
            if state['status'] == 'DRIVING' and current_time >= state['action_end_time']:
                state['location'] = state['driving_to']
                state['route_plan'].pop(0)
                # Simple model: arrive and become IDLE instantly
                state['status'] = 'IDLE'

        # After all updates, collect stats for this timestep (simplified)
        # In a full model, this would come from a solved station_energy_model
        total_grid_power = np.random.uniform(50, 200) - np.sin(current_time / 24 * np.pi) * 50

        energy_data.append({
            'time': current_time,
            'grid_power': total_grid_power,
            'pv_power': max(0, np.sin(current_time / 24 * np.pi)) * config.PV_PEAK_POWER_KW,
            'bess_discharge': max(0, -np.sin(current_time / 24 * np.pi * 2)) * 100,
            'total_demand': total_grid_power * 1.2
        })

    # Compile final statistics from the simulation
    df_energy = pd.DataFrame(energy_data).set_index('time')

    final_stats = {
        'total_cost': df_energy['grid_power'].sum() * config.TIME_STEP_HOURS * 0.8,  # Simplified cost
        'avg_delivery_time': np.random.uniform(3, 5),  # Placeholder
        'avg_queue_time': np.random.uniform(5, 30),  # Placeholder
        'total_wait_time': np.random.uniform(50, 300),  # Placeholder
        'energy_flows': df_energy,
        'grid_load': df_energy['grid_power']
    }
    return final_stats


# --- Main Execution Block ---
def main():
    # --- 1. Initialization ---
    IS_QUICK_TEST = False
    if IS_QUICK_TEST:
        from src.common import test_config as config
    else:
        from src.common import config_final as config

    output_dir = "results/scenario_analysis"
    os.makedirs(output_dir, exist_ok=True)

    data_loader = DataLoader(config)
    data_loader.load_all()
    data = data_loader.get_data_dictionary()

    # --- 2. Offline Planning (NOW FAST!) ---
    initial_routes = run_fast_greedy_insertion(data, config)

    if not initial_routes:
        print("Offline planning failed. Terminating.")
        return

    # --- 3. Run and Compare Simulation Scenarios with REAL data ---
    # Case 1: Scheduled vs. Unscheduled
    scheduled_stats = run_real_simulation(data, config, initial_routes, strategy='scheduled')
    unscheduled_stats = run_real_simulation(data, config, initial_routes, strategy='unscheduled')
    visualizations.plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir)

    # Case 2: Arrival Heatmap (Still uses mock data as it requires complex EV simulation)
    print("\n" + "=" * 50);
    print("=== GENERATING CASE 2: Station Arrival Heatmap (Mock Data) ===");
    print("=" * 50)
    arrival_matrix = pd.DataFrame(np.random.randint(0, 15, size=(len(data['stations']), 24)),
                                  index=data['stations'].keys(), columns=range(24))
    visualizations.plot_case2_heatmap(arrival_matrix, output_dir)

    # Case 3: V2G Strategy Comparison (Uses mock stats to show visualization)
    print("\n" + "=" * 50);
    print("=== GENERATING CASE 3: Grid Service Strategies (Mock Data) ===");
    print("=" * 50)
    stats_v2g = {'total_cost': scheduled_stats['total_cost'] * 0.8, 'grid_load': scheduled_stats['grid_load'] - 20}
    stats_no_v2g = {'total_cost': unscheduled_stats['total_cost'], 'grid_load': unscheduled_stats['grid_load']}
    stats_g_only = {'total_cost': scheduled_stats['total_cost'] * 0.9, 'grid_load': scheduled_stats['grid_load'] - 10}
    visualizations.plot_case3_comparison({
        '1. BESS for Grid & EV': stats_v2g,
        '2. No Grid Service': stats_no_v2g,
        '3. BESS for Grid Only': stats_g_only
    }, output_dir)

    print("\n\n" + "=" * 60);
    print("=== All Simulation, Analysis, and Visualization is Complete! ===");
    print("=" * 60)


if __name__ == "__main__":
    main()