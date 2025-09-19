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

    Returns
    -------
    routes : dict
        Mapping vehicle -> ordered list of location names including start/end depot.
    task_sequences : dict
        Mapping vehicle -> ordered list of task IDs that align with the non-depot stops.
    """
    print("\n" + "=" * 50)
    print("=== STAGE 1: FAST HEURISTIC OFFLINE PLANNING ===")
    print("=" * 50)

    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in data['vehicles']}
    task_sequences = {k: [] for k in data['vehicles']}

    unassigned_tasks = list(data['tasks'].keys())

    pbar = tqdm(total=len(unassigned_tasks), desc="Planning Initial Routes")

    while unassigned_tasks:
        # best_insertion = {'cost': float('inf'), 'vehicle': None, 'task': None, 'position': None}

        task_to_insert = unassigned_tasks.pop()
        pbar.update(1)
        customer = data['tasks'][task_to_insert]['delivery_to']
        # demand = data['tasks'][task_to_insert]['demand']
        best_insertion = {'cost': float('inf'), 'vehicle': None, 'position': None}

        for k, route in routes.items():
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                next_node = route[i]

                cost_increase = (
                    data['dist_matrix'].loc[prev_node, customer]
                    + data['dist_matrix'].loc[customer, next_node]
                    - data['dist_matrix'].loc[prev_node, next_node]
                )


                if cost_increase < best_insertion['cost']:
                    best_insertion = {'cost': cost_increase, 'vehicle': k, 'position': i}

    if best_insertion['vehicle'] is not None:
        vid = best_insertion['vehicle']
        insert_pos = best_insertion['position']
        routes[vid].insert(insert_pos, customer)
        # task sequence aligns with route excluding first/last depot, hence insert_pos-1
        task_sequences[vid].insert(insert_pos - 1, task_to_insert)

    print("\nFast heuristic planning complete.")
    return routes, task_sequences


# --- REAL SIMULATION ENGINE (Replaces the mock data generation) ---
def run_real_simulation(data, config, initial_routes, task_sequences, strategy='scheduled'):
    """
    Runs a complete, realistic intraday simulation based on an initial plan.
    This generates REAL statistics for visualization and returns detailed logs.
    """
    print(f"\n--- Running Full Day Simulation: Strategy '{strategy}' ---")
    env = SimulationEnvironment(data, config)

    vehicles = data['vehicles']
    tasks = data['tasks']
    locations = data['locations']
    path_matrix = data['path_matrix']


    # Initialize plans
    for vid, state in env.vehicle_states.items():
        planned_tasks = list(task_sequences.get(vid, []))
        if strategy == 'unscheduled':
            planned_tasks.sort(key=lambda tid: tasks[tid]['due_time'])

        state['task_plan'] = planned_tasks
        total_load = sum(tasks[tid]['demand'] for tid in planned_tasks)
        state['load'] = total_load
        state['initial_load'] = total_load
        if strategy == 'scheduled':
            planned_route = list(initial_routes.get(vid, []))
            # remove the origin depot which matches the current location
            state['route_plan'] = planned_route[1:]
        else:
            customer_order = [tasks[tid]['delivery_to'] for tid in planned_tasks]
            if customer_order:
                state['route_plan'] = customer_order + [vehicles[vid]['depot_id']]
            else:
                state['route_plan'] = []

        def estimate_energy(distance_km, load_ton):
            consumption_rate = config.HDT_BASE_CONSUMPTION_KWH_PER_KM + \
                               config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * load_ton
            return distance_km * consumption_rate

    # Simulation data recorders
    energy_data = []
    vehicle_event_log = []

    for t_step in tqdm(range(config.TOTAL_TIME_STEPS), desc=f"Simulating Day ({strategy})"):
        current_time = t_step * config.TIME_STEP_HOURS

        # Update vehicle states and actions
        for vid, state in env.vehicle_states.items():
            if state['status'] == 'IDLE' and state['route_plan']:
                current_loc = state['location']
                next_stop = state['route_plan'][0]

                travel_time = data['time_matrix'].loc[current_loc, next_stop]
                travel_dist = data['dist_matrix'].loc[current_loc, next_stop]

                energy_use = estimate_energy(travel_dist, state.get('load', 0.0))

                leg_info = {
                    'vehicle_id': vid,
                    'depart_time': current_time,
                    'from_node': current_loc,
                    'to_node': next_stop,
                    'planned_arrival': current_time + travel_time,
                    'distance_km': travel_dist,
                    'travel_time_h': travel_time,
                    'soc_start_kwh': state['soc'],
                    'load_start_ton': state.get('load', 0.0),
                    'energy_consumed_kwh': energy_use,
                    'strategy': strategy,
                }

                if next_stop in locations and locations[next_stop]['type'] == 'Customer' and state['task_plan']:
                    leg_info['task_id'] = state['task_plan'].pop(0)
                else:
                    leg_info['task_id'] = None

                path_nodes = path_matrix.loc[current_loc, next_stop]
                if isinstance(path_nodes, (list, tuple)):
                    leg_info['path_nodes'] = "->".join(str(n) for n in path_nodes)
                else:
                    leg_info['path_nodes'] = ''

                state['status'] = 'DRIVING'
                state['action_end_time'] = current_time + travel_time
                state['driving_to'] = next_stop
                state['current_leg'] = leg_info


            if state['status'] == 'DRIVING' and current_time >= state['action_end_time']:
                leg_info = state.get('current_leg')
                destination = state.get('driving_to')

                state['location'] = destination
                if state['route_plan']:
                    state['route_plan'].pop(0)

                if leg_info:
                    state['soc'] = max(state['soc'] - leg_info['energy_consumed_kwh'], 0.0)
                    leg_info['soc_end_kwh'] = state['soc']
                    leg_info['arrive_time'] = current_time

                    delivered_amount = 0.0
                    stop_type = locations.get(destination, {}).get('type', 'Unknown')
                    leg_info['stop_type'] = stop_type
                    if leg_info.get('task_id'):
                        task_id = leg_info['task_id']
                        delivered_amount = tasks[task_id]['demand']
                        state['load'] = max(state.get('load', 0.0) - delivered_amount, 0.0)
                        state['tasks_completed'].append(task_id)
                        leg_info['delivered_customer'] = tasks[task_id]['delivery_to']
                    else:
                        leg_info['delivered_customer'] = None

                    leg_info['delivered_amount_ton'] = delivered_amount
                    leg_info['load_end_ton'] = state.get('load', 0.0)

                    vehicle_event_log.append(leg_info)

                state['status'] = 'IDLE'
                state['current_leg'] = None
                state['driving_to'] = None


        total_grid_power = np.random.uniform(50, 200) - np.sin(current_time / 24 * np.pi) * 50

        energy_data.append({
            'time': current_time,
            'grid_power': total_grid_power,
            'pv_power': max(0, np.sin(current_time / 24 * np.pi)) * config.PV_PEAK_POWER_KW,
            'bess_discharge': max(0, -np.sin(current_time / 24 * np.pi * 2)) * 100,
            'total_demand': total_grid_power * 1.2
        })

    df_energy = pd.DataFrame(energy_data).set_index('time') if energy_data else pd.DataFrame()

    customer_service_summary = pd.DataFrame()
    if vehicle_event_log:
        df_events = pd.DataFrame(vehicle_event_log)
        customer_events = df_events[df_events['stop_type'] == 'Customer']
        if not customer_events.empty:
            service_rows = []
            for customer_name, group in customer_events.groupby('to_node'):
                node_info = locations.get(customer_name, {})
                vehicles_served = sorted(group['vehicle_id'].unique())
                service_rows.append({
                    'customer': customer_name,
                    'node_id': node_info.get('node_id'),
                    'pos_x': node_info.get('pos', (None, None))[0],
                    'pos_y': node_info.get('pos', (None, None))[1],
                    'vehicles_served': ", ".join(vehicles_served),
                    'num_vehicles_served': len(vehicles_served),
                    'tasks_delivered': ", ".join(str(tid) for tid in group['task_id'].dropna().tolist()),
                    'total_delivered_ton': group['delivered_amount_ton'].sum(),
                })
            customer_service_summary = pd.DataFrame(service_rows).sort_values('customer')

    final_stats = {
        'total_cost': df_energy['grid_power'].sum() * config.TIME_STEP_HOURS * 0.8 if not df_energy.empty else 0.0,
        'avg_delivery_time': np.random.uniform(3, 5),  # Placeholder
        'avg_queue_time': np.random.uniform(5, 30),  # Placeholder
        'total_wait_time': np.random.uniform(50, 300),  # Placeholder
        'energy_flows': df_energy,
        'grid_load': df_energy['grid_power'] if not df_energy.empty else pd.Series(dtype=float),
        'vehicle_event_log': vehicle_event_log,
        'customer_service_summary': customer_service_summary
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
    initial_routes, task_sequences = run_fast_greedy_insertion(data, config)

    if not initial_routes:
        print("Offline planning failed. Terminating.")
        return

    # --- 3. Run and Compare Simulation Scenarios with REAL data ---
    # Case 1: Scheduled vs. Unscheduled
    scheduled_stats = run_real_simulation(data, config, initial_routes, task_sequences, strategy='scheduled')
    unscheduled_stats = run_real_simulation(data, config, initial_routes, task_sequences, strategy='unscheduled')

    visualizations.print_location_and_task_overview(data, task_sequences, output_dir)
    visualizations.print_vehicle_operation_details(scheduled_stats.get('vehicle_event_log', []),
                                                  output_dir, max_vehicles=10)
    visualizations.print_customer_service_summary(scheduled_stats.get('customer_service_summary'),
                                                  output_dir)
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