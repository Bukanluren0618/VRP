# main_two_stage.py

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *
import inspect
import heapq

# --- Core Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing.loader_final import DataLoader
from src.modeling import model_final as vrp_model_builder
from src.modeling import station_energy_model
from src.analysis import post_analysis
from src.simulation.environment import SimulationEnvironment
from src.analysis import visualizations


def _format_table_for_print(df, float_cols=None, digits=2):
    """Fallback formatter mirroring the visualization helpers."""
    if df is None or df.empty:
        return ""

    formatters = {}
    if float_cols:
        for col in float_cols:
            if col in df.columns:
                formatters[col] = lambda x, d=digits: "--" if pd.isna(x) else f"{x:.{d}f}"

    return df.to_string(index=False, formatters=formatters, na_rep='--')

def _print_vehicle_operation_summary(data, vehicle_summary_df, output_dir=None, title=None, file_tag=None):
    """Call the visualization summary helper with a graceful fallback."""
    summary_printer = getattr(visualizations, 'print_vehicle_operation_summary', None)
    if callable(summary_printer):
        summary_printer(data, vehicle_summary_df, output_dir, title=title, file_tag=file_tag)
        return

    header = title or "车辆运营总览"
    print("\n" + "=" * 30 + f" {header} " + "=" * 30)
    print("（提示：检测到旧版可视化模块缺少车辆运营总览方法，已启用回退输出。）")

    vehicles_info = data.get('vehicles', {})
    vehicle_ids = list(vehicles_info.keys())
    base_home_map = {vid: info.get('depot_id') for vid, info in vehicles_info.items()}
    base_df = pd.DataFrame({
        'vehicle_id': vehicle_ids,
        'base_home_depot': [base_home_map.get(vid) for vid in vehicle_ids]
    })

    if vehicle_summary_df is not None and not vehicle_summary_df.empty:
        summary_df = base_df.merge(vehicle_summary_df, on='vehicle_id', how='left')
    else:
        summary_df = base_df.copy()

    if 'home_depot' in summary_df.columns:
        summary_df['home_depot'] = summary_df['home_depot'].fillna(
            summary_df['vehicle_id'].map(base_home_map)
        )
    else:
        summary_df['home_depot'] = summary_df['vehicle_id'].map(base_home_map)
    summary_df = summary_df.drop(columns=['base_home_depot'], errors='ignore')

    defaults = {
        'total_tasks': 0,
        'unique_customers': 0,
        'total_delivered_ton': 0.0,
        'total_distance_km': 0.0,
        'total_travel_time_h': 0.0,
        'total_energy_kwh': 0.0,
        'earliest_depart_h': np.nan,
        'latest_return_h': np.nan,
        'min_soc_kwh': np.nan,
        'end_soc_kwh': np.nan,
        'delivery_details': ''
    }

    for col, default in defaults.items():
        if col not in summary_df.columns:
            summary_df[col] = default

    initial_soc_map = {vid: info.get('initial_soc', np.nan) for vid, info in vehicles_info.items()}
    summary_df['min_soc_kwh'] = summary_df['min_soc_kwh'].fillna(
        summary_df['vehicle_id'].map(initial_soc_map)
    )
    summary_df['end_soc_kwh'] = summary_df['end_soc_kwh'].fillna(
        summary_df['vehicle_id'].map(initial_soc_map)
    )

    summary_df['total_tasks'] = summary_df['total_tasks'].fillna(0).astype(int)
    summary_df['unique_customers'] = summary_df['unique_customers'].fillna(0).astype(int)
    summary_df['total_delivered_ton'] = summary_df['total_delivered_ton'].fillna(0.0)
    summary_df['total_distance_km'] = summary_df['total_distance_km'].fillna(0.0)
    summary_df['total_travel_time_h'] = summary_df['total_travel_time_h'].fillna(0.0)
    summary_df['total_energy_kwh'] = summary_df['total_energy_kwh'].fillna(0.0)
    summary_df['delivery_details'] = summary_df['delivery_details'].fillna('')

    summary_df = summary_df.sort_values(by=['total_tasks', 'total_distance_km'], ascending=False)

    rename_map = {
        'vehicle_id': '车辆',
        'home_depot': '所属仓库',
        'total_tasks': '任务数量',
        'unique_customers': '服务客户数',
        'total_delivered_ton': '累计卸货量(t)',
        'total_distance_km': '累计行驶距离(km)',
        'total_travel_time_h': '累计行驶时间(h)',
        'total_energy_kwh': '能耗(kWh)',
        'earliest_depart_h': '最早出发(h)',
        'latest_return_h': '最晚返回(h)',
        'min_soc_kwh': '最低SOC(kWh)',
        'end_soc_kwh': '返回SOC(kWh)',
        'delivery_details': '客户任务汇总'
    }

    float_cols = ['累计卸货量(t)', '累计行驶距离(km)', '累计行驶时间(h)', '能耗(kWh)',
                  '最早出发(h)', '最晚返回(h)', '最低SOC(kWh)', '返回SOC(kWh)']

    display_df = summary_df.rename(columns=rename_map)
    print(_format_table_for_print(display_df, float_cols=float_cols))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = 'vehicle_operation_summary.csv'
        if file_tag:
            filename = f'vehicle_operation_summary_{file_tag}.csv'
        export_path = os.path.join(output_dir, filename)
        summary_df.to_csv(export_path, index=False)
        summary_df.to_csv(export_path, index=False)
        print(f"车辆运营总览已保存至: {export_path}")


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
            if planned_tasks:
                # remove the origin depot which matches the current location
                state['route_plan'] = planned_route[1:] if len(planned_route) > 1 else []
            else:
                state['route_plan'] = []
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
    vehicle_summary = pd.DataFrame({
        'vehicle_id': list(vehicles.keys()),
        'home_depot': [vehicles[vid]['depot_id'] for vid in vehicles]
    })
    if vehicle_event_log:
        df_events = pd.DataFrame(vehicle_event_log)
        df_events = df_events.sort_values(by=['vehicle_id', 'depart_time']).reset_index(drop=True)
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

            deliveries = df_events[df_events['task_id'].notna()]
            if not deliveries.empty:
                delivery_totals = deliveries.groupby('vehicle_id').agg(
                    total_tasks=('task_id', 'count'),
                    total_delivered_ton=('delivered_amount_ton', 'sum'),
                    unique_customers=('delivered_customer', lambda s: s.dropna().nunique())
                )

                distance_totals = df_events.groupby('vehicle_id')['distance_km'].sum()
                time_totals = df_events.groupby('vehicle_id')['travel_time_h'].sum()
                energy_totals = df_events.groupby('vehicle_id')['energy_consumed_kwh'].sum()
                depart_times = df_events.groupby('vehicle_id')['depart_time'].min()
                arrive_times = df_events.groupby('vehicle_id')['arrive_time'].max()
                soc_cols = [c for c in ['soc_start_kwh', 'soc_end_kwh'] if c in df_events.columns]
                if soc_cols:
                    min_soc = (
                        df_events.groupby('vehicle_id')[soc_cols]
                        .min()
                        .min(axis=1)
                    )
                else:
                    min_soc = pd.Series(dtype=float)

                end_soc = pd.Series(dtype=float)
                if 'soc_end_kwh' in df_events.columns:
                    end_soc = df_events.groupby('vehicle_id')['soc_end_kwh'].last()

                customer_breakdown = (
                    deliveries.groupby(['vehicle_id', 'delivered_customer'])
                    .agg(
                        delivered_ton=('delivered_amount_ton', 'sum'),
                        num_tasks=('task_id', 'count')
                    )
                    .reset_index()
                )

                if not customer_breakdown.empty:
                    customer_breakdown = customer_breakdown.sort_values(
                        ['vehicle_id', 'delivered_ton'], ascending=[True, False]
                    )
                    customer_breakdown['detail'] = (
                            customer_breakdown['delivered_customer'].astype(str)
                            + ":"
                            + customer_breakdown['delivered_ton'].map(lambda ton: f"{ton:.2f}t")
                            + "/"
                            + customer_breakdown['num_tasks'].astype(int).astype(str)
                            + "单"
                    )
                    delivery_details = customer_breakdown.groupby('vehicle_id')['detail'].agg('; '.join)
                else:
                    delivery_details = pd.Series(dtype=str)

                vehicle_summary = (
                    vehicle_summary
                    .merge(delivery_totals, left_on='vehicle_id', right_index=True, how='left')
                    .merge(distance_totals.rename('total_distance_km'), left_on='vehicle_id', right_index=True,
                           how='left')
                    .merge(time_totals.rename('total_travel_time_h'), left_on='vehicle_id', right_index=True,
                           how='left')
                    .merge(energy_totals.rename('total_energy_kwh'), left_on='vehicle_id', right_index=True, how='left')
                    .merge(depart_times.rename('earliest_depart_h'), left_on='vehicle_id', right_index=True, how='left')
                    .merge(arrive_times.rename('latest_return_h'), left_on='vehicle_id', right_index=True, how='left')
                    .merge(min_soc.rename('min_soc_kwh'), left_on='vehicle_id', right_index=True, how='left')
                    .merge(end_soc.rename('end_soc_kwh'), left_on='vehicle_id', right_index=True, how='left')
                )

                vehicle_summary['delivery_details'] = vehicle_summary['vehicle_id'].map(delivery_details).fillna('')

        defaults = {
            'total_tasks': 0,
            'unique_customers': 0,
            'total_delivered_ton': 0.0,
            'total_distance_km': 0.0,
            'total_travel_time_h': 0.0,
            'total_energy_kwh': 0.0,
            'earliest_depart_h': np.nan,
            'latest_return_h': np.nan,
            'min_soc_kwh': np.nan,
            'end_soc_kwh': np.nan,
            'delivery_details': ''
        }

        for col, default in defaults.items():
            if col not in vehicle_summary.columns:
                vehicle_summary[col] = default

        vehicle_summary['total_tasks'] = vehicle_summary['total_tasks'].fillna(0).astype(int)
        vehicle_summary['unique_customers'] = vehicle_summary['unique_customers'].fillna(0).astype(int)
        vehicle_summary['total_delivered_ton'] = vehicle_summary['total_delivered_ton'].fillna(0.0)
        vehicle_summary['total_distance_km'] = vehicle_summary['total_distance_km'].fillna(0.0)
        vehicle_summary['total_travel_time_h'] = vehicle_summary['total_travel_time_h'].fillna(0.0)
        vehicle_summary['total_energy_kwh'] = vehicle_summary['total_energy_kwh'].fillna(0.0)
        initial_soc_map = {vid: vehicles[vid]['initial_soc'] for vid in vehicles}
        vehicle_summary['min_soc_kwh'] = vehicle_summary['min_soc_kwh'].fillna(
            vehicle_summary['vehicle_id'].map(initial_soc_map))
        vehicle_summary['end_soc_kwh'] = vehicle_summary['end_soc_kwh'].fillna(
            vehicle_summary['vehicle_id'].map(initial_soc_map))
        vehicle_summary['delivery_details'] = vehicle_summary['delivery_details'].fillna('')

    final_stats = {
        'total_cost': df_energy['grid_power'].sum() * config.TIME_STEP_HOURS * 0.8 if not df_energy.empty else 0.0,
        'avg_delivery_time': np.random.uniform(3, 5),  # Placeholder
        'avg_queue_time': np.random.uniform(5, 30),  # Placeholder
        'total_wait_time': np.random.uniform(50, 300),  # Placeholder
        'energy_flows': df_energy,
        'grid_load': df_energy['grid_power'] if not df_energy.empty else pd.Series(dtype=float),
        'vehicle_event_log': vehicle_event_log,
        'customer_service_summary': customer_service_summary,
        'vehicle_summary': vehicle_summary
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
    plot_network_fn = getattr(visualizations, "plot_full_road_network", None)
    if callable(plot_network_fn):
        plot_network_fn(
            data,
            output_dir,
            title="Complete Road Network with Key Facilities"
        )
    else:
        print("plot_full_road_network function not found; skipping full road network plot.")

    _print_vehicle_operation_summary(
        data,
        scheduled_stats.get('vehicle_summary'),
        output_dir,
        title="车辆运营总览（计划调度）"
    )
    visualizations.print_vehicle_operation_details(
        scheduled_stats.get('vehicle_event_log', []),
        output_dir,
        max_vehicles=None,
        title="车辆执行动作明细（计划调度）"
    )
    visualizations.print_customer_service_summary(
        scheduled_stats.get('customer_service_summary'),
        output_dir,
        title="客户服务统计（计划调度）"
    )

    _print_vehicle_operation_summary(
        data,
        unscheduled_stats.get('vehicle_summary'),
        output_dir,
        title="车辆运营总览（即时调度）",
        file_tag="unscheduled"
    )
    visualizations.print_vehicle_operation_details(
        unscheduled_stats.get('vehicle_event_log', []),
        output_dir,
        max_vehicles=None,
        title="车辆执行动作明细（即时调度）",
        file_tag="unscheduled"
    )
    visualizations.print_customer_service_summary(
        unscheduled_stats.get('customer_service_summary'),
        output_dir,
        title="客户服务统计（即时调度）",
        file_tag="unscheduled"
    )

    visualizations.plot_vehicle_routes_on_network(
        data,
        scheduled_stats.get('vehicle_event_log', []),
        output_dir,
        title="Vehicle Routes (Scheduled Dispatch)",
        strategy_tag="scheduled"
    )
    visualizations.plot_vehicle_routes_on_network(
        data,
        unscheduled_stats.get('vehicle_event_log', []),
        output_dir,
        title="Vehicle Routes (On-Demand Dispatch)",
        strategy_tag="unscheduled"
    )

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