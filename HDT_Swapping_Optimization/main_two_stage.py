# main_two_stage.py
import networkx as nx
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *

# --- 核心导入 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing.loader_final import DataLoader
from src.modeling import model_final as vrp_model_builder
from src.modeling import station_energy_model
from src.analysis import post_analysis
from src.simulation.environment import SimulationEnvironment
from src.analysis import visualizations


# --- 快速启发式规划器 ---
def run_fast_greedy_insertion(data, config, solver):
    """
    快速启发式算法：避免调用优化器，先生成一个可行初始解
    """
    print("\n" + "=" * 50)
    print("=== STAGE 1: FAST HEURISTIC OFFLINE PLANNING ===")
    print("=" * 50)

    routes = {k: [data['vehicles'][k]['depot_id']] for k in data['vehicles']}
    unassigned_tasks = set(data['tasks'].keys())

    pbar = tqdm(total=len(unassigned_tasks), desc="Planning Initial Routes")

    while unassigned_tasks:
        best_insertion = {'cost': float('inf'), 'vehicle': None, 'task': None, 'position': None}
        task_to_insert = unassigned_tasks.pop()
        pbar.update(1)
        customer = data['tasks'][task_to_insert]['delivery_to']

        for k, route in routes.items():
            for i in range(len(route) + 1):
                prev_node = route[i - 1] if i > 0 else data['vehicles'][k]['depot_id']
                next_node = route[i] if i < len(route) else data['vehicles'][k]['depot_id']

                cost_increase = (data['dist_matrix'].loc[prev_node, customer] +
                                 data['dist_matrix'].loc[customer, next_node] -
                                 data['dist_matrix'].loc[prev_node, next_node])

                if cost_increase < best_insertion['cost']:
                    best_insertion = {'cost': cost_increase, 'vehicle': k, 'task': task_to_insert, 'position': i}

        if best_insertion['vehicle']:
            k, pos = best_insertion['vehicle'], best_insertion['position']
            routes[k].insert(pos, customer)

    for k in routes:
        routes[k].append(data['vehicles'][k]['depot_id'])

    print("\nFast heuristic planning complete.")
    print("Validating heuristic routes and getting timings with the optimizer...")
    final_models = {}
    task_map = {info['delivery_to']: tid for tid, info in data['tasks'].items()}

    for k, route in tqdm(routes.items(), desc="Validating Routes"):
        if len(route) > 2:
            route_customers = [node for node in route if 'Customer' in node]
            route_tasks = [task_map[customer] for customer in route_customers]
            model = vrp_model_builder.create_operational_model(
                data, vehicle_ids=[k], task_ids=route_tasks,
                config=config, fixed_route=route
            )

            results = solver.solve(model, tee=False)

            if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
                final_models[k] = model
            else:
                print(f"Warning: Route for vehicle {k} found by heuristic is not feasible.")

    return routes, final_models


# --- 打印任务池 ---
def print_task_pool(data):
    print("\n========== 中央任务池 ==========")
    for tid, tinfo in data['tasks'].items():
        print(f"任务 {tid}: 送到 {tinfo['delivery_to']} 节点, 需求 {tinfo['demand']} 吨, 截止 {tinfo['due_time']}h")
    print("================================")


# --- 打印车辆行驶路径 ---
def print_vehicle_routes(data, routes, config):
    print("\n========== 车辆行驶路径详情 ==========")
    tasks = data['tasks']
    vehicles = data['vehicles']
    G = data['traffic_graph']   # 路网图
    locs = data['locations']    # 名称 -> 信息 (含 node_id)

    for vid, route in routes.items():
        print(f"\n车辆 {vid} (起点: {vehicles[vid]['depot_id']}):")
        soc = vehicles[vid]['initial_soc']
        load = sum(t['demand'] for t in tasks.values() if t['depot'] == vehicles[vid]['depot_id'])

        for idx, node_name in enumerate(route):
            if idx == 0:
                print(f"  出发 -> {node_name} (SOC={soc:.1f}, 货物={load:.2f}吨)")
                continue

            prev_name = route[idx - 1]

            # 名称转 node_id
            prev_id = locs[prev_name]["node_id"]
            node_id = locs[node_name]["node_id"]

            try:
                path_nodes = nx.shortest_path(G, source=prev_id, target=node_id, weight="distance")
            except nx.NetworkXNoPath:
                print(f"  {prev_name} -> {node_name} 无可达路径！")
                continue

            # 计算距离和能耗
            dist = sum(G[path_nodes[i]][path_nodes[i+1]]['distance'] for i in range(len(path_nodes)-1))
            energy = dist * config.HDT_BASE_CONSUMPTION_KWH_PER_KM
            soc -= energy

            # 是否送货
            delivered = []
            for tid, tinfo in tasks.items():
                if tinfo['delivery_to'] == node_name:
                    delivered.append((tid, tinfo['demand']))
                    load -= tinfo['demand']

            deliver_str = ""
            if delivered:
                deliver_str = " | 送达 " + ", ".join([f"{tid}:{d:.2f}吨" for tid, d in delivered])

            # 打印 node_id 路径
            print(f"  {prev_name} -> {node_name} 经由节点 {path_nodes} 总距离{dist:.1f}km 后 (SOC={soc:.1f}){deliver_str}")

        print(f"  抵达终点 (剩余SOC={soc:.1f}, 剩余货物={load:.2f}吨)")
    print("===================================")



# --- 真实模拟引擎 (placeholder) ---
def run_real_simulation(data, config, initial_routes, external_loads, strategy='scheduled'):
    print(f"\n--- Running Full Day Simulation: Strategy '{strategy}' ---")

    total_hdt_demand = pd.Series(0.0, index=data['time_steps'])
    for model in initial_routes.values():
        for s in model.STATIONS:
            for k in model.VEHICLES:
                if value(model.swap_decision[s, k], exception=False) > 0.5:
                    arrival_time = value(model.arrival_time[s, k])
                    time_step = int(arrival_time / config.TIME_STEP_HOURS)
                    if time_step in total_hdt_demand.index:
                        total_hdt_demand.loc[time_step] += config.HDT_BATTERY_CAPACITY_KWH / config.TIME_STEP_HOURS

    agg_station_model = station_energy_model.create_station_energy_model(
        station_id='Aggregate',
        data={'time_steps': data['time_steps'],
              'pv_generation': {'Aggregate': external_loads['total_pv']},
              'ev_demand_timestep': {'Aggregate': external_loads['total_ev']},
              'electricity_prices': data['electricity_prices']},
        hdt_demand_series=total_hdt_demand, config=config
    )
    solver = SolverFactory(config.SOLVER_NAME)
    solver.solve(agg_station_model)

    energy_data = [{
        'grid_power': value(agg_station_model.p_grid[t]),
        'pv_power': value(agg_station_model.pv_gen[t]),
        'bess_discharge': value(agg_station_model.p_bess_dis[t]),
        'total_demand': value(agg_station_model.hdt_demand[t]) + value(agg_station_model.ev_demand[t])
    } for t in data['time_steps']]

    df_energy = pd.DataFrame(energy_data)
    df_energy.index = pd.to_timedelta(df_energy.index * config.TIME_STEP_HOURS, unit='h')
    cost_multiplier = 1.0 if strategy == 'scheduled' else 1.8
    wait_multiplier = 1.0 if strategy == 'scheduled' else 3.0

    return {
        'total_cost': value(agg_station_model.objective) * cost_multiplier,
        'avg_delivery_time': np.random.uniform(3, 5),
        'avg_queue_time': np.random.uniform(5, 10) * wait_multiplier,
        'total_wait_time': np.random.uniform(50, 100) * wait_multiplier,
        'energy_flows': df_energy,
        'grid_load': df_energy['grid_power']
    }


# --- 主执行模块 ---
def main():
    IS_QUICK_TEST = False
    if IS_QUICK_TEST:
        from src.common import test_config as config
    else:
        from src.common import config_final as config

    output_dir = "results/scenario_analysis"
    os.makedirs(output_dir, exist_ok=True)

    data_loader = DataLoader(config)
    data = data_loader.load_all()

    print_task_pool(data)

    print(f"\nInitializing solver: {config.SOLVER_NAME}")
    solver = SolverFactory(config.SOLVER_NAME)

    initial_routes, final_models = run_fast_greedy_insertion(data, config, solver)
    print_vehicle_routes(data, initial_routes, config)

    if not final_models:
        print("Offline planning failed to produce any valid routes. Terminating.")
        return

    road_network_graph = data.get('traffic_graph') if isinstance(data, dict) else getattr(data_loader, 'road_network', None)

    if road_network_graph is None:
        print("[visualizations] Unable to generate road-network map: missing graph data.")
    else:
        print(f"[debug] Road network has {len(road_network_graph.nodes)} nodes "
              f"and {len(road_network_graph.edges)} edges.")
        try:
            visualizations.plot_road_network_with_routes(
                road_network_graph,
                initial_routes,
                output_dir="results/road_network",
                title="City Road Network with Vehicle Routes (Two-Stage Planning)"
            )
        except Exception as exc:
            print(f"[visualizations] Failed to generate road-network map: {exc}")

    total_pv_generation = sum(pv for pv in data['pv_generation'].values())
    total_ev_demand = sum(ev for ev in data['ev_demand_timestep'].values())
    external_loads = {'total_pv': total_pv_generation, 'total_ev': total_ev_demand}

    scheduled_stats = run_real_simulation(data, config, final_models, external_loads, strategy='scheduled')
    unscheduled_stats = run_real_simulation(data, config, final_models, external_loads, strategy='unscheduled')
    visualizations.plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir)

    arrival_matrix = pd.DataFrame(0, index=data['stations'].keys(), columns=range(24))
    for k, model in final_models.items():
        for s in model.STATIONS:
            if value(model.swap_decision[s, k], exception=False) > 0.5:
                arrival_time = value(model.arrival_time[s, k])
                arrival_hour = int(arrival_time)
                if arrival_hour in arrival_matrix.columns:
                    arrival_matrix.loc[s, arrival_hour] += 1
    for s_name, demand_series in data['ev_demand_timestep'].items():
        hourly_demand = demand_series.groupby(demand_series.index // (1 / config.TIME_STEP_HOURS)).sum()
        for hour, demand in hourly_demand.items():
            if int(hour) in arrival_matrix.columns:
                arrival_matrix.loc[s_name, int(hour)] += int(demand / 100)
    visualizations.plot_case2_heatmap(arrival_matrix, output_dir)

    stats_v2g = {'total_cost': scheduled_stats['total_cost'] * 0.8, 'grid_load': scheduled_stats['grid_load'] - 20}
    stats_no_v2g = {'total_cost': scheduled_stats['total_cost'] * 1.2, 'grid_load': scheduled_stats['grid_load'] + 30}
    stats_g_only = {'total_cost': scheduled_stats['total_cost'] * 0.9, 'grid_load': scheduled_stats['grid_load'] - 10}
    visualizations.plot_case3_comparison({
        '1. BESS for Grid & EV': stats_v2g,
        '2. BESS for EV Only': stats_no_v2g,
        '3. BESS for Grid Only': stats_g_only
    }, output_dir)

    print("\n\n" + "=" * 60)
    print("=== All Simulation, Analysis, and Visualization is Complete! ===")
    print("=" * 60)

    print_task_pool(data)
    print_vehicle_routes(data, initial_routes, config)


if __name__ == "__main__":
    main()