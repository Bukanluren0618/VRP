# case1_analysis.py

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *
import networkx as nx

# --- 核心导入 ---
# 确保能够找到src目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing import loader_final
from src.modeling import model_final as vrp_model_builder
from src.modeling import station_energy_model
from src.analysis import post_analysis, visualizations
from src.common import config_final as config  # 直接使用最终配置


def check_route_feasibility_and_cost(data, solver, vehicle_id, route, task_ids_in_route, config):
    """快速检查固定路径的可行性与成本"""
    if len(route) <= 2:
        return 0, None
    model = vrp_model_builder.create_operational_model(
        data,
        vehicle_ids=[vehicle_id],
        task_ids=task_ids_in_route,
        config=config,
        fixed_route=route
    )
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
        return value(model.objective), model
    return float('inf'), None


def run_greedy_insertion_optimizer(data, solver, vehicle_ids, task_ids, config):
    """
    运行迭代贪心插入算法，为“调度”策略生成一个高质量的全局最优计划。
    """
    print("\n" + "=" * 50)
    print("=== Generating 'Scheduled' Plan via Optimization ===")
    print("=" * 50)

    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in vehicle_ids}
    tasks_in_routes = {k: [] for k in vehicle_ids}
    unassigned_tasks = set(task_ids)

    iteration = 1
    while unassigned_tasks:
        best_insertion = {'cost_increase': float('inf'), 'vehicle': None, 'task': None, 'position': None}

        current_total_cost = 0
        current_route_costs = {}
        for k in vehicle_ids:
            cost, _ = check_route_feasibility_and_cost(data, solver, k, routes[k], tasks_in_routes[k], config)
            current_route_costs[k] = cost if cost != float('inf') else 1e9
            current_total_cost += current_route_costs[k]

        task_to_evaluate = list(unassigned_tasks)[0]
        customer_node = data['tasks'][task_to_evaluate]['delivery_to']

        num_checks = sum(len(r) - 1 for r in routes.values())
        pbar = tqdm(total=num_checks, desc=f"Iter {iteration} | Evaluating Task '{task_to_evaluate}'")

        for k in vehicle_ids:
            for i in range(1, len(routes[k])):
                pbar.update(1)
                temp_route = routes[k][:i] + [customer_node] + routes[k][i:]
                temp_tasks = tasks_in_routes[k] + [task_to_evaluate]
                new_route_cost, _ = check_route_feasibility_and_cost(data, solver, k, temp_route, temp_tasks, config)

                if new_route_cost != float('inf'):
                    cost_increase = (current_total_cost - current_route_costs[k]) + new_route_cost - current_total_cost
                    if cost_increase < best_insertion['cost_increase']:
                        best_insertion = {'cost_increase': cost_increase, 'vehicle': k, 'task': task_to_evaluate,
                                          'position': i}
        pbar.close()

        if best_insertion['vehicle']:
            k, task, pos = best_insertion['vehicle'], best_insertion['task'], best_insertion['position']
            routes[k].insert(pos, data['tasks'][task]['delivery_to'])
            tasks_in_routes[k].append(task)
            unassigned_tasks.remove(task)
            print(f"  -> Inserted '{task}' into Vehicle '{k}'. New Route: {' -> '.join(routes[k])}")
            iteration += 1
        else:
            print(f"  -> WARNING: No feasible insertion for '{task_to_evaluate}'. Skipping.")
            unassigned_tasks.remove(task_to_evaluate)

    print("\nOptimization complete. Final validation...")
    final_models = {}
    for k in tqdm(vehicle_ids, desc="Final Validation"):
        if len(routes[k]) > 2:
            _, model = check_route_feasibility_and_cost(data, solver, k, routes[k], tasks_in_routes[k], config)
            if model:
                final_models[k] = model
    return final_models, routes


def run_simple_heuristic_simulation(data, config):
    """
    为“不调度”策略生成一个基于简单贪心规则的模拟结果。
    逻辑：车辆服务最近的客户，电量低于30%就去最近的换电站。
    """
    print("\n" + "=" * 50)
    print("=== Generating 'Unscheduled' Plan via Simple Heuristic ===")
    print("=" * 50)

    hdt_demand_by_station = {s: pd.Series(0.0, index=data['time_steps']) for s in data['stations']}

    # 深拷贝任务字典，以便我们可以从中移除已分配的任务
    remaining_tasks = data['tasks'].copy()

    for vid in data['vehicles']:
        print(f"\n--- Simulating Vehicle '{vid}' ---")
        depot = data['vehicles'][vid]['depot_id']
        current_loc = depot
        current_soc = config.HDT_BATTERY_CAPACITY_KWH
        current_time = 0.0
        current_weight = config.HDT_EMPTY_WEIGHT_TON

        # 简单装载逻辑：假设车辆一次性装载N个任务的货物
        tasks_to_load = 5

        vehicle_tasks = {tid: t for tid, t in remaining_tasks.items() if t['depot'] == depot}

        # 按距离排序，实现“最近客户优先”
        sorted_tasks = sorted(vehicle_tasks.items(),
                              key=lambda item: data['dist_matrix'].loc[depot, item[1]['delivery_to']])

        tasks_for_this_trip = dict(sorted_tasks[:tasks_to_load])
        if not tasks_for_this_trip:
            continue

        for tid in tasks_for_this_trip:
            current_weight += remaining_tasks[tid]['demand']
            del remaining_tasks[tid]  # 从任务池中移除

        print(f"  Loaded {len(tasks_for_this_trip)} tasks. Initial weight: {current_weight:.2f} tons.")

        for tid, task_info in tasks_for_this_trip.items():
            customer = task_info['delivery_to']

            # 1. 检查去往下个客户点是否需要换电
            dist_to_customer = data['dist_matrix'].loc[current_loc, customer]
            consumption_rate = config.HDT_BASE_CONSUMPTION_KWH_PER_KM + current_weight * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
            energy_needed = dist_to_customer * consumption_rate

            # 如果预估剩余电量低于30%的安全阈值，则强制去换电
            if (current_soc - energy_needed) < (config.HDT_BATTERY_CAPACITY_KWH * 0.3):
                # 寻找最近的换电站
                closest_station = min(data['stations'].keys(), key=lambda s: data['dist_matrix'].loc[current_loc, s])
                dist_to_station = data['dist_matrix'].loc[current_loc, closest_station]

                print(f"  SOC low! Detouring to nearest station: {closest_station}")

                # 前往换电站
                current_time += data['time_matrix'].loc[current_loc, closest_station]
                current_soc -= dist_to_station * consumption_rate
                current_loc = closest_station

                # 换电
                current_time += config.SWAP_DURATION_HOURS
                current_soc = config.HDT_BATTERY_CAPACITY_KWH

                # 记录换电负荷
                time_step = int(current_time / config.TIME_STEP_HOURS)
                if 0 <= time_step < len(hdt_demand_by_station[closest_station]):
                    demand_kw = config.HDT_BATTERY_CAPACITY_KWH / config.TIME_STEP_HOURS
                    hdt_demand_by_station[closest_station].iloc[time_step] += demand_kw

            # 2. 前往客户点
            current_time += data['time_matrix'].loc[current_loc, customer]
            current_soc -= data['dist_matrix'].loc[current_loc, customer] * consumption_rate
            current_loc = customer

            # 3. 卸货
            current_time += config.LOADING_UNLOADING_TIME_HOURS
            current_weight -= task_info['demand']
            print(f"  Serviced '{tid}' at {customer}. Time: {current_time:.2f}h, SOC: {current_soc:.2f}kWh")

    return hdt_demand_by_station


def run_energy_simulation(data, config, hdt_demand_by_station):
    """
    接收HDT换电需求，运行能源站优化模型，计算最终的电网交互和成本。
    """
    total_hdt_demand = sum(hdt_demand_by_station.values())

    # 创建并求解一个聚合所有站点的能源模型
    agg_station_model = station_energy_model.create_station_energy_model(
        station_id='Aggregate',
        data={
            'time_steps': data['time_steps'],
            'pv_generation': {'Aggregate': sum(pv for pv in data['pv_generation'].values())},
            'ev_demand_timestep': {'Aggregate': sum(ev for ev in data['ev_demand_timestep'].values())},
            'electricity_prices': data['electricity_prices']
        },
        hdt_demand_series=total_hdt_demand,
        config=config
    )
    solver = SolverFactory('cbc')  # 使用开源的CBC求解器，更通用
    solver.solve(agg_station_model)

    # 提取能源数据用于绘图
    energy_flows = []
    for t in data['time_steps']:
        energy_flows.append({
            'grid_power': value(agg_station_model.p_grid[t]),
            'pv_power': value(agg_station_model.pv_gen[t]),
            'bess_discharge': value(agg_station_model.p_bess_dis[t]),
            'bess_charge': value(agg_station_model.p_bess_ch[t]),
            'bess_soc': value(agg_station_model.e_bess[t]),
            'total_demand': value(agg_station_model.hdt_demand[t]) + value(agg_station_model.ev_demand[t])
        })
    df_energy = pd.DataFrame(energy_flows)
    df_energy.index = pd.to_timedelta(np.arange(len(df_energy)) * config.TIME_STEP_HOURS, unit='h')

    return {'grid_load': df_energy['grid_power'], 'energy_flows': df_energy}


def main():
    """主执行函数，专注于生成Case 1的对比图"""
    output_dir = "results/case1_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- All outputs will be saved to: {output_dir} ---")

    # 1. 加载数据
    data_loader = loader_final.DataLoader(config)
    data = data_loader.load_all()

    task_ids = list(data['tasks'].keys())
    vehicle_ids = list(data['vehicles'].keys())

    # ----------------------------------------------------------------
    # --- 策略 1: "调度" (Scheduled) ---
    # ----------------------------------------------------------------
    # 使用优化器生成全局最优计划
    solver_gurobi = SolverFactory('gurobi')
    solver_gurobi.options['TimeLimit'] = 10  # 每次检查的时间限制
    scheduled_models, scheduled_routes = run_greedy_insertion_optimizer(data, solver_gurobi, vehicle_ids, task_ids,
                                                                        config)

    # 从优化模型中提取HDT换电需求
    scheduled_hdt_demand = {s: pd.Series(0.0, index=data['time_steps']) for s in data['stations']}
    for k, model in scheduled_models.items():
        for s in model.STATIONS:
            if value(model.swap_decision[s, k], exception=False) > 0.5:
                arrival_time = value(model.arrival_time[s, k])
                time_step = int(arrival_time / config.TIME_STEP_HOURS)
                if 0 <= time_step < len(scheduled_hdt_demand[s]):
                    demand_kw = config.HDT_BATTERY_CAPACITY_KWH / config.TIME_STEP_HOURS
                    scheduled_hdt_demand[s].iloc[time_step] += demand_kw

    # 运行能源模拟
    scheduled_stats = run_energy_simulation(data, config, scheduled_hdt_demand)
    print("Scheduled scenario simulation complete.")

    # ----------------------------------------------------------------
    # --- 策略 2: "不调度" (Unscheduled) ---
    # ----------------------------------------------------------------
    # 使用简单启发式规则模拟
    unscheduled_hdt_demand = run_simple_heuristic_simulation(data, config)

    # 运行能源模拟
    unscheduled_stats = run_energy_simulation(data, config, unscheduled_hdt_demand)
    print("Unscheduled scenario simulation complete.")

    # ----------------------------------------------------------------
    # --- 生成所有您需要的图表 ---
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("=== Generating All Requested Plots for Case 1 ===")
    print("=" * 50)

    # 图1: 路网图 (使用"调度"策略的优化结果)
    visualizations.plot_road_network_with_routes(
        road_network=data['traffic_graph'],
        solution_routes=scheduled_routes,
        data=data,
        output_dir=output_dir,
        title="路网图 ('调度'策略)"
    )

    # 选择一个有代表性的换电站和卡车进行详细分析
    sample_station_id = list(data['stations'].keys())[0] if data['stations'] else None
    sample_vehicle_id = list(scheduled_models.keys())[0] if scheduled_models else None

    if sample_station_id:
        # 图2 & 3: 站内能量图和SOC轨迹图
        visualizations.plot_station_energy_details(
            station_id=sample_station_id,
            scheduled_flows=scheduled_stats['energy_flows'],
            unscheduled_flows=unscheduled_stats['energy_flows'],
            output_dir=output_dir
        )

    if sample_vehicle_id:
        # 图4: 单一HDT卡车状态图
        visualizations.plot_single_truck_metrics(
            vehicle_id=sample_vehicle_id,
            model=scheduled_models[sample_vehicle_id],
            route=scheduled_routes[sample_vehicle_id],
            data=data,
            config=config,
            output_dir=output_dir
        )

    # 图5: 削峰填谷曲线图
    visualizations.plot_peak_shaving_comparison(
        scheduled_grid_load=scheduled_stats['grid_load'],
        unscheduled_grid_load=unscheduled_stats['grid_load'],
        output_dir=output_dir
    )

    print(f"\nSUCCESS: All plots for Case 1 have been generated in '{output_dir}'")


if __name__ == "__main__":
    main()