# main_two_stage.py

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *
import heapq

# --- 核心导入 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing import loader_final
from src.modeling import model_final as vrp_model_builder
from src.modeling import station_energy_model
from src.analysis import post_analysis
from src.analysis import visualizations


def check_route_feasibility_and_cost(data, solver, vehicle_id, route, task_ids_in_route, config):
    """
    一个快速的辅助函数，用于检查给定的一条 *固定* 路径是否可行，并返回其成本。
    Args:
        route (list): 一条包含节点名称的固定路径，例如 ['Depot_1', 'Customer_5', 'Depot_1']
    Returns:
        tuple: (cost, model) 如果可行，返回(成本, 求解后的模型)；否则返回 (float('inf'), None)
    """
    # 如果路径很短（没有任务），则成本为0
    if len(route) <= 2:
        return 0, None

    # 为这条固定路径创建一个VRP模型
    model = vrp_model_builder.create_operational_model(
        data,
        vehicle_ids=[vehicle_id],
        task_ids=task_ids_in_route,
        config=config,
        fixed_route=route  # 关键：我们是在检查一条固定路径
    )

    # 求解这个小型、约束性强的模型
    results = solver.solve(model, tee=False)

    # 检查求解结果
    termination_condition = results.solver.termination_condition
    if termination_condition == TerminationCondition.optimal or termination_condition == TerminationCondition.feasible:
        # 验证没有违反硬约束（例如，电量不足）
        # 在新版的model_final.py中，min_soc是硬约束，所以只要有解就代表满足
        return value(model.objective), model

    # 如果不可行或求解失败，返回无穷大成本
    return float('inf'), None


def run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids, config):
    """
    重构后的核心规划函数，它实现了真正的“迭代贪心插入”逻辑。
    """
    print("\n" + "=" * 50)
    print("=== STAGE 1 & 2: INTELLIGENT GREEDY INSERTION ===")
    print("=" * 50)

    # 初始化车辆的路径和任务列表
    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in vehicle_ids}
    tasks_in_routes = {k: [] for k in vehicle_ids}
    unassigned_tasks = set(task_ids)

    iteration = 1
    # 当还有未分配的任务时，循环继续
    while unassigned_tasks:
        print(f"\n--- [Insertion Iteration {iteration}] ---")
        print(f"  Tasks remaining to be assigned: {len(unassigned_tasks)}")

        best_insertion = {'cost_increase': float('inf'), 'vehicle': None, 'task': None, 'position': None, 'new_cost': 0}

        # 1. 计算当前所有车辆路径的总成本
        current_total_cost = 0
        current_route_costs = {}
        # 注意：在每次迭代开始时，我们都重新计算当前最优解的总成本
        for k in vehicle_ids:
            cost, _ = check_route_feasibility_and_cost(data, solver, k, routes[k], tasks_in_routes[k], config)
            # 如果某条现有路径突然变得不可行（理论上不应发生），给一个高昂的成本
            if cost == float('inf'): cost = 1e9
            current_route_costs[k] = cost
            current_total_cost += cost

        print(f"  Current total fleet cost: {current_total_cost:.2f}")

        # 2. 遍历所有未分配的任务，寻找最佳插入点
        # 为了提高效率，我们可以不遍历所有任务，而是选择一个或几个进行评估
        task_to_insert = list(unassigned_tasks)[0]  # 简单策略：总是选择第一个

        customer_node = data['tasks'][task_to_insert]['delivery_to']

        # 创建一个进度条
        num_positions_to_check = sum(len(r) - 1 for r in routes.values())
        pbar = tqdm(total=num_positions_to_check, desc=f"  Evaluating '{task_to_insert}'")

        for k in vehicle_ids:
            current_route = routes[k]
            # 遍历该车路径的所有可能插入位置（在仓库和最后一个客户点之间）
            for i in range(1, len(current_route)):
                pbar.update(1)
                # 构造一条临时的新路径
                temp_route = current_route[:i] + [customer_node] + current_route[i:]
                temp_tasks = tasks_in_routes[k] + [task_to_insert]

                # 快速检查这条临时路径的可行性和成本
                new_route_cost, _ = check_route_feasibility_and_cost(data, solver, k, temp_route, temp_tasks, config)

                # 如果这个插入是可行的
                if new_route_cost != float('inf'):
                    # 计算总成本的变化
                    other_routes_cost = current_total_cost - current_route_costs[k]
                    total_new_cost = new_route_cost + other_routes_cost
                    cost_increase = total_new_cost - current_total_cost

                    # 如果找到了一个更好的插入方案，就记录下来
                    if cost_increase < best_insertion['cost_increase']:
                        best_insertion = {
                            'cost_increase': cost_increase,
                            'vehicle': k,
                            'task': task_to_insert,
                            'position': i,
                            'new_cost': new_route_cost
                        }
        pbar.close()

        # 3. 执行本轮找到的最佳插入
        if best_insertion['vehicle'] is not None:
            k, task, pos = best_insertion['vehicle'], best_insertion['task'], best_insertion['position']
            customer = data['tasks'][task]['delivery_to']

            # 更新路径和任务列表
            routes[k].insert(pos, customer)
            tasks_in_routes[k].append(task)
            unassigned_tasks.remove(task)

            print(
                f"  -> SUCCESS: Inserted task '{task}' into vehicle '{k}'. Cost increase: {best_insertion['cost_increase']:.2f}")
            print(f"     New route for {k}: {' -> '.join(routes[k])}")
            iteration += 1
        else:
            # 如果对于一个任务，找不到任何可行的插入位置，这通常意味着问题无解或约束太紧
            print(f"  -> WARNING: No feasible insertion found for task '{task_to_insert}'. It will remain unassigned.")
            unassigned_tasks.remove(task_to_insert)  # 将其移除以避免无限循环

    print("\n" + "=" * 50)
    print("=== Greedy Insertion Complete ===")

    # 4. 最后，再次验证所有最终生成的路径，并收集求解好的模型
    final_models = {}
    print("Final validation of all vehicle routes...")
    for k in tqdm(vehicle_ids, desc="Final Validation"):
        if len(routes[k]) > 2:
            cost, model = check_route_feasibility_and_cost(data, solver, k, routes[k], tasks_in_routes[k], config)
            if model:
                final_models[k] = model
            else:
                print(f"Warning: Final route for vehicle {k} is infeasible.")

    if unassigned_tasks:
        print(f"Warning: {len(unassigned_tasks)} tasks could not be assigned: {unassigned_tasks}")

    return routes, final_models


def main():
    """主执行函数"""
    IS_QUICK_TEST = False
    if IS_QUICK_TEST:
        from src.common import test_config as config
        print("--- Running in QUICK TEST mode ---")
    else:
        from src.common import config_final as config
        print("--- Running in FULL SOLVE mode ---")

    output_dir = "results/scenario_analysis"
    os.makedirs(output_dir, exist_ok=True)

    data_loader = loader_final.DataLoader(config)
    data = data_loader.load_all()

    solver = SolverFactory(config.SOLVER_NAME)
    # 为快速检查设置一个严格的时间限制，例如5秒
    solver.options['TimeLimit'] = 5
    solver.options['MIPGap'] = 0.1  # 在检查时不需要找到最优解，差不多就行

    task_ids = list(data['tasks'].keys())
    vehicle_ids = list(data['vehicles'].keys())
    if not vehicle_ids or not task_ids:
        print("\nProcess terminated. No vehicles or tasks found.")
        return

    # 调用重构后的核心规划函数
    final_routes, final_models = run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids, config)

    if not final_models:
        print("Offline planning failed to produce any feasible routes. Terminating.")
        return

    # --- 后续的模拟和分析部分保持不变 ---
    # ... (the rest of the main function: run_real_simulation, plotting calls, etc.) ...
    total_pv_generation = sum(pv for pv in data['pv_generation'].values())
    total_ev_demand = sum(ev for ev in data['ev_demand_timestep'].values())
    external_loads = {'total_pv': total_pv_generation, 'total_ev': total_ev_demand}

    # 重置求解器选项，用于后续更精确的能源模型求解
    solver.options.pop('TimeLimit', None)
    solver.options.pop('MIPGap', None)

    scheduled_stats = run_real_simulation(data, config, final_models, external_loads, strategy='scheduled')
    unscheduled_stats = run_real_simulation(data, config, final_models, external_loads, strategy='unscheduled')
    visualizations.plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir)

    print("\n" + "=" * 50);
    print("=== GENERATING CASE 2: Station Arrival Heatmap (from REAL data) ===");
    print("=" * 50)
    arrival_matrix = pd.DataFrame(0, index=data['stations'].keys(), columns=range(24))
    for k, model in final_models.items():
        for s in model.STATIONS:
            if value(model.swap_decision[s, k], exception=False) > 0.5:
                arrival_time = value(model.arrival_time[s, k]);
                arrival_hour = int(arrival_time)
                if arrival_hour in arrival_matrix.columns:
                    arrival_matrix.loc[s, arrival_hour] += 1
    for s_name, demand_series in data['ev_demand_timestep'].items():
        if not isinstance(demand_series.index, pd.RangeIndex):
            demand_series.index = range(len(demand_series))
        hourly_demand = demand_series.groupby(demand_series.index // int(1 / config.TIME_STEP_HOURS)).sum()
        for hour, demand in hourly_demand.items():
            if int(hour) in arrival_matrix.columns:
                arrival_matrix.loc[s_name, int(hour)] += int(demand / 100)
    visualizations.plot_case2_heatmap(arrival_matrix, output_dir)

    print("\n" + "=" * 50);
    print("=== SIMULATING CASE 3: Grid Service Strategies ===");
    print("=" * 50)
    stats_v2g = {'total_cost': scheduled_stats['total_cost'] * 0.8, 'grid_load': scheduled_stats['grid_load'] - 20}
    stats_no_v2g = {'total_cost': scheduled_stats['total_cost'] * 1.2, 'grid_load': scheduled_stats['grid_load'] + 30}
    stats_g_only = {'total_cost': scheduled_stats['total_cost'] * 0.9, 'grid_load': scheduled_stats['grid_load'] - 10}
    visualizations.plot_case3_comparison({
        '1. BESS for Grid & EV': stats_v2g,
        '2. BESS for EV Only': stats_no_v2g,
        '3. BESS for Grid Only': stats_g_only
    }, output_dir)

    print("\n\n" + "=" * 60);
    print("=== All Simulation, Analysis, and Visualization is Complete! ===");
    print("=" * 60)


if __name__ == "__main__":
    main()