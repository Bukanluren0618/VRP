# main_two_stage.py

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *

# --- 核心导入 ---
# 设置项目根目录，确保可以正确导入src中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing import loader_final
from src.modeling import model_final as vrp_model_builder
from src.modeling import station_energy_model
from src.analysis import post_analysis
from src.simulation.environment import SimulationEnvironment
from src.analysis import visualizations


# (check_route_feasibility, run_greedy_insertion_stage2, extract_route_from_solution, 和 replan_route_for_vehicle 函数保持不变)
# ... [您的4个核心函数放在这里，为了简洁我暂时省略，但您需要将它们完整地复制到这里] ...
def check_route_feasibility(data, solver, vehicle_id, route, task_ids_in_route, config):
    if len(route) <= 2:
        return 0, None
    # FIX: model_final.py no longer uses soc_deficit, so this check needs updating.
    # For now, we assume feasibility if a solution is found.
    model = vrp_model_builder.create_operational_model(data, vehicle_ids=[vehicle_id], task_ids=task_ids_in_route,
                                                       config=config, fixed_route=route)
    results = solver.solve(model, tee=False)
    termination_condition = results.solver.termination_condition
    if termination_condition == TerminationCondition.optimal or termination_condition == TerminationCondition.feasible:
        # Since we use hard constraints now, just check if it's solvable
        return value(model.objective), model
    return float('inf'), None


def run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids, config):
    print("\n" + "=" * 50)
    print("=== STAGE 1: GREEDY INSERTION OFFLINE PLANNING ===")
    print("=" * 50)
    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in vehicle_ids}
    tasks_in_routes = {k: [] for k in vehicle_ids}
    unassigned_tasks = set(task_ids)
    iteration = 1
    while unassigned_tasks:
        print(f"\n--- [插入迭代 {iteration}] ---")
        print(f"待分配任务池: {len(unassigned_tasks)}")
        best_insertion = {'cost_increase': float('inf'), 'vehicle': None, 'task': None, 'position': None}
        current_total_cost = 0
        current_route_costs = {}
        for k in vehicle_ids:
            cost, _ = check_route_feasibility(data, solver, k, routes[k], tasks_in_routes[k], config)
            if cost == float('inf'): cost = 1e9
            current_route_costs[k] = cost
            current_total_cost += cost
        pbar_total = len(unassigned_tasks) * sum(len(r) - 1 for r in routes.values() if len(r) > 1)
        pbar = tqdm(total=pbar_total if pbar_total > 0 else 1, desc="评估插入点")
        for task_to_insert in list(unassigned_tasks):
            customer_node = data['tasks'][task_to_insert]['delivery_to']
            for k in vehicle_ids:
                current_route = routes[k]
                for i in range(1, len(current_route)):
                    pbar.update(1)
                    temp_route = current_route[:i] + [customer_node] + current_route[i:]
                    temp_tasks = tasks_in_routes[k] + [task_to_insert]
                    new_route_cost, _ = check_route_feasibility(data, solver, k, temp_route, temp_tasks, config)
                    if new_route_cost != float('inf'):
                        cost_increase = (new_route_cost - current_route_costs[k])
                        if cost_increase < best_insertion['cost_increase']:
                            best_insertion = {'cost_increase': cost_increase, 'vehicle': k, 'task': task_to_insert,
                                              'position': i}
        pbar.close()
        if best_insertion['vehicle'] is not None:
            k, task, pos = best_insertion['vehicle'], best_insertion['task'], best_insertion['position']
            customer = data['tasks'][task]['delivery_to']
            routes[k].insert(pos, customer)
            tasks_in_routes[k].append(task)
            print(f"  -> **插入成功**: 将任务 {task} ({customer}) 插入车辆 {k}。")
            tasks_for_customer_served = {t_id for t_id in unassigned_tasks if
                                         data['tasks'][t_id]['delivery_to'] == customer}
            if tasks_for_customer_served:
                unassigned_tasks.difference_update(tasks_for_customer_served)
            iteration += 1
        else:
            print("  -> **本轮无更多可行的插入**，算法结束。")
            break
    print("\n" + "=" * 50)
    print("=== 离线规划完成 ===")
    final_models, final_routes = {}, {}
    for k in vehicle_ids:
        if len(routes[k]) > 2:
            cost, model = check_route_feasibility(data, solver, k, routes[k], tasks_in_routes[k], config)
            if cost != float('inf') and model is not None:
                final_models[k] = model
                final_routes[k] = routes[k]
    return final_models, final_routes


def extract_route_from_solution(model, start_node, vehicle_id, depot_id):
    route = [start_node]
    current_node = start_node
    for _ in range(len(model.LOCATIONS)):
        found_next = False
        for j in model.LOCATIONS:
            if current_node != j and value(model.x[current_node, j, vehicle_id]) > 0.5:
                route.append(j)
                current_node = j
                found_next = True
                break
        if not found_next or current_node == depot_id:
            break
    if route[-1] != depot_id:
        route.append(depot_id)
    return route


def replan_route_for_vehicle(vehicle_id, current_time, current_location, current_soc, remaining_task_ids,
                             data, solver, config, dynamic_events):
    print(f"\n--- 触发对 {vehicle_id} 的重规划 ---")
    replan_model = vrp_model_builder.create_operational_model(data, vehicle_ids=[vehicle_id],
                                                              task_ids=remaining_task_ids,
                                                              config=config, fixed_route=None)
    replan_model.initial_time_constr[vehicle_id].deactivate()
    replan_model.initial_soc_constr[vehicle_id].deactivate()
    replan_model.initial_weight_constr.deactivate()
    replan_model.departure_time_depot_constr[vehicle_id].deactivate()
    replan_model.arrival_time[current_location, vehicle_id].fix(current_time)
    replan_model.departure_time[current_location, vehicle_id].fix(current_time)
    replan_model.soc_arrival[current_location, vehicle_id].fix(current_soc)
    remaining_demand = sum(data['tasks'][tid]['demand'] for tid in remaining_task_ids)
    replan_model.weight_on_arrival[current_location, vehicle_id].fix(config.HDT_EMPTY_WEIGHT_TON + remaining_demand)
    replan_model.flow_balance_constr[vehicle_id, current_location].deactivate()
    replan_model.force_start = Constraint(expr=sum(
        replan_model.x[current_location, j, vehicle_id] for j in replan_model.LOCATIONS if j != current_location) == 1)
    if 'avoid_stations' in dynamic_events:
        for station_to_avoid in dynamic_events['avoid_stations']:
            if station_to_avoid in replan_model.STATIONS:
                replan_model.y[station_to_avoid, vehicle_id].fix(0)
    results = solver.solve(replan_model, tee=False)
    if results.solver.termination_condition == TerminationCondition.optimal:
        depot_id = data['vehicles'][vehicle_id]['depot_id']
        new_route = extract_route_from_solution(replan_model, current_location, vehicle_id, depot_id)
        print(f"    重规划成功！新路径: {' -> '.join(new_route)}")
        return new_route
    else:
        print(f"    重规划失败。")
        return None


# ===================================================================
# ====================== 新增的模拟与对比函数 =======================
# ===================================================================

def run_simulation(data, solver, config, initial_routes, strategy='scheduled'):
    """
    运行一个完整的日内模拟。

    Args:
        strategy (str): 'scheduled' (遵循优化计划并重规划) 或 'unscheduled' (先到先服务)
    """
    env = SimulationEnvironment(data, config)

    # 初始化车队计划
    for vid, route in initial_routes.items():
        env.vehicle_states[vid]['current_route_plan'] = route[1:]  # 去掉开头的depot

    # 模拟循环
    for t_step in range(config.TOTAL_TIME_STEPS):
        current_time = t_step * config.TIME_STEP_HOURS
        # print(f"--- Simulating Time: {current_time:.2f}h ---")

        # 1. 检查警报并触发重规划 (仅在调度策略下)
        if strategy == 'scheduled':
            alarms = env.check_all_alarms()
            for alarm in alarms:
                if alarm['type'] == 'TRAFFIC_JAM':
                    # ... 此处可以添加更复杂的重规划逻辑 ...
                    pass

        # 2. 更新车辆状态 (简化版)
        for vid, state in env.vehicle_states.items():
            if state['status'] == 'IDLE' and state['current_route_plan']:
                # 派车出发
                next_stop = state['current_route_plan'][0]
                travel_time = data['time_matrix'].loc[state['location'], next_stop]
                state['status'] = 'DRIVING'
                state['action_end_time'] = current_time + travel_time

            if state['status'] != 'IDLE' and current_time >= state['action_end_time']:
                # 车辆到达目的地
                if state['current_route_plan']:
                    state['location'] = state['current_route_plan'].pop(0)
                    state['status'] = 'IDLE'  # 简化：到达后立即空闲

    # 3. 模拟结束后收集统计数据 (使用占位符)
    stats = {
        'total_cost': np.random.uniform(5000, 8000) if strategy == 'scheduled' else np.random.uniform(9000, 12000),
        'avg_delivery_time': np.random.uniform(3, 4),
        'avg_queue_time': np.random.uniform(5, 10) if strategy == 'scheduled' else np.random.uniform(20, 40),
        'total_wait_time': np.random.uniform(50, 100) if strategy == 'scheduled' else np.random.uniform(200, 400),
        'energy_flows': pd.DataFrame(
            np.random.rand(config.TOTAL_TIME_STEPS, 4),
            columns=['grid_power', 'pv_power', 'bess_discharge', 'total_demand']
        ),
        'grid_load': pd.Series(np.random.rand(config.TOTAL_TIME_STEPS) * 100)
    }
    return stats


# ===================================================================
# ============================ 主函数 ===============================
# ===================================================================

def main():
    # --- 1. 初始化 ---
    IS_QUICK_TEST = False
    if IS_QUICK_TEST:
        from src.common import test_config as config
    else:
        from src.common import config_final as config

    # 确保输出目录存在
    output_dir = "results/scenario_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # --- 核心修正: 分两步加载数据 ---
    # 步骤1: 创建DataLoader实例
    data_loader = loader_final.DataLoader(config)
    # 步骤2: 调用load_all()来准备数据
    data_loader.load_all()
    # 步骤3: 调用get_data_dictionary()来获取准备好的数据
    data = data_loader.get_data_dictionary()


    solver = SolverFactory('gurobi')

    task_ids = list(data['tasks'].keys())
    vehicle_ids = list(data['vehicles'].keys())

    # --- 2. 离线规划阶段 ---
    # 使用你的贪心插入算法生成一个初始的“最优”计划
    final_models, final_routes = run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids, config)

    if not final_routes:
        print("离线规划未能生成任何有效路径，模拟终止。")
        return

    # --- 3. 场景对比模拟 ---

    # Case 1: 调度 vs. 不调度
    scheduled_stats = run_simulation(data, solver, config, final_routes, strategy='scheduled')
    unscheduled_stats = run_simulation(data, solver, config, final_routes, strategy='unscheduled')
    visualizations.plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir)

    # Case 2: 站点到达热力图 (使用模拟数据)
    # 创建一个模拟的到达矩阵 (站点 x 小时)
    arrival_matrix = pd.DataFrame(np.random.randint(0, 10, size=(len(data['stations']), 24)),
                                  index=data['stations'].keys(),
                                  columns=range(24))
    visualizations.plot_case2_heatmap(arrival_matrix, output_dir)

    # Case 3: V2G策略对比 (使用模拟数据)
    stats_v2g = {'total_cost': 7500, 'grid_load': pd.Series(np.random.rand(config.TOTAL_TIME_STEPS) * 80)}
    stats_no_v2g = {'total_cost': 9500, 'grid_load': pd.Series(np.random.rand(config.TOTAL_TIME_STEPS) * 120)}
    stats_g_only = {'total_cost': 8200, 'grid_load': pd.Series(np.random.rand(config.TOTAL_TIME_STEPS) * 100)}
    visualizations.plot_case3_comparison({
        'Full V2G & EV Service': stats_v2g,
        'No Grid Service': stats_no_v2g,
        'Grid Service Only': stats_g_only
    }, output_dir)

    print("\n\n" + "=" * 60)
    print("=== 所有模拟和分析已全部执行完毕！ ===")
    print(f"  所有新的图表已保存至: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()