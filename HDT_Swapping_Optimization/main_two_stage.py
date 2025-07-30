import sys
import os
from pyomo.environ import *
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_processing import loader_final
from src.modeling import model_final as stage2_model_builder

from src.analysis import post_analysis


# (check_route_feasibility 和 run_greedy_insertion_stage2 函数保持不变)
def check_route_feasibility(data, solver, vehicle_id, route, task_ids_in_route):
    if len(route) <= 2:
        return 0, None
    model = stage2_model_builder.create_operational_model(
        data,
        vehicle_ids=[vehicle_id],
        task_ids=task_ids_in_route,
        fixed_route=route,
    )
    results = solver.solve(model, tee=False)
    termination_condition = results.solver.termination_condition
    if termination_condition == TerminationCondition.optimal or termination_condition == TerminationCondition.feasible:
        total_soc_deficit = sum(value(model.soc_deficit[idx]) for idx in model.soc_deficit if idx[1] == vehicle_id)
        total_delay = sum(value(model.delay_hours[idx]) for idx in model.delay_hours if idx[1] == vehicle_id)
        if total_soc_deficit < 0.01 and total_delay < 0.01:
            return value(model.objective), model
    return float('inf'), None


def run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids, config):
    print("\n" + "=" * 50)
    print("=== STAGE 1 & 2: GREEDY INSERTION + ROUTE CHECKING ===")
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
            cost, _ = check_route_feasibility(
                data,
                solver,
                k,
                routes[k],
                tasks_in_routes[k],
            )
            if cost == float('inf'):
                cost = 1e9
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
                    new_route_cost, _ = check_route_feasibility(
                        data,
                        solver,
                        k,
                        temp_route,
                        temp_tasks,
                    )
                    if new_route_cost != float('inf'):
                        other_routes_cost = current_total_cost - current_route_costs[k]
                        total_new_cost = new_route_cost + other_routes_cost
                        cost_increase = total_new_cost - current_total_cost
                        if cost_increase < best_insertion['cost_increase']:
                            best_insertion = {'cost_increase': cost_increase, 'vehicle': k, 'task': task_to_insert,
                                              'position': i}
        pbar.close()

        if best_insertion['vehicle'] is not None:
            k, task, pos = best_insertion['vehicle'], best_insertion['task'], best_insertion['position']
            customer = data['tasks'][task]['delivery_to']

            # 原始的更新逻辑
            routes[k].insert(pos, customer)
            tasks_in_routes[k].append(task)

            print(
                f"  -> **插入成功**: 将任务 {task} 插入车辆 {k}。成本增加: {best_insertion['cost_increase']:.2f}。新路径: {' -> '.join(routes[k])}")

            # =================================================================
            # ========================= 核心逻辑修正 =========================
            # =================================================================
            # 修正前：只移除了当前被插入的任务ID。
            # unassigned_tasks.remove(task)

            # 修正后：一旦一个客户被服务，就从待分配池中移除所有指向该客户的任务。
            tasks_for_customer_served = {t_id for t_id in unassigned_tasks if
                                         data['tasks'][t_id]['delivery_to'] == customer}

            if tasks_for_customer_served:
                print(
                    f"  -> INFO: 客户 {customer} 已被服务，将移除与该客户相关的全部 {len(tasks_for_customer_served)} 个任务: {tasks_for_customer_served}")
                unassigned_tasks.difference_update(tasks_for_customer_served)
            # =================================================================

            iteration += 1
        else:
            print("  -> **本轮无更多可行的插入**，算法结束。")
            break

    print("\n" + "=" * 50)
    print("=== 第二阶段规划完成 ===")
    final_cost, total_assigned_tasks, final_models, final_routes = 0, 0, {}, {}
    for k in vehicle_ids:
        if len(routes[k]) > 2:
            print(f"车辆 {k} 的最终路径: {' -> '.join(routes[k])}")
            cost, model = check_route_feasibility(
                data,
                solver,
                k,
                routes[k],
                tasks_in_routes[k],
            )
            if cost != float('inf') and model is not None:
                final_cost += cost
                total_assigned_tasks += len(tasks_in_routes[k])
                final_models[k] = model
                final_routes[k] = routes[k]
            else:
                print(f"警告: 车辆 {k} 的最终路径被验证为不可行或无解，将从最终方案中排除。")
    print(f"总计分配了 {total_assigned_tasks} 个任务。")
    if unassigned_tasks:
        print(f"警告: {len(unassigned_tasks)} 个任务未能被任何车辆服务。任务ID: {unassigned_tasks}")
    return final_cost, True, final_models, final_routes


# ===================================================================
# ====================== 新增的在线重规划函数 ========================
# ===================================================================
def extract_route_from_solution(model, start_node, vehicle_id, depot_id):
    """从求解后的模型中，根据x变量的值解析出一条完整的路径。"""
    route = [start_node]
    current_node = start_node

    # 循环查找下一个节点，直到回到仓库
    for _ in range(len(model.LOCATIONS)):
        found_next = False
        for j in model.LOCATIONS:
            if current_node != j and value(model.x[current_node, j, vehicle_id]) > 0.5:
                route.append(j)
                current_node = j
                found_next = True
                break
        if not found_next or current_node == depot_id:
            break  # 如果找不到下一个节点或已回到仓库，则结束

    # 确保最后是仓库
    if route[-1] != depot_id:
        route.append(depot_id)

    return route


def replan_route_for_vehicle(vehicle_id, current_time, current_location, current_soc, remaining_task_ids,
                             data, solver, config, dynamic_events):
    """
    为单个车辆执行在线重规划。

    Args:
        dynamic_events (dict): 包含实时事件信息的字典，例如 {'avoid_stations': ['Station_3']}
    """
    print(f"\n--- 触发对 {vehicle_id} 的重规划 ---")
    print(f"    当前状态: T={current_time:.2f}h, 位置={current_location}, SOC={current_soc:.2f}kWh")
    print(f"    剩余任务ID: {remaining_task_ids}")
    print(f"    动态事件: {dynamic_events}")

    # 1. 创建一个新的、小型的VRP模型，用于求解剩余任务
    # 这个模型不是检查固定路径，而是寻找最优路径，所以fixed_route=None
    replan_model = stage2_model_builder.create_operational_model(
        data,
        vehicle_ids=[vehicle_id],
        task_ids=remaining_task_ids,
        config=config,
        fixed_route=None  # 重要：我们正在寻找新路径，而不是检查固定路径
    )

    # 2. 修改模型以反映当前状态（而不是从0开始）
    # 停用模型中关于从仓库出发的初始约束
    replan_model.initial_time_constr[vehicle_id].deactivate()
    replan_model.initial_soc_constr[vehicle_id].deactivate()
    replan_model.initial_weight_constr.deactivate()  # 载重约束比较复杂，先全部停用
    replan_model.departure_time_depot_constr[vehicle_id].deactivate()

    # 强制车辆的“新起点”为当前位置、时间和SOC
    replan_model.arrival_time[current_location, vehicle_id].fix(current_time)
    replan_model.departure_time[current_location, vehicle_id].fix(current_time)
    replan_model.soc_arrival[current_location, vehicle_id].fix(current_soc)
    # 假设从当前点出发时，已装载所有剩余任务的货物
    remaining_demand = sum(data['tasks'][tid]['demand'] for tid in remaining_task_ids)
    replan_model.weight_on_arrival[current_location, vehicle_id].fix(config.HDT_EMPTY_WEIGHT_TON + remaining_demand)

    # 强制路径从当前节点离开
    replan_model.flow_balance_constr[vehicle_id, current_location].deactivate()
    replan_model.force_start = Constraint(expr=sum(
        replan_model.x[current_location, j, vehicle_id] for j in replan_model.LOCATIONS if j != current_location) == 1)

    # 3. 根据动态事件更新决策环境
    if 'avoid_stations' in dynamic_events:
        for station_to_avoid in dynamic_events['avoid_stations']:
            if station_to_avoid in replan_model.STATIONS:
                print(f"    更新约束: 禁止访问换电站 {station_to_avoid}")
                # 强制该车辆不能访问这个换电站
                replan_model.y[station_to_avoid, vehicle_id].fix(0)

    # 4. 求解这个新的、小型的优化问题
    results = solver.solve(replan_model, tee=False)

    # 5. 解析并返回新路径
    if results.solver.termination_condition == TerminationCondition.optimal:
        depot_id = data['vehicles'][vehicle_id]['depot_id']
        new_route = extract_route_from_solution(replan_model, current_location, vehicle_id, depot_id)
        print(f"    重规划成功！新路径: {' -> '.join(new_route)}")
        return new_route
    else:
        print(f"    重规划失败，未能为 {vehicle_id} 找到可行的剩余路径。")
        return None


def main():
    IS_QUICK_TEST = False  # 在此切换模式：True为快速测试，False为完整运行

    if IS_QUICK_TEST:
        print("=" * 60, "\n=== 运行模式: 快速测试 ===\n" + "=" * 60)
        from src.common import test_config as config
    else:
        print("=" * 60, "\n=== 运行模式: 完整求解 ===\n" + "=" * 60)
        from src.common import config_final as config

    try:
        model_data = loader_final.create_urban_delivery_scenario()
        if model_data is None: return
        solver = SolverFactory(config.SOLVER_NAME)
        if IS_QUICK_TEST:
            solver.options['TimeLimit'] = 2.0
            solver.options['MIPGap'] = 0.1
        else:
            solver.options['TimeLimit'] = 30
            solver.options['MIPGap'] = 0.01
    except Exception as e:
        print(f"\n[致命错误] 环境初始化失败: {e}")
        return

    task_ids = list(model_data['tasks'].keys())
    vehicle_ids = list(model_data['vehicles'].keys())
    if not vehicle_ids or not task_ids:
        print("\n流程终止，无车辆或任务。")
        return

    final_cost, success, final_models, final_routes = run_greedy_insertion_stage2(model_data, solver, vehicle_ids,
                                                                                  task_ids, config)

    if success and final_models:
        final_routes_info = {k: {'route': final_routes[k], 'model': final_models[k]} for k in final_routes}

        # ... (Stage 3.1 & 3.2 基础可视化部分保持不变)
        print("\n" + "=" * 50)
        print("=== STAGE 3.1 & 3.2: 路径规划与能源调度可视化 ===")
        print("=" * 50)
        # ... (此处省略未改变的可视化和能源调度代码)

        # =================================================================
        # ========== 在线动态重规划模拟 (ONLINE RE-PLANNING SIMULATION) ==========
        # =================================================================
        print("\n" + "=" * 60)
        print("=== 在线重规划模拟启动 ===")
        print("=" * 60)

        # 1. 模拟一个事件：假设在第8小时，Station_1所在的电网出现高压
        event_time = 8.0
        congested_station = 'Station_1'

        if congested_station in model_data['stations']:
            print(f"--- 模拟事件 @ T={event_time}h: {congested_station} 电网压力过大，禁止访问！ ---")

            # 2. 找到受影响的车辆：在原计划中，哪辆车在8小时后要去受影响的电站？
            affected_vehicle_id = None
            original_route_info = None
            for vid, r_info in final_routes_info.items():
                model = r_info['model']
                if congested_station in model.STATIONS and post_analysis.safe_value(
                        model.swap_decision[congested_station, vid]) > 0.5:
                    arrival_at_station = post_analysis.safe_value(model.arrival_time[congested_station, vid])
                    if arrival_at_station >= event_time:
                        affected_vehicle_id = vid
                        original_route_info = r_info
                        print(
                            f"找到受影响车辆: {affected_vehicle_id}，原计划在 T={arrival_at_station:.2f}h 到达 {congested_station}")
                        break

            if affected_vehicle_id:
                # 3. 获取车辆在事件发生时的状态
                original_route = original_route_info['route']
                vehicle_model = original_route_info['model']

                # 找到事件发生时，车辆刚离开的那个节点
                current_loc_index = -1
                for i, loc in enumerate(original_route[:-1]):  # 遍历到倒数第二个节点
                    departure_time = post_analysis.safe_value(vehicle_model.departure_time[loc, affected_vehicle_id])
                    if departure_time <= event_time:
                        current_loc_index = i
                    else:
                        break  # 已经超过事件发生时间

                if current_loc_index != -1:
                    last_visited_node = original_route[current_loc_index]

                    # 计算车辆在事件发生时的精确位置和SOC（简化为刚离开节点的状态）
                    current_time_at_replan = post_analysis.safe_value(
                        vehicle_model.departure_time[last_visited_node, affected_vehicle_id])
                    current_soc_at_replan = post_analysis.safe_value(
                        vehicle_model.soc_arrival[last_visited_node, affected_vehicle_id])  # 简化为到达时的SOC

                    # 获取所有尚未访问的客户节点
                    remaining_customer_nodes = [node for node in original_route[current_loc_index + 1:-1] if
                                                'Customer' in node]
                    remaining_tasks_ids = [tid for tid, tinfo in model_data['tasks'].items() if
                                           tinfo['delivery_to'] in remaining_customer_nodes]

                    # 4. 调用重规划引擎
                    dynamic_events = {'avoid_stations': [congested_station]}
                    new_partial_route = replan_route_for_vehicle(affected_vehicle_id, current_time_at_replan,
                                                                 last_visited_node,
                                                                 current_soc_at_replan, remaining_tasks_ids, model_data,
                                                                 solver, config, dynamic_events)

                    if new_partial_route:
                        # 将旧路线和新路线拼接，形成完整的调整后路线
                        final_adjusted_route = original_route[:current_loc_index + 1] + new_partial_route
                        print("\n--- 重规划总结 ---")
                        print(f"原计划路线: {' -> '.join(original_route)}")
                        print(f"调整后路线: {' -> '.join(final_adjusted_route)}")
                else:
                    print("无法确定车辆在事件发生时的位置。")
            else:
                print("没有车辆受到该模拟事件的影响。")
        else:
            print(f"快速测试场景中不存在电站 {congested_station}，跳过在线重规划模拟。")

        print("\n流程成功结束！")

    else:
        print("\n流程结束，但未产生可供分析的最终解。")


if __name__ == "__main__":
    main()