# main_two_stage.py

import sys
import os
from pyomo.environ import *
import pandas as pd
import math
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.common import config_final as config
from src.data_processing import loader_final
from src.modeling import model_final as stage2_model_builder
from src.analysis import post_analysis


def check_route_feasibility(data, solver, vehicle_id, route, task_ids_in_route):
    """
    检查给定路径的可行性，并返回其成本。
    """
    if len(route) <= 2:
        return 0, None  # 空路径成本为0

    model = stage2_model_builder.create_operational_model(
        data,
        vehicle_ids=[vehicle_id],
        task_ids=task_ids_in_route,
        fixed_route=route
    )
    results = solver.solve(model, tee=False)

    # *** FIX 2: 接受 'optimal' 或 'feasible' 作为有效解的状态 ***
    termination_condition = results.solver.termination_condition
    if termination_condition == TerminationCondition.optimal or termination_condition == TerminationCondition.feasible:
        # *** FIX 1: 修正Pyomo变量的访问方式，不再使用.get() ***
        # 安全地计算总soc赤字
        total_soc_deficit = sum(value(model.soc_deficit[idx]) for idx in model.soc_deficit if idx[1] == vehicle_id)
        # 安全地计算总延误
        total_delay = sum(value(model.delay_hours[idx]) for idx in model.delay_hours if idx[1] == vehicle_id)

        # 检查软约束是否被满足 (允许微小的计算误差)
        if total_soc_deficit < 0.01 and total_delay < 0.01:
            return value(model.objective), model

    # 如果解不是最优/可行，或者软约束被违反，则视为不可行
    return float('inf'), None


def run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids):
    print("\n" + "=" * 50)
    print("=== RUNNING STAGE 2: GREEDY INSERTION + CORRECTED EXACT CHECKING ===")
    print("=" * 50)
    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in vehicle_ids}
    tasks_in_routes = {k: [] for k in vehicle_ids}
    unassigned_tasks = set(task_ids)

    iteration = 1
    while unassigned_tasks:
        print(f"\n--- [插入迭代 {iteration}] ---")
        print(f"待分配任务池: {len(unassigned_tasks)}")
        best_insertion = {'cost_increase': float('inf'), 'vehicle': None, 'task': None, 'position': None}

        # 计算当前所有车辆的总成本
        current_total_cost = 0
        current_route_costs = {}
        for k in vehicle_ids:
            cost, _ = check_route_feasibility(data, solver, k, routes[k], tasks_in_routes[k])
            # 如果初始状态不可行（虽然不太可能），给一个极大的成本
            if cost == float('inf'):
                print(f"[警告] 车辆 {k} 的当前路径成本计算为无穷大，可能存在问题。")
                cost = 1e9  # 使用一个大数代替，以允许插入尝试
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
                    # 尝试将任务插入到位置i
                    temp_route = current_route[:i] + [customer_node] + current_route[i:]
                    temp_tasks = tasks_in_routes[k] + [task_to_insert]

                    # 检查新路径的可行性和成本
                    new_route_cost, _ = check_route_feasibility(data, solver, k, temp_route, temp_tasks)

                    if new_route_cost != float('inf'):
                        # 计算总成本的变化
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
            routes[k].insert(pos, customer)
            tasks_in_routes[k].append(task)
            unassigned_tasks.remove(task)
            print(
                f"  -> **插入成功**: 将任务 {task} 插入车辆 {k}。成本增加: {best_insertion['cost_increase']:.2f}。新路径: {' -> '.join(routes[k])}")
            iteration += 1
        else:
            print("  -> **本轮无更多可行的插入**，算法结束。")
            break

    print("\n" + "=" * 50)
    print("=== 第二阶段规划完成 ===")

    final_cost = 0
    total_assigned_tasks = 0
    final_models = {}
    for k in vehicle_ids:
        if len(routes[k]) > 2:
            print(f"车辆 {k} 的最终路径: {' -> '.join(routes[k])}")
            cost, model = check_route_feasibility(data, solver, k, routes[k], tasks_in_routes[k])
            if cost != float('inf'):
                final_cost += cost
                total_assigned_tasks += len(tasks_in_routes[k])
                final_models[k] = model

    print(f"总计分配了 {total_assigned_tasks} 个任务。")
    if unassigned_tasks:
        print(f"警告: {len(unassigned_tasks)} 个任务未能被任何车辆服务。任务ID: {unassigned_tasks}")
    return final_cost, True, final_models


def main():
    print("=" * 60)
    print("=== HDT两步走策略优化模型 (贪心插入+精确验证最终版) ===")
    print("=" * 60)
    try:
        model_data = loader_final.create_urban_delivery_scenario()
        if model_data is None: return
        solver = SolverFactory(config.SOLVER_NAME)
        solver.options['TimeLimit'] = 20  # 每次可行性检查的时间限制
        solver.options['MIPGap'] = 0.01
    except Exception as e:
        print(f"\n[致命错误] 环境初始化失败: {e}")
        return

    task_ids = list(model_data['tasks'].keys())
    vehicle_ids = list(model_data['vehicles'].keys())
    if not vehicle_ids or not task_ids:
        print("\n流程终止，无车辆或任务。")
        return

    final_cost, success, final_models = run_greedy_insertion_stage2(model_data, solver, vehicle_ids, task_ids)

    if success and final_models:
        print("\n" + "=" * 50)
        print("=== POST ANALYSIS ===")
        print("=" * 50)
        print(f"所有车辆规划的总成本: {final_cost:.2f}")
        try:
            for k, model in final_models.items():
                if model:
                    post_analysis.plot_road_network_with_routes(model, model_data, filename=f"route_{k}.png")
                    print(f"已为车辆 {k} 生成路径图。")
        except Exception as e:
            print(f"\n[警告] 后分析绘图失败: {e}")
        print("流程成功结束！")
    else:
        print("\n流程结束，但未产生可供分析的最终解。")


if __name__ == "__main__":
    main()