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
    **核心验证函数**
    调用精确模型来验证一条固定路径的可行性与成本。
    """
    # 如果路径上只有仓库，成本为0
    if len(route) <= 2:
        return 0, None

    # 构建一个专门用于验证的模型，传入固定的路径
    model = stage2_model_builder.create_operational_model(
        data,
        vehicle_ids=[vehicle_id],
        task_ids=task_ids_in_route,
        fixed_route=route
    )

    results = solver.solve(model, tee=False)

    # 只有当模型找到最优解，并且没有任何电量亏空时，才认为该路径是"严格可行"的
    if results.solver.termination_condition == 'optimal':
        total_soc_deficit = sum(value(model.soc_deficit.get((n, vehicle_id), 0)) for n in model.NODES)
        if total_soc_deficit < 0.01:
            return value(model.objective), model

    # 任何其他情况（超时、不可行、有电量亏空）都返回无穷大成本
    return float('inf'), None


def run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids):
    """
    **主算法**
    贪心插入，每一步都使用精确模型进行验证。
    """
    print("\n" + "=" * 50)
    print("=== RUNNING STAGE 2: GREEDY INSERTION + EXACT CHECKING ===")
    print("=" * 50)

    # 初始化所有车辆的路径（只包含起点和终点仓库）
    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in vehicle_ids}
    tasks_in_routes = {k: [] for k in vehicle_ids}
    unassigned_tasks = set(task_ids)

    iteration = 1
    while unassigned_tasks:
        print(f"\n--- [插入迭代 {iteration}] ---")
        print(f"待分配任务池: {len(unassigned_tasks)}")

        best_insertion = {'cost_increase': float('inf'), 'vehicle': None, 'task': None, 'position': None, 'new_cost': 0}

        # 1. 计算当前所有路径的总成本
        current_total_cost = 0
        for k in vehicle_ids:
            cost, _ = check_route_feasibility(data, solver, k, routes[k], tasks_in_routes[k])
            current_total_cost += cost

        # 2. 尝试所有可能的插入
        pbar = tqdm(
            total=len(unassigned_tasks) * len(vehicle_ids) * (len(max(routes.values(), key=len)) if routes else 1),
            desc="评估插入点")
        for task_to_insert in list(unassigned_tasks):
            customer_node = data['tasks'][task_to_insert]['delivery_to']
            for k in vehicle_ids:
                current_route = routes[k]
                for i in range(1, len(current_route)):
                    pbar.update(1)
                    temp_route = current_route[:i] + [customer_node] + current_route[i:]
                    temp_tasks = tasks_in_routes[k] + [task_to_insert]

                    # 3. 精确验证新路径
                    new_cost, _ = check_route_feasibility(data, solver, k, temp_route, temp_tasks)

                    if new_cost != float('inf'):
                        # 计算成本增量
                        cost_increase = new_cost - current_total_cost
                        if cost_increase < best_insertion['cost_increase']:
                            best_insertion = {'cost_increase': cost_increase, 'vehicle': k, 'task': task_to_insert,
                                              'position': i, 'new_cost': new_cost}
        pbar.close()

        # 4. 执行最佳插入
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
            final_cost += cost
            total_assigned_tasks += len(tasks_in_routes[k])
            final_models[k] = model

    print(f"总计分配了 {total_assigned_tasks} 个任务。")
    if unassigned_tasks:
        print(f"警告: {len(unassigned_tasks)} 个任务未能被任何车辆服务。")

    return final_cost, True, final_models


def main():
    print("=" * 60)
    print("=== HDT两步走策略优化模型 (贪心插入+精确验证最终版) ===")
    print("=" * 60)
    try:
        model_data = loader_final.create_urban_delivery_scenario()
        if model_data is None: return
        solver = SolverFactory(config.SOLVER_NAME)
        # 为快速验证设置参数
        solver.options['TimeLimit'] = 15  # 验证一个路径通常很快
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

    if success:
        print("\n" + "=" * 50)
        print("=== POST ANALYSIS ===")
        print("=" * 50)
        print(f"所有车辆规划的总成本: {final_cost:.2f}")
        try:
            # 只为第一个有路径的车辆绘制路线图
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