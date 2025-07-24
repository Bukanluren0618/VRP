# main_two_stage.py

import sys
import os
from pyomo.environ import *
import pandas as pd
import math

# 将src目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.common import config_final as config
from src.data_processing import loader_final
from src.modeling import stage1_tactical
from src.modeling import model_final as stage2_model_builder
from src.analysis import post_analysis


def run_stage1(data, solver):
    """第一阶段：战术规划，筛选出值得服务的任务。"""
    print("\n" + "=" * 50)
    print("=== RUNNING STAGE 1: TACTICAL ALLOCATION ===")
    print("=" * 50)
    model_s1 = stage1_tactical.create_tactical_model(data)
    results_s1 = solver.solve(model_s1, tee=False)

    if (results_s1.solver.status == 'ok') and (
            results_s1.solver.termination_condition in ['optimal', 'locallyOptimal']):
        print("[S1-SUCCESS] 战术规划求解成功！")
        allocated_trucks = sum(math.ceil(value(model_s1.n_trucks[d])) for d in model_s1.DEPOTS)
        served_tasks = [t for t in model_s1.TASKS if value(model_s1.is_task_served[t]) > 0.5]
        unserved_tasks_count = len(model_s1.TASKS) - len(served_tasks)

        print(f"  -> 决策: 总共分配 {allocated_trucks} 辆车。")
        print(f"  -> 决策: 服务 {len(served_tasks)} 个任务。")
        if unserved_tasks_count > 0:
            print(f"  -> 警告: {unserved_tasks_count} 个任务因运力或时间限制被放弃。")

        available_vehicles = list(data['vehicles'].keys())
        vehicle_ids_to_plan = available_vehicles[:min(allocated_trucks, len(available_vehicles))]
        return vehicle_ids_to_plan, served_tasks
    else:
        print("[S1-FAILURE] 战术规划求解失败。")
        return None, None


def run_greedy_insertion_stage2(data, solver, vehicle_ids, task_ids):
    """
    第二阶段（最终版）：采用贪心插入启发式，保证能得到解。
    """
    print("\n" + "=" * 50)
    print("=== RUNNING STAGE 2: GREEDY INSERTION EXECUTION ===")
    print("=" * 50)

    remaining_tasks_pool = set(task_ids)
    final_aggregated_model = ConcreteModel()
    final_aggregated_model.objective = Objective(expr=0)
    all_assigned_tasks = set()

    for k_idx, vehicle_id in enumerate(vehicle_ids):
        if not remaining_tasks_pool:
            print("\n所有任务已分配完毕，提前结束规划。")
            break

        print(f"\n--- [规划第 {k_idx + 1}/{len(vehicle_ids)} 辆车: {vehicle_id}] ---")

        committed_tasks_for_this_vehicle = []

        # 持续为当前车辆插入任务，直到无法再插入为止
        while True:
            best_insertion_cost = float('inf')
            best_task_to_insert = None
            best_model_objective = float('inf')

            # 如果任务池空了，就跳出内层循环
            if not remaining_tasks_pool:
                break

            print(
                f"  -> 开始新一轮插入尝试，当前已分配 {len(committed_tasks_for_this_vehicle)} 个任务，剩余任务池 {len(remaining_tasks_pool)} 个。")

            # 遍历所有剩余任务，找到插入成本最低的一个
            for candidate_task in remaining_tasks_pool:

                # 创建一个包含已锁定任务 + 1个候选任务的模型
                tasks_for_this_run = committed_tasks_for_this_vehicle + [candidate_task]

                model_single_insertion = stage2_model_builder.create_operational_model(data, [vehicle_id],
                                                                                       tasks_for_this_run)

                # 对于这种小问题，求解时间可以很短
                solver.options['TimeLimit'] = 20
                results = solver.solve(model_single_insertion, tee=False)

                # 检查这个插入是否可行
                try:
                    # 尝试读取目标函数值，如果成功，说明找到了一个可行解
                    current_objective = value(model_single_insertion.objective)

                    # 检查候选任务是否真的被服务了
                    is_candidate_served = value(
                        model_single_insertion.y[data['tasks'][candidate_task]['delivery_to'], vehicle_id]) > 0.5

                    if is_candidate_served:
                        # 计算插入成本（这里简化为总目标函数值）
                        insertion_cost = current_objective
                        if insertion_cost < best_insertion_cost:
                            best_insertion_cost = insertion_cost
                            best_task_to_insert = candidate_task
                            best_model_objective = current_objective

                except (ValueError, AttributeError):
                    # 如果读取失败，说明这个插入不可行，直接跳过
                    continue

            # 如果在一整轮遍历后，找到了可以插入的任务
            if best_task_to_insert is not None:
                print(f"  --> 成功! 找到最佳插入任务: {best_task_to_insert}，成本为 {best_insertion_cost:.2f}")
                # 将这个任务正式“提交”
                committed_tasks_for_this_vehicle.append(best_task_to_insert)
                all_assigned_tasks.add(best_task_to_insert)
                remaining_tasks_pool.remove(best_task_to_insert)
            else:
                # 如果遍历完所有任务都无法插入，说明这辆车满了
                print(f"  --> 本轮无任务可插入，车辆 {vehicle_id} 规划完成。")
                break  # 结束当前车辆的规划

        # 更新总成本
        if committed_tasks_for_this_vehicle:
            # 使用最后一次成功插入时的模型目标值作为该车的最终成本
            final_aggregated_model.objective.expr += best_model_objective

    print("\n" + "=" * 50)
    print("=== 第二阶段贪心插入规划完成 ===")
    print(f"总计分配了 {len(all_assigned_tasks)} 个任务。")
    if remaining_tasks_pool:
        print(f"警告: {len(remaining_tasks_pool)} 个任务在第二阶段未能被任何车辆服务。")

    return final_aggregated_model, True


def main():
    """主工作流"""
    print("======================================================")
    print("=== HDT两步走策略优化模型 (城市配送场景) ===")
    print("======================================================")

    try:
        model_data = loader_final.create_urban_delivery_scenario()
        if model_data is None: return

        solver = SolverFactory(config.SOLVER_NAME)
        solver.options['MIPGap'] = 0.05
        solver.options['MIPFocus'] = 1
        solver.options['Heuristics'] = 0.8  # 稍微调高启发式，帮助快速找到解
    except Exception as e:
        print(f"\n[致命错误] 环境初始化失败: {e}")
        return

    vehicle_ids, task_ids = run_stage1(model_data, solver)
    if not vehicle_ids or not task_ids:
        print("\n流程终止，因为第一阶段未能成功分配资源。")
        return

    # 调用全新的、更稳健的第二阶段函数
    final_model, success = run_greedy_insertion_stage2(model_data, solver, vehicle_ids, task_ids)

    if success and final_model:
        print("\n" + "=" * 50)
        print("=== POST ANALYSIS ===")
        print("=" * 50)
        if final_model.objective.expr != 0:
            print(f"所有车辆规划的总估算成本: {value(final_model.objective):.2f}")
        else:
            print("所有车辆规划的总估算成本: 0.00 (没有任务被成功分配)")
        print("\n注意：由于采用启发式方法，无法绘制包含所有路径的全局路线图，且成本非最优。")
        print("流程成功结束！")
    else:
        print("\n流程结束，但未产生可供分析的最终解。")


if __name__ == "__main__":
    main()