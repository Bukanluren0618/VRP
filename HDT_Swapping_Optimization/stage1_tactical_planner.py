# stage1_tactical_planner.py

"""
第一阶段：战术规划模型

核心功能：
- 根据总体的任务需求预测和公司总的车辆数。
- 决定如何为各个发车点（Depot）分配车辆资源。
- 目标是最小化总成本，包括车辆的估算运营成本和因运力不足导致任务失败的惩罚成本。
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import config as cfg  # 导入配置文件

def load_and_aggregate_demand(data_folder_path: str) -> dict[int, int]:
    """
    加载并聚合需求数据。
    这里我们简化处理，从 '车队典型数据汇总.csv' 中读取任务，并按其所属的
    发车点（depot_id）进行分组计数，以估算每个发车点的总任务量。
    在实际应用中，这应是您的TMS/OMS的订单数据接口。
    """
    print("[阶段一] 正在加载和聚合任务需求...")
    try:
        # 实际应根据您的数据文件名修改
        file_path = os.path.join(data_folder_path, "副本4、路网与EV需求参数-final.xlsx - 车队典型数据汇总.csv")
        df = pd.read_csv(file_path)

        if 'depot_id' not in df.columns or 'task_id' not in df.columns:
            raise ValueError("需求数据CSV中必须包含 'depot_id' 和 'task_id' 列。")

        # 按 'depot_id' 分组并计算每个发车点的任务数量
        depot_tasks = df.groupby('depot_id')['task_id'].count().to_dict()
        print(f"[阶段一] 成功加载并聚合需求数据: {depot_tasks}")
        return depot_tasks
    except FileNotFoundError:
        print(f"[错误] 找不到需求数据文件: {file_path}")
        return {}
    except Exception as e:
        print(f"[错误] 处理需求数据时出错: {e}")
        return {}

def solve_fleet_allocation(depot_tasks: dict[int, int]) -> dict[int, int] | None:
    """
    求解第一阶段的车辆分配优化问题。

    Args:
        depot_tasks: 一个字典，键是发车点ID，值是该发车点的总任务数。

    Returns:
        一个字典，表示每个发车点分配到的最优车辆数。如果求解失败则返回 None。
    """
    depots = list(depot_tasks.keys())
    if not depots:
        print("[阶段一] 错误：没有可用的发车点任务数据，无法进行规划。")
        return None

    print("\n--- [阶段一：战术规划] 开始求解 ---")
    print(f"总车队规模: {cfg.TOTAL_TRUCK_FLEET_SIZE} 辆")
    print(f"各发车点任务量: {depot_tasks}")
    print(f"核心假设: 每车能服务 {cfg.AVG_TASKS_PER_TRUCK} 个任务, 未服务任务惩罚为 {cfg.PENALTY_PER_UNSERVED_TASK}")

    model = gp.Model("Tactical_Fleet_Allocation")

    # --- 决策变量 ---
    # n_trucks[d]: 分配给发车点 d 的卡车数量 (整数)
    n_trucks = model.addVars(depots, vtype=GRB.INTEGER, name="n_trucks", lb=0)
    # unserved_tasks[d]: 发车点 d 未能服务的任务数量 (整数)
    unserved_tasks = model.addVars(depots, vtype=GRB.INTEGER, name="unserved_tasks", lb=0)

    # --- 约束 ---
    # 1. 总车辆数约束：所有发车点分配的车辆总数不能超过公司总车队规模
    model.addConstr(gp.quicksum(n_trucks[d] for d in depots) <= cfg.TOTAL_TRUCK_FLEET_SIZE, "Total_Truck_Constraint")

    # 2. 服务能力约束: 将车辆数与可完成的任务数关联起来
    for d in depots:
        required_tasks = depot_tasks.get(d, 0)
        # 未服务的任务 = max(0, 总任务数 - (分配的车辆数 * 每辆车的平均服务能力))
        # Gurobi中表达 max(0, ...) 的方式是 y >= x, y >= 0
        model.addConstr(unserved_tasks[d] >= required_tasks - n_trucks[d] * cfg.AVG_TASKS_PER_TRUCK, f"Service_Capacity_{d}")

    # --- 目标函数 ---
    # 估算的运营成本
    operating_cost = gp.quicksum(n_trucks[d] * cfg.AVG_KM_PER_TRUCK_DAY * cfg.COST_PER_TRUCK_KM for d in depots)
    # 任务失败的惩罚成本
    penalty_cost = gp.quicksum(unserved_tasks[d] * cfg.PENALTY_PER_UNSERVED_TASK for d in depots)
    # 总目标：最小化 (运营成本 + 惩罚成本)
    model.setObjective(operating_cost + penalty_cost, GRB.MINIMIZE)

    # --- 求解 ---
    model.optimize()

    # --- 提取并返回结果 ---
    if model.status == GRB.OPTIMAL:
        allocation = {d: round(n_trucks[d].X) for d in depots}
        total_unserved = sum(unserved_tasks[d].X for d in depots)
        print("--- [阶段一] 求解成功！---")
        print(f"最优车辆分配方案: {allocation}")
        print(f"预计总成本: {model.ObjVal:.2f}")
        print(f"预计无法服务的任务数: {total_unserved:.0f}")
        return allocation
    else:
        if model.status == GRB.INFEASIBLE:
            print("--- [阶段一] 模型无解，执行IIS分析... ---")
            model.computeIIS()
            model.write("stage1_model.ilp")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  -> IIS约束: {c.ConstrName}")
            print("IIS written to stage1_model.ilp")
        else:
            print(f"--- [阶段一] 求解失败！模型状态码: {model.status}")
        return None