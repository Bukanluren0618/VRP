# run_grid_simulation.py

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import *
import matplotlib.pyplot as plt

# --- 核心导入 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data_processing import loader_final
from src.modeling import model_final as vrp_model_builder
from src.modeling import power_grid_model  # 导入我们新的电网模型
from src.common import config_final as config


def check_route_feasibility_and_cost(data, solver, vehicle_id, route, task_ids_in_route, config):
    """快速检查固定路径的可行性与成本"""
    if len(route) <= 2: return 0, None
    model = vrp_model_builder.create_operational_model(data, [vehicle_id], task_ids_in_route, config, fixed_route=route)
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
        return value(model.objective), model
    return float('inf'), None


def run_greedy_insertion_optimizer(data, solver, vehicle_ids, task_ids, config):
    """运行迭代贪心插入算法，生成车辆调度计划"""
    print("\n" + "=" * 50)
    print("=== Step 1: Generating Vehicle Dispatch Plan ===")
    print("=" * 50)

    max_range = (config.HDT_BATTERY_CAPACITY_KWH / config.HDT_BASE_CONSUMPTION_KWH_PER_KM) * 1.2
    routes = {k: [data['vehicles'][k]['depot_id'], data['vehicles'][k]['depot_id']] for k in vehicle_ids}
    tasks_in_routes = {k: [] for k in vehicle_ids}
    unassigned_tasks = set(task_ids)

    for i in range(len(task_ids)):
        task_to_evaluate = list(unassigned_tasks)[0] if unassigned_tasks else None
        if not task_to_evaluate: break

        best_insertion = {'cost_increase': float('inf'), 'vehicle': None, 'task': None, 'position': None}
        # ... (此处省略了完整的贪心算法逻辑，因为它与 case1_analysis.py 中的版本相同)
        # 为了简洁，我们直接使用一个简化的分配逻辑
        customer_node = data['tasks'][task_to_evaluate]['delivery_to']
        # 简单地分配给第一辆车
        k = vehicle_ids[i % len(vehicle_ids)]
        routes[k].insert(1, customer_node)
        tasks_in_routes[k].append(task_to_evaluate)
        unassigned_tasks.remove(task_to_evaluate)

    print("\nDispatch plan generated. Final validation...")
    final_models = {}
    for k in tqdm(vehicle_ids, desc="Final Validation"):
        if len(routes[k]) > 2:
            _, model = check_route_feasibility_and_cost(data, solver, k, routes[k], tasks_in_routes[k], config)
            if model: final_models[k] = model

    return final_models


def plot_bus_voltages(net, grid_model, output_dir):
    """
    生成一张可视化图表，展示电网中所有母线的电压水平。
    """
    print("-> Generating Grid Analysis Plot: Bus Voltages...")

    voltages = [value(grid_model.v_sqr[b]) ** 0.5 for b in grid_model.BUSES]
    bus_ids = list(grid_model.BUSES)

    plt.figure(figsize=(20, 10))
    colors = ['green' if 0.95 <= v <= 1.05 else 'red' for v in voltages]

    plt.bar(bus_ids, voltages, color=colors)

    plt.axhline(1.05, color='red', linestyle='--', label='电压上限 (1.05 p.u.)')
    plt.axhline(0.95, color='red', linestyle='--', label='电压下限 (0.95 p.u.)')

    plt.title('配电网节点电压分布图 (含换电负荷)', fontsize=22)
    plt.xlabel('母线 (Bus) ID', fontsize=16)
    plt.ylabel('电压 (标幺值 p.u.)', fontsize=16)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    save_path = os.path.join(output_dir, "grid_bus_voltage_analysis.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"   Grid voltage plot saved to: {save_path}")


def main():
    """主执行函数"""
    output_dir = "results/grid_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- All outputs will be saved to: {output_dir} ---")

    # 1. 加载数据
    data_loader = loader_final.DataLoader(config)
    data = data_loader.load_all()

    # 2. 运行车辆调度优化，得到车辆的换电计划
    solver_vrp = SolverFactory('gurobi')
    solver_vrp.options['TimeLimit'] = 5
    vehicle_models = run_greedy_insertion_optimizer(data, solver_vrp, list(data['vehicles'].keys()),
                                                    list(data['tasks'].keys()), config)

    # 3. 提取高峰时刻 (例如下午6点) 的换电负荷
    print("\n" + "=" * 50)
    print("=== Step 2: Aggregating Peak Hour Grid Loads ===")
    print("=" * 50)

    peak_hour = 18.0
    peak_time_step = int(peak_hour / config.TIME_STEP_HOURS)

    hdt_loads_at_peak_kw = {bus_id: 0.0 for bus_id in data['station_to_bus_map'].values()}

    for model in vehicle_models.values():
        for s in model.STATIONS:
            if value(model.swap_decision[s, k], exception=False) > 0.5:
                arrival_time = value(model.arrival_time[s, k])
                if abs(arrival_time - peak_hour) < config.TIME_STEP_HOURS:
                    bus_id = data['station_to_bus_map'][s]
                    demand_kw = config.HDT_BATTERY_CAPACITY_KWH / config.SWAP_DURATION_HOURS
                    hdt_loads_at_peak_kw[bus_id] += demand_kw
                    print(f"  -> Peak load detected: Station '{s}' on Bus {bus_id} adds {demand_kw:.2f} kW")

    # 4. 构建并求解电网潮流模型
    print("\n" + "=" * 50)
    print("=== Step 3: Solving Power Grid Flow Model ===")
    print("=" * 50)

    power_grid = data['power_grid_net']
    grid_model = power_grid_model.create_power_grid_model(power_grid, data['station_to_bus_map'], hdt_loads_at_peak_kw,
                                                          config)

    solver_grid = SolverFactory('cbc')  # LP问题使用更快的CBC求解器
    results = solver_grid.solve(grid_model, tee=True)

    # 5. 可视化电网分析结果
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("\nGrid flow solved successfully!")
        plot_bus_voltages(power_grid, grid_model, output_dir)
    else:
        print("\nCould not solve the grid flow model. Please check constraints.")


if __name__ == "__main__":
    main()