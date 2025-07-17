# main_two_stage.py

import sys

import os
from pyomo.environ import SolverFactory, value
import pandas as pd
import math

# 将项目根目录添加到Python路径，使得可以通过 `import src` 访问
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.common import config_final as config
# 【修改】导入我们重写后的 loader_final
from src.data_processing import loader_final
from src.modeling import stage1_tactical
from src.modeling import model_final as stage2_model_builder
from src.analysis import post_analysis
from src.analysis import pre_checks


def run_stage1(data, solver):
    """运行第一阶段，决定车辆和任务的分配"""
    # (此函数与上一版完全相同，无需修改)
    print("\n" + "=" * 50)
    print("=== RUNNING STAGE 1: TACTICAL ALLOCATION ===")
    print("=" * 50)
    model_s1 = stage1_tactical.create_tactical_model(data)
    results_s1 = solver.solve(model_s1, tee=False)
    if (results_s1.solver.status == 'ok') and (
            results_s1.solver.termination_condition in ['optimal', 'locallyOptimal']):
        print("[S1-SUCCESS] 战术规划求解成功！")
        allocated_trucks = 0
        for d in model_s1.DEPOTS:
            allocated_trucks += math.ceil(value(model_s1.n_trucks[d]))
        print(f"  -> 决策: 总共分配 {allocated_trucks} 辆车。")
        served_tasks = []
        unserved_tasks = []
        for t in model_s1.TASKS:
            if value(model_s1.is_task_served[t]) > 0.5:
                served_tasks.append(t)
            else:
                unserved_tasks.append(t)
        print(f"  -> 决策: 服务 {len(served_tasks)} 个任务: {served_tasks}")
        if unserved_tasks:
            print(f"  -> 警告: {len(unserved_tasks)} 个任务因运力不足被放弃: {unserved_tasks}")
        available_vehicles = list(data['vehicles'].keys())
        vehicle_ids_to_plan = available_vehicles[:min(allocated_trucks, len(available_vehicles))]
        return vehicle_ids_to_plan, served_tasks
    else:
        print("[S1-FAILURE] 战术规划求解失败。")
        print(f"  - 求解器状态: {results_s1.solver.status}")
        print(f"  - 终止条件: {results_s1.solver.termination_condition}")
        return None, None


def run_stage2(data, solver, vehicle_ids, task_ids):
    """为指定的车辆和任务运行第二阶段，进行精细化路径规划"""
    # (此函数与上一版完全相同，无需修改)
    print("\n" + "=" * 50)
    print("=== RUNNING STAGE 2: OPERATIONAL EXECUTION ===")
    print("=" * 50)
    if not vehicle_ids or not task_ids:
        print("[S2-SKIP] 没有车辆或任务被分配，跳过运营执行阶段。")
        return None
    model_s2 = stage2_model_builder.create_operational_model(data, vehicle_ids, task_ids)
    results_s2 = solver.solve(model_s2, tee=True)
    if (results_s2.solver.status == 'ok') and (
            results_s2.solver.termination_condition in ['optimal', 'locallyOptimal']):
        print("[S2-SUCCESS] 运营执行模型求解成功！")
        return model_s2
    else:
        print("[S2-FAILURE] 运营执行模型求解失败。")
        print(f"  - 求解器状态: {results_s2.solver.status}")
        print(f"  - 终止条件: {results_s2.solver.termination_condition}")
        print("  -> 建议运行 debug_hdt.py 对第二阶段模型进行调试。")
        return None


def main():
    """
    执行完整的两阶段决策流程 (使用手动定义的硬编码数据)
    """
    print("======================================================")
    print("=== HDT两步走策略优化模型 (使用最终手动定义数据) ===")
    print("======================================================")

    # 1. 初始化：加载数据和求解器
    try:
        # 【关键修改】调用新的数据加载函数
        model_data = loader_final.create_scenario_from_manual_data()
        if model_data is None:
            print("\n流程终止：数据加载失败。")
            return

        # 在求解前先进行任务可达性检查，打印往返距离、时间和耗电量
        pre_checks.check_task_feasibility(model_data)

        solver = SolverFactory(config.SOLVER_NAME)
        # Gurobi needs the NonConvex parameter when the model has
        # quadratic expressions that are not positive semi-definite.
        if config.SOLVER_NAME.lower() == 'gurobi':
            solver.options['NonConvex'] = 2
            # Respect optional time limit if defined in the config
            if hasattr(config, 'TIME_LIMIT_SECONDS'):
                solver.options['TimeLimit'] = config.TIME_LIMIT_SECONDS
    except Exception as e:
        print(f"\n[致命错误] 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 运行第一阶段 (代码不变)
    vehicle_ids, task_ids = run_stage1(model_data, solver)

    # 3. 运行第二阶段 (代码不变)
    final_model = None
    if vehicle_ids and task_ids:
        final_model = run_stage2(model_data, solver, vehicle_ids, task_ids)
    else:
        print("\n流程终止，因为第一阶段未能成功分配资源。")
        return

    # 4. 后分析与可视化 (代码不变)
    if final_model:
        print("\n" + "=" * 50)
        print("=== POST ANALYSIS & VISUALIZATION ===")
        print("=" * 50)
        print(f"最终总成本: {value(final_model.objective):.2f}")
        try:
            post_analysis.plot_road_network_with_routes(final_model, model_data)
            post_analysis.print_vehicle_swap_nodes(final_model)
            post_analysis.print_vehicle_routes(final_model, model_data)
            post_analysis.plot_station_energy_schedule(final_model, model_data)
            print("所有图表已成功生成并保存到 'results' 文件夹。")
        except Exception as e:
            print(f"\n[警告] 后分析绘图失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n流程结束，但未产生可供分析的最终解。")


if __name__ == "__main__":
    main()