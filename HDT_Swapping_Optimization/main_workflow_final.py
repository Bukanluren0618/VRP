# HDT_Swapping_Optimization/main_workflow_final.py

import datetime
from pyomo.environ import *
from src.data_processing import loader_final
from src.modeling import model_final
from src.common import config_final as config
import pandas as pd


def run_final_workflow():
    """
    执行最终的、完整的、保证有解的物流-能源协同优化工作流。
    """
    start_time = datetime.datetime.now()
    print(f"最终版工作流启动: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 创建场景数据
    model_data = loader_final.create_final_scenario()

    # 创建并求解模型
    model = model_final.create_final_model(model_data)

    # # 使用旧版Pyomo兼容的方式调用Gurobi
    # solver_name = 'gurobi_direct'
    # print(f"\n准备求解器: {solver_name} (强制使用Python直接接口)")
    solver_name = config.SOLVER_NAME
    if solver_name == 'gurobi':
        solver_name = 'gurobi_direct'
        print(f"\n准备求解器: {solver_name} (强制使用Python直接接口)")
    else:
        print(f"\n准备求解器: {solver_name}")
    solver = SolverFactory(solver_name)
    solver.options['timelimit'] = config.TIME_LIMIT_SECONDS

    print("开始求解...")
    results = solver.solve(model, tee=True)

    # 结果分析
    print("\n" + "=" * 60)
    print("               最终版协同优化结果分析")
    print("=" * 60)

    if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible,
                                                TerminationCondition.maxTimeLimit]:
        print(f"求解成功或达到时间限制！状态: {results.solver.termination_condition}")
        print(f"\n[最终目标] 最小化综合成本: {value(model.objective):.2f} 元")

        print("\n--- [诊断] 任务完成情况 ---")
        for t in model.TASKS:
            if value(model.is_task_unassigned[t]) > 0.5:
                print(f"  - 任务 '{t}' 被放弃，产生了 {config.UNASSIGNED_TASK_PENALTY} 的罚款。")
            else:
                customer = model_data['tasks'][t]['delivery_to']
                assigned_vehicle = next(
                    (k for k in model.VEHICLES if value(model.y[customer, k]) > 0.5),
                    "未知车辆")
                delay = value(model.task_delay[t, assigned_vehicle])
                if delay > 1e-6:
                    print(f"  - 任务 '{t}' 由车辆 '{assigned_vehicle}' 完成，延迟 {delay:.2f} 小时。")
                else:
                    print(f"  - 任务 '{t}' 由车辆 '{assigned_vehicle}' 成功按时完成。")

        # 详细的结果分析与可视化
        from src.analysis import post_analysis
        print("\n正在生成详细的可视化结果...")
        post_analysis.plot_road_network_with_routes(model, model_data)
        post_analysis.plot_station_energy_schedule(model, model_data)
        post_analysis.print_task_demands(model_data)
        for v in model.VEHICLES:
            post_analysis.plot_vehicle_metrics(model, model_data, v)
    else:
        print(f"\n求解失败。状态: {results.solver.termination_condition}")

    end_time = datetime.datetime.now()
    print("\n" + "=" * 60)
    print(f"工作流结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time}")


if __name__ == '__main__':
    run_final_workflow()