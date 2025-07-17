# debug_hdt.py

import sys
import os
from pyomo.environ import SolverFactory
import random

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common import config_final as config
from src.data_processing import loader_final
from src.modeling import model_final as stage2_model_builder


def debug_runner():
    """
    自动化调试器，专注解决第二阶段HDT路径、换电和物理状态传播问题。
    """
    print("======================================================")
    print("=== HDT相关问题自动化调试器 (使用IPOPT) ===")
    print("======================================================")

    print("[调试] 正在准备测试用的环境和数据...")
    try:
        model_data = loader_final.create_scenario_from_files()
        if model_data is None: return
        solver = SolverFactory(config.SOLVER_NAME)
        if config.SOLVER_NAME.lower() == 'gurobi':
            solver.options['NonConvex'] = 2
            if hasattr(config, 'TIME_LIMIT_SECONDS'):
    except Exception as e:
        print(f"[致命错误] 无法初始化调试环境: {e}")
        return

    # 使用从数据加载的车辆和任务进行测试
    test_vehicles = list(model_data['vehicles'].keys())[:1]  # 只用一辆车测试
    test_tasks = list(model_data['tasks'].keys())[:2]  # 只用两个任务测试
    if not test_vehicles or not test_tasks:
        print("[错误] 无法从数据中获取足够的车辆或任务进行测试。")
        return

    print(f"[调试] 将使用车辆 {test_vehicles} 和任务 {test_tasks} 进行测试。")

    constraint_groups_to_test = [
        "hdt_routing",
        "hdt_time",
        "hdt_weight",
        "hdt_soc",
    ]

    for i in range(len(constraint_groups_to_test) + 1):
        enabled_groups = constraint_groups_to_test[:i]
        deactivated_groups = constraint_groups_to_test[i:]

        print(f"\n--- [调试轮次 {i + 1}/{len(constraint_groups_to_test) + 1}] ---")
        if enabled_groups:
            print(f"  本轮启用的约束组: {enabled_groups}")
        else:
            print("  本轮无HDT核心约束 (仅测试基础模型结构)")

        model = stage2_model_builder.create_operational_model(
            model_data,
            vehicle_ids=test_vehicles,
            task_ids=test_tasks,
            deactivated_constraints=deactivated_groups
        )

        try:
            results = solver.solve(model, tee=False)
        except Exception as e:
            print(f"\n[调试结果] 在添加 '{constraint_groups_to_test[i - 1]}' 约束组后，求解器发生严重错误！")
            print(f"错误详情: {e}")
            return

        if (results.solver.status == 'ok') and (results.solver.termination_condition in ['optimal', 'locallyOptimal']):
            print(f"[调试结果] -> 成功: 当前约束组合下模型可解。")
            continue
        else:
            problematic_group = "基础模型" if i == 0 else constraint_groups_to_test[i - 1]
            print("\n" + "=" * 60)
            print("!!! [调试器发现问题] !!!")
            print(f"在添加 '{problematic_group}' 约束组后，模型变得不可解或求解失败。")
            print(f"!!! 问题根源极有可能就在这个约束组中。!!!")
            print("=" * 60)
            return

    print("\n=============================================")
    print("✓✓✓ 调试完成 ✓✓✓")
    print("=============================================")


if __name__ == '__main__':
    debug_runner()