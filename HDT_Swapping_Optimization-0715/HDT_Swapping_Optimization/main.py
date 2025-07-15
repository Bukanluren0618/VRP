# main.py
import sys
import os
from pyomo.environ import SolverFactory, value

# 将src目录添加到Python路径，以便可以导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from common import config_final as config
from data_processing import loader_final
from modeling import model_final
from analysis import post_analysis

def main():
    """
    主工作流函数：加载数据 -> 构建模型 -> 求解 -> 后分析
    """
    print("======================================================")
    print("=== HDT换电与配电网耦合优化模型 - 主程序入口 ===")
    print("======================================================")

    # 1. 加载场景数据
    try:
        model_data = loader_final.create_final_scenario()
    except Exception as e:
        print(f"\n[错误] 数据加载失败: {e}")
        return

    # 2. 构建优化模型
    try:
        model = model_final.create_final_model(model_data)
    except Exception as e:
        print(f"\n[错误] 模型构建失败: {e}")
        return

    # 3. 创建并配置求解器 (IPOPT)
    # 重要提示：您必须已经在您的系统中安装了IPOPT可执行文件，并将其添加到了系统路径。
    print(f"\n正在创建求解器: {config.SOLVER_NAME}")
    try:
        solver = SolverFactory(config.SOLVER_NAME)
        # solver.options['max_iter'] = 2000 # 增加最大迭代次数
        # solver.options['tol'] = 1e-6 # 设置收敛容差
        solver.options['max_iter'] = 2000  # 增加最大迭代次数
        solver.options['tol'] = 1e-6  # 设置收敛容差
        if config.SOLVER_NAME.lower() == 'gurobi':
            solver.options['NonConvex'] = 2
            if hasattr(config, 'TIME_LIMIT_SECONDS'):
                solver.options['TimeLimit'] = config.TIME_LIMIT_SECONDS

    except Exception as e:
        print(f"\n[错误] 创建求解器 '{config.SOLVER_NAME}' 失败。请确保它已正确安装并配置。")
        print(f"详细信息: {e}")
        return

    # 4. 求解模型
    print("开始求解模型，这可能需要一些时间...")
    try:
        results = solver.solve(model, tee=True) # tee=True 会在控制台显示求解器日志
    except Exception as e:
        print(f"\n[错误] 模型求解过程中发生异常: {e}")
        return

    # 5. 检查结果并进行后分析
    if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal' or results.solver.termination_condition == 'locallyOptimal'):
        print("\n[成功] 模型求解成功！找到了一个（局部）最优解。")
        print(f"目标函数值 (总成本): {value(model.objective):.2f}")

        # 调用后分析脚本来绘制图表
        print("\n正在生成结果图表...")
        try:
            post_analysis.plot_road_network_with_routes(model, model_data)
            post_analysis.plot_station_energy_schedule(model, model_data)
            print("所有图表已成功生成并保存到 'results' 文件夹。")
        except Exception as e:
            print(f"\n[警告] 后分析绘图失败: {e}")

    else:
        print("\n[失败] 未能找到最优解。")
        print(f"  - 求解器状态: {results.solver.status}")
        print(f"  - 终止条件: {results.solver.termination_condition}")
        print("\n建议：")
        print("1. 检查求解器日志输出，寻找'infeasible'或'unbounded'等关键词。")
        print("2. 运行 'debug_hdt.py' 脚本来对HDT相关约束进行自动化调试。")

if __name__ == "__main__":
    main()