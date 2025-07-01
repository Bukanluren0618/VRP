HDT Swapping Optimization
本项目提供了一个针对重卡（HDT）与社会电动车（EV）的换电‑充电站运营与路线调度一体化优化框架。
主要依赖 Pyomo/Gurobi 进行混合整数规划建模，兼顾车辆行驶、电量消耗、换电站电池库存以及 EV 充电需求等多方面因素。

目录结构
HDT_Swapping_Optimization/
│
├── src/                   # 代码模块
│   ├── data_processing/   # 场景数据构建
│   ├── modeling/          # Pyomo 模型
│   ├── simulation/        # 路网仿真等
│   └── analysis/          # 后处理与可视化
├── data/                  # 原始数据（CSV）
├── results/               # 求解结果及生成的图表
└── main_workflow_final.py # 主工作流
环境依赖
Python 3.8 及以上

主要库：pandas、numpy、networkx、matplotlib、pyomo

求解器：建议使用 Gurobi (或 CPLEX)，需自行安装并配置许可

可以通过以下命令安装所需 Python 包：

pip install pandas numpy networkx matplotlib pyomo
如何运行
准备数据
将充电负荷与电池 SOC 等 CSV 文件放入 data/raw/ 目录，文件名与 src/common/config_final.py 中的配置保持一致。

执行主工作流

python main_workflow_final.py
脚本会依次：

创建完整的仿真场景

构建并求解混合整数线性规划模型

自动生成路径规划和能源调度图表，保存在 results/ 目录

查看结果
求解完成后，在终端可看到目标值和任务完成情况，results/ 目录下会生成如 hdt_routing_plan.png、energy_schedule_SwapStation1.png 等可视化文件。

配置说明
所有参数（时间步长、车队规模、电价等）集中在 src/common/config_final.py 中，可按需要调整。

若求解规模过大，可在 config_final.py 中适当缩短时间范围或调小车辆/任务数量。

参考
该框架基于 Python 和 Pyomo 构建，旨在研究充换电站与重卡车队的协同优化问题。示例数据、模型和分析脚本均在 HDT_Swapping_Optimization 子目录下。
若需了解模型细节，可查阅 src/modeling/model_final.py 与 src/data_processing/loader_final.py 等文件中的注释。

