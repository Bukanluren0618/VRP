# config.py

"""
全局配置文件
集中管理所有可调参数，便于模型优化和场景测试。
"""

# --- 基础配置 ---
SEED = 42  # 随机种子，用于保证实验结果的可复现性

# --- 路网生成参数 ---
NODE_COUNT = 100      # 虚拟城市路网中的节点总数
GRAPH_RADIUS = 0.15   # 节点连接半径，用于生成随机几何图

# --- 车辆物理参数 ---
TRUCK_CAPACITY_KWH = 200.0       # 电池总容量 (kWh)
TRUCK_CONSUMPTION_RATE = 20.0    # 基础耗电速率 (kWh/单位距离)，这里单位距离是几何距离
MAX_PAYLOAD_KG = 10000.0         # 最大载重 (kg)
TRUCK_RESERVE_SOC = 0.25         # 安全电量阈值 (25%)，电量低于此值时必须寻找换电站

# --- 能源站参数 ---
INITIAL_SWAP_BATTERIES = 8       # 每个换电站初始可用电池数量
CHARGERS_PER_STATION = 4         # 每个换电站的充电桩数量 (用于模拟EV服务)
EV_LAMBDA_PER_HOUR = 2.0         # 每个换电站每小时EV平均到达率

# --- 任务与仿真参数 ---
TRUCK_COUNT_FOR_SIM = 30         # 在纯仿真模式下生成的卡车数量
SERVICE_MIN_PER_TRUCK = 10       # 每辆卡车最少服务的任务点数量
SERVICE_MAX_PER_TRUCK = 25       # 每辆卡车最多服务的任务点数量
SIMULATION_STEPS = 48            # 仿真总步长 (例如，每步30分钟，共24小时)
TIME_WINDOW_MIN_MINUTES = 15     # 时间窗最小单位（分钟）

# --- 第一阶段：战术规划成本参数 ---
TOTAL_TRUCK_FLEET_SIZE = 50      # 公司总共拥有的重卡数量
PENALTY_PER_UNSERVED_TASK = 10000.0 # 每个未被服务的任务产生的罚款
AVG_TASKS_PER_TRUCK = 4          # [重要假设] 平均一辆卡车在一个运营周期内能完成的任务数，用于战术规划
AVG_KM_PER_TRUCK_DAY = 300       # [重要假设] 平均每辆车每天的行驶里程（公里），用于估算运营成本
COST_PER_TRUCK_KM = 5.0          # 每公里的综合运营成本（燃油、折旧、人力等）

# --- 第二阶段：运营执行 Gurobi 参数 ---
GUROBI_TIME_LIMIT = 300          # Gurobi求解器的最长运行时间（秒）