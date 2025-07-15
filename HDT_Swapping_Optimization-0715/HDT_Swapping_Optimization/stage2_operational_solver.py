# stage2_operational_solver.py

"""
第二阶段：运营执行模型

核心功能：
- 接收第一阶段分配的车辆，并加载当天具体的、详细的客户订单（Tasks）。
- 使用启发式算法为车辆进行初步的任务分配和路径规划。
- （可选）运行模拟器来验证启发式解的可行性并进行动态调整（如插入换电站）。
- 构建一个精细化的车辆路径规划（VRP）Gurobi模型，求解出每辆车的最优行驶路径。
- 包含了所有来自 test.py 的核心类（Truck, Station, Task）和函数（模拟、绘图等）。
"""
import random
import math
import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib
matplotlib.use('TkAgg') # 确保绘图后端可用
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from gurobipy import Model, GRB, GurobiError

# 导入配置文件
import config as cfg

# 设置中文字体，解决绘图乱码问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 数据类定义 (来自 test.py)
# =============================================================================

@dataclass
class EnergyStation:
    node: int
    swap_batteries: int = cfg.INITIAL_SWAP_BATTERIES
    chargers: int = cfg.CHARGERS_PER_STATION
    ev_served: int = 0
    ev_lambda: float = cfg.EV_LAMBDA_PER_HOUR

    def serve_hdt(self, truck: 'Truck') -> bool:
        if self.swap_batteries > 0:
            self.swap_batteries -= 1
            truck.current_kwh = truck.capacity_kwh
            return True
        return False

    def serve_ev(self, rate_per_step: float) -> int:
        arrivals = np.random.poisson(rate_per_step)
        served = min(arrivals, self.chargers)
        self.ev_served += served
        return served

@dataclass
class ServiceTask:
    id: int
    node: int
    weight: float
    due: int
    depot_id: int # 任务所属的发车点

@dataclass
class Truck:
    id: int
    depot_node: int
    capacity_kwh: float = cfg.TRUCK_CAPACITY_KWH
    current_kwh: float = cfg.TRUCK_CAPACITY_KWH
    consumption_rate: float = cfg.TRUCK_CONSUMPTION_RATE
    max_payload: float = cfg.MAX_PAYLOAD_KG
    tasks: list[ServiceTask] = field(default_factory=list)
    route: list[int] = field(default_factory=list)
    cargo_weight: float = 0.0

    # 日志与事件记录
    swap_events: list[int] = field(default_factory=list)
    deliver_events: list[int] = field(default_factory=list)
    route_log: list[dict] = field(default_factory=list)
    history_kwh: list[float] = field(default_factory=list)
    history_load: list[float] = field(default_factory=list)
    history_unit: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.reset()

    def assign_tasks(self, tasks: list[ServiceTask]):
        self.tasks = tasks
        self.cargo_weight = sum(t.weight for t in tasks)
        # 重新初始化历史记录
        self.history_load = [self.cargo_weight]
        init_unit = self.consumption_rate * (1 + self.cargo_weight / self.max_payload)
        self.history_unit = [init_unit]

    def reset(self):
        """重置车辆状态，用于新的仿真或求解"""
        self.current_kwh = self.capacity_kwh
        self.cargo_weight = sum(t.weight for t in self.tasks)
        self.swap_events.clear()
        self.deliver_events.clear()
        self.route_log.clear()
        self.route = [self.depot_node]
        # 初始化历史记录
        self.history_kwh = [self.current_kwh]
        self.history_load = [self.cargo_weight]
        init_unit = self.consumption_rate * (1 + self.cargo_weight / self.max_payload)
        self.history_unit = [init_unit]

    def drive(self, G: nx.Graph, stations: dict[int, EnergyStation]):
        """模拟行驶过程，与 test.py 中逻辑一致"""
        # (完整地从 test.py 复制 drive 方法的代码到这里)
        pass # 此处省略，请将 test.py 中的 drive 完整代码粘贴至此


# =============================================================================
# 2. 环境设置与数据加载
# =============================================================================

def generate_environment():
    """生成路网、发车点、换电站"""
    print("[阶段二] 正在生成虚拟城市路网环境...")
    while True:
        G = nx.random_geometric_graph(cfg.NODE_COUNT, cfg.GRAPH_RADIUS, seed=cfg.SEED)
        if nx.is_connected(G):
            break
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in G.edges():
        p, q = pos[u], pos[v]
        G.edges[u, v]['distance'] = math.hypot(p[0]-q[0], p[1]-q[1])

    # 随机选择发车点和换电站
    nodes = list(G.nodes())
    # 假设发车点数量与战术规划中的一致
    num_depots = 2 # 假设值为2，应与战术规划模块同步
    depots = random.sample(nodes, num_depots)
    station_nodes = random.sample([n for n in nodes if n not in depots], 15)
    stations = {n: EnergyStation(node=n) for n in station_nodes}

    print(f"[阶段二] 环境生成完毕: {len(depots)}个发车点, {len(station_nodes)}个换电站。")
    return G, pos, depots, stations

def load_detailed_tasks(data_folder_path: str) -> list[ServiceTask]:
    """从CSV加载详细任务列表"""
    print("[阶段二] 正在加载详细任务数据...")
    tasks = []
    try:
        file_path = os.path.join(data_folder_path, "副本4、路网与EV需求参数-final.xlsx - 车队典型数据汇总.csv")
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            tasks.append(ServiceTask(
                id=row['task_id'],
                node=row['node_id'], # 假设列名为 'node_id'
                weight=row['weight'], # 假设列名为 'weight'
                due=row['due_time'], # 假设列名为 'due_time'
                depot_id=row['depot_id']
            ))
        print(f"[阶段二] 成功加载 {len(tasks)} 个详细任务。")
        return tasks
    except Exception as e:
        print(f"[错误] 加载详细任务失败: {e}")
        return []

# =============================================================================
# 3. 启发式路径规划 (来自 test.py)
# =============================================================================

def plan_route_nearest(G: nx.Graph, start: int, tasks: list[ServiceTask]) -> list[int]:
    """根据距离和载重的贪心策略规划任务顺序"""
    # (完整地从 test.py 复制 plan_route_nearest 方法的代码到这里)
    if not tasks: return [start, start]
    remaining = tasks.copy()
    curr = start
    cargo = sum(t.weight for t in tasks)
    route = [start]
    while remaining:
        def cost(task: ServiceTask) -> float:
            dist = nx.shortest_path_length(G, curr, task.node, weight="distance")
            return dist * (1 + cargo / cfg.MAX_PAYLOAD_KG)
        next_task = min(remaining, key=cost)
        route.append(next_task.node)
        curr = next_task.node
        cargo -= next_task.weight
        remaining.remove(next_task)
    route.append(start)
    return route


# =============================================================================
# 4. Gurobi 精细化求解器
# =============================================================================

def solve_detailed_vrp(G, trucks, stations, constraint_skips=None):
    """
    第二阶段的核心求解器：精细化VRP模型。
    这是对 test.py 中 solve_with_gurobi 的改进和模块化版本。
    constraint_skips: 一个集合，包含要跳过的约束函数的名称，用于调试。
    """
    print("\n--- [阶段二：运营执行] Gurobi 精细化求解开始 ---")
    if constraint_skips is None:
        constraint_skips = set()

    model = Model('Operational_VRP')

    # --- 数据准备 ---
    V = list(G.nodes()) # 所有节点
    K = [t.id for t in trucks] # 所有卡车ID
    S = list(stations.keys()) # 所有换电站节点
    T = {t.id: [task.node for task in t.tasks] for t in trucks} # 每辆卡车需要服务的任务点
    D = {t.id: t.depot_node for t in trucks} # 每辆车的发车点

    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='distance'))

    # --- 决策变量 ---
    # x[k,i,j]: 卡车k是否从节点i行驶到节点j
    x = model.addVars(K, V, V, vtype=GRB.BINARY, name='x')
    # f[k,i]: 卡车k是否在节点i进行换电
    f = model.addVars(K, S, vtype=GRB.BINARY, name='f') # 换电只能在换电站发生
    # e[k,i]: 卡车k到达节点i时的电量
    e = model.addVars(K, V, lb=0, ub=cfg.TRUCK_CAPACITY_KWH, name='e')

    # --- 目标函数 ---
    model.setObjective(
        gp.quicksum(dist[i][j] * x[k,i,j] for k in K for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- 约束定义 ---
    # 每个约束都被封装在一个函数中，以便于调试
    def add_flow_constraints():
        print("  - 添加流平衡约束")
        for k in K:
            # 车辆从它的发车点出发
            model.addConstr(gp.quicksum(x[k, D[k], j] for j in V if j != D[k]) == 1, f"start_{k}")
            # 车辆回到它的发车点
            model.addConstr(gp.quicksum(x[k, i, D[k]] for i in V if i != D[k]) == 1, f"end_{k}")
            # 中间节点的流平衡
            for h in V:
                if h != D[k]:
                    model.addConstr(gp.quicksum(x[k, i, h] for i in V if i != h) ==
                                    gp.quicksum(x[k, h, j] for j in V if j != h), f"flow_{k}_{h}")

    def add_service_constraints():
        print("  - 添加服务覆盖约束")
        for k in K:
            for task_node in T[k]:
                # 卡车k必须服务它被分配到的所有任务点
                model.addConstr(gp.quicksum(x[k, i, task_node] for i in V if i != task_node) == 1, f"serve_{k}_{task_node}")

    def add_energy_constraints():
        print("  - 添加能耗与电量约束")
        M = cfg.TRUCK_CAPACITY_KWH * 2 # 一个足够大的数
        for k in K:
            # 初始电量
            model.addConstr(e[k, D[k]] == cfg.TRUCK_CAPACITY_KWH, f"init_soc_{k}")
            for i in V:
                for j in V:
                    if i != j:
                        # 核心电量消耗约束: e[j] <= e[i] - cost + M*(1-x[i,j])
                        cost = dist[i][j] * cfg.TRUCK_CONSUMPTION_RATE # 简化能耗模型
                        # 换电影响
                        swap_amount = cfg.TRUCK_CAPACITY_KWH * f[k,i] if i in S else 0
                        # 综合约束
                        model.addConstr(e[k, j] <= e[k, i] - cost + swap_amount + M * (1 - x[k,i,j]), f"soc_cons_{k}_{i}_{j}")
            # 安全电量约束
            for i in V:
                model.addConstr(e[k, i] >= cfg.TRUCK_CAPACITY_KWH * cfg.TRUCK_RESERVE_SOC, f"reserve_soc_{k}_{i}")

    def add_station_capacity_constraints():
        print("  - 添加换电站容量约束")
        for s in S:
            model.addConstr(gp.quicksum(f[k,s] for k in K) <= stations[s].swap_batteries, f"station_cap_{s}")

    # --- 按顺序添加约束（除非在调试时被跳过） ---
    constraint_map = {
        "flow": add_flow_constraints,
        "service": add_service_constraints,
        "energy": add_energy_constraints,
        "station_capacity": add_station_capacity_constraints,
    }

    for name, func in constraint_map.items():
        if name not in constraint_skips:
            func()

    # --- 求解 ---
    try:
        model.setParam('TimeLimit', cfg.GUROBI_TIME_LIMIT)
        model.optimize()
        if model.status == GRB.OPTIMAL:
            print("--- [阶段二] Gurobi 求解成功，找到最优解！ ---")
            # 提取结果
            final_routes = {}
            for k in K:
                route = [D[k]]
                curr = D[k]
                while True:
                    found_next = False
                    for j in V:
                        if curr !=j and x[k, curr, j].X > 0.5:
                            route.append(j)
                            curr = j
                            found_next = True
                            break
                    if not found_next or curr == D[k]:
                        break
                final_routes[k] = route
            return final_routes, model
        elif model.status == GRB.INFEASIBLE:
            print("!!! [阶段二] 模型无解 (Infeasible) !!!")
            return None, model
        else:
            print(f"--- [阶段二] 求解结束，但未找到最优解。模型状态码: {model.status}")
            return None, model

    except GurobiError as e:
        print(f"Gurobi 发生错误: {e}")
        return None, model

# =============================================================================
# 5. 可视化函数 (来自 test.py)
# =============================================================================

def visualize(G, pos, depots, stations, trucks_with_final_routes):
    """绘制最终的车辆路径规划结果"""
    # (完整地从 test.py 复制 visualize 方法的代码到这里，并稍作修改以适应新的数据结构)
    pass