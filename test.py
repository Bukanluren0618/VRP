import random
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Gurobi 求解器
from gurobipy import Model, GRB
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False



# 配置常量
SEED = 42
TRUCK_COUNT = 30
SERVICE_MIN = 10
SERVICE_MAX = 25
STEPS = 48
TIME_WINDOW_MIN = 15
MAX_PAYLOAD_KG = 10000.0
# 重卡参数
TRUCK_CAPACITY_KWH = 200.0         # 电池总容量 (kWh)
TRUCK_CONSUMPTION_RATE = 20.0      # 耗电速率 (kWh/单位距离)

random.seed(SEED)
np.random.seed(SEED)

@dataclass
class EnergyStation:
    node: int
    swap_batteries: int = 8
    chargers: int = 4
    ev_served: int = 0
    ev_lambda: float = 2.0  # 每小时EV到达率

    def serve_hdt(self, truck) -> bool:
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


print("[MAIN] Script start")


@dataclass
class ServiceTask:
    node: int
    weight: float
    due: int


@dataclass
class Truck:
    id: int
    capacity_kwh: float
    current_kwh: float
    consumption_rate: float
    route: list[int]
    original_services: list[int]
    cargo_weight: float
    max_payload: float
    tasks: list[ServiceTask] = field(default_factory=list)

    swap_events: list[int] = field(default_factory=list)
    deliver_events: list[int] = field(default_factory=list)
    route_log: list[dict] = field(default_factory=list)


    def __post_init__(self):
        # 初始化待卸货节点集合和卸货量
        self._pending = list(self.original_services)
        self.drop_amt = self.max_payload / len(self.original_services) if self.original_services else 0.0
        # 初始化历史记录
        self.history_kwh = [self.current_kwh]
        self.history_load = [self.cargo_weight]
        init_unit = self.consumption_rate * (1 + self.cargo_weight / self.max_payload)
        self.history_unit = [init_unit]


    def reset(self):
        """重置车辆状态，清空日志、事件和历史记录。"""
        self.current_kwh = self.capacity_kwh
        self.cargo_weight = self.max_payload
        self.swap_events.clear()
        self.deliver_events.clear()
        self.route_log.clear()
        self._pending = list(self.original_services)
        # 重置历史记录
        self.history_kwh = [self.current_kwh]
        self.history_load = [self.cargo_weight]
        init_unit = self.consumption_rate * (1 + self.cargo_weight / self.max_payload)
        self.history_unit = [init_unit]

    def drive(self, G: nx.Graph, stations: dict[int, EnergyStation]) -> None:
        """
        模拟行驶过程，记录电量消耗、换电和卸货事件，
        并同步更新 history_kwh、history_load、history_unit。
        """
        reserve = 0.25 * self.capacity_kwh
        for u, v in zip(self.route[:-1], self.route[1:]):
            unit = self.consumption_rate * (1 + self.cargo_weight / self.max_payload)
            # —— 记录行驶前状态 ——
            self.history_kwh.append(self.current_kwh)
            self.history_load.append(self.cargo_weight)
            self.history_unit.append(unit)

            # 计算到 v 的需电量
            path = nx.shortest_path(G, u, v, weight='distance')
            need = sum(G.edges[a, b]['distance'] * unit for a, b in zip(path[:-1], path[1:]))

            # 判断是否需要换电
            if need > self.current_kwh or self.current_kwh - need < reserve:
                lengths = nx.single_source_dijkstra_path_length(G, u, weight='distance')
                candidates = [(s, d) for s, d in lengths.items()
                              if s in stations and s != u and d * unit <= self.current_kwh]
                if candidates:
                    swap_node = min(candidates, key=lambda x: x[1])[0]
                    self.swap_events.append(swap_node)
                    # 记录日志：开始换电
                    self.route_log.append({'节点': swap_node, '事件': '开始换电', '换电前电量': self.current_kwh})

                    # 绕行至换电站
                    detour = nx.shortest_path(G, u, swap_node, weight='distance')
                    for a, b in zip(detour[:-1], detour[1:]):
                        cost = G.edges[a, b]['distance'] * unit
                        self.current_kwh -= cost
                        # —— 记录绕行后状态 ——
                        self.history_kwh.append(self.current_kwh)
                        self.history_load.append(self.cargo_weight)
                        self.history_unit.append(unit)
                        # 日志：绕行行驶
                        self.route_log.append({
                            '起点': a, '终点': b, '事件': '绕行行驶',
                            '消耗电量': cost, '行驶后电量': self.current_kwh
                        })
                    # 执行换电
                    stations[swap_node].serve_hdt(self)
                    # —— 记录换电后状态 ——
                    self.history_kwh.append(self.current_kwh)
                    self.history_load.append(self.cargo_weight)
                    self.history_unit.append(unit)
                    # 日志：完成换电
                    self.route_log.append({'节点': swap_node, '事件': '完成换电', '换电后电量': self.current_kwh})

                    # 重算从换电站到 v 的需电
                    path = nx.shortest_path(G, swap_node, v, weight='distance')
                    need = sum(G.edges[a, b]['distance'] * unit for a, b in zip(path[:-1], path[1:]))

            # —— 行驶至 v ——
            self.current_kwh -= need
            # —— 记录行驶后状态 ——
            self.history_kwh.append(self.current_kwh)
            self.history_load.append(self.cargo_weight)
            self.history_unit.append(unit)
            # 日志：行驶完成
            self.route_log.append({
                '起点': u, '终点': v, '事件': '行驶完成',
                '消耗电量': need, '行驶后电量': self.current_kwh
            })

            # —— 卸货 ——
            if v in self._pending:
                self.deliver_events.append(v)
                self.cargo_weight = max(0.0, self.cargo_weight - self.drop_amt)
                self._pending.remove(v)
                # —— 记录卸货后状态 ——
                self.history_kwh.append(self.current_kwh)
                self.history_load.append(self.cargo_weight)
                self.history_unit.append(unit)
                # 日志：卸货完成
                self.route_log.append({
                    '节点': v, '事件': '卸货完成',
                    '卸货后载重': self.cargo_weight
                })

        return self.swap_events


# print functions


def plot_metrics(truck: Truck):
    steps = range(len(truck.history_kwh))
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(steps, truck.history_kwh, label='电量', color='tab:blue')
    ax1.set_ylabel('电量 (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.step(steps, truck.history_load, where='post', label='载重', color='tab:orange')
    ax2.set_ylabel('载重 (kg)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(steps, truck.history_unit, label='耗电率', color='tab:green')
    ax3.set_ylabel('耗电率', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    plt.xlabel('步骤')
    plt.title(f'Truck {truck.id} 电量/载重/耗电率')
    plt.tight_layout()
    plt.show()





print("[MAIN] generate graph")
def generate_connected_graph(N: int, radius: float):
    while True:
        G = nx.random_geometric_graph(N, radius)
        if nx.is_connected(G):
            break
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in G.edges():
        p, q = pos[u], pos[v]
        G.edges[u, v]['distance'] = math.hypot(p[0]-q[0], p[1]-q[1])
    return G, pos

def plan_route_greedy(G: nx.Graph, start: int, services: list[int]) -> list[int]:
    """按能耗启发式规划服务节点顺序"""
    if not services:
        return [start, start]

    remaining = set(services)
    route = [start]
    curr = start
    max_payload = MAX_PAYLOAD_KG
    drop_amt = max_payload / len(services)
    cargo = max_payload

    while remaining:
        def cost(n: int) -> float:
            d_curr = nx.shortest_path_length(G, curr, n, weight="distance")
            d_home = nx.shortest_path_length(G, n, start, weight="distance")
            energy_to_n = d_curr * (1 + cargo / max_payload)
            energy_back = d_home * (1 + (cargo - drop_amt) / max_payload)
            return energy_to_n + energy_back

        next_node = min(remaining, key=cost)
        route.append(next_node)
        curr = next_node
        remaining.remove(next_node)
        cargo -= drop_amt

    route.append(start)
    return route


def plan_route_nearest(G: nx.Graph, start: int, tasks: list[ServiceTask]) -> list[int]:
    """根据距离和载重的贪心策略规划任务顺序"""
    if not tasks:
        return [start, start]

    remaining = tasks.copy()
    curr = start
    cargo = sum(t.weight for t in tasks)
    max_payload = MAX_PAYLOAD_KG
    route = [start]

    while remaining:
        def cost(task: ServiceTask) -> float:
            dist = nx.shortest_path_length(G, curr, task.node, weight="distance")
            return dist * (1 + cargo / max_payload)

        next_task = min(remaining, key=cost)
        route.append(next_task.node)
        curr = next_task.node
        cargo -= next_task.weight
        remaining.remove(next_task)

    route.append(start)
    return route


def plan_route_with_tasks(G: nx.Graph, start: int, tasks: list[ServiceTask]) -> list[int]:
    """根据任务权重和时间窗规划路线"""
    if not tasks:
        return [start, start]

    remaining = tasks.copy()
    curr = start
    cargo = sum(t.weight for t in tasks)
    max_payload = MAX_PAYLOAD_KG
    time = 0.0
    route = [start]

    while remaining:
        def score(task: ServiceTask) -> float:
            d_curr = nx.shortest_path_length(G, curr, task.node, weight="distance")
            d_home = nx.shortest_path_length(G, task.node, start, weight="distance")
            energy_to = d_curr * (1 + cargo / max_payload)
            energy_back = d_home * (1 + (cargo - task.weight) / max_payload)
            arrival = time + d_curr
            penalty = max(0.0, arrival - task.due) * 1000
            return energy_to + energy_back + penalty

        next_task = min(remaining, key=score)
        travel = nx.shortest_path_length(G, curr, next_task.node, weight="distance")
        time += travel
        cargo -= next_task.weight
        curr = next_task.node
        route.append(curr)
        remaining.remove(next_task)

    route.append(start)
    return route



def insert_stations(route: list[int],
                    G: nx.Graph,
                    stations: list[int],
                    capacity: float,
                    rate: float) -> list[int]:
    """
    根据剩余电量动态插入换电站（允许绕行）:
    1. 展开全程真实节点 full_path;
    2. 模拟行驶并动态插入换电站:
       - 若到下个节点耗电不足以保留25%:
         a) 使用全图 Dijkstra 查找以当前电量可达的最近站点;
         b) 绕行至该站点换电;
       - 否则直接行驶;
    返回最终带有插入站点的路径列表。
    """
    # 展开完整路径
    full_path: list[int] = [route[0]]
    for u, v in zip(route[:-1], route[1:]):
        seg = nx.shortest_path(G, u, v, weight='distance')
        if full_path[-1] == seg[0]:
            full_path.extend(seg[1:])
        else:
            full_path.extend(seg)

    reserve = 0.25 * capacity
    kwh = capacity
    planned: list[int] = [full_path[0]]

    for next_node in full_path[1:]:
        u = planned[-1]
        dist = G.edges[u, next_node]['distance']
        need = dist * rate
        # 判断是否需换电
        if need > kwh or kwh - need < reserve:
            # 全图搜索最近可达站点（只需 d*rate <= kwh）
            lengths = nx.single_source_dijkstra_path_length(G, u, weight='distance')
            candidates = [
                (s, d) for s, d in lengths.items()
                if s in stations and s != u and d * rate <= kwh
            ]
            if not candidates:
                raise RuntimeError(f"无可达换电站，{u}->{next_node} 段过远")
            # 选最近站点
            s_near = min(candidates, key=lambda x: x[1])[0]
            # 绕行换电
            detour = nx.shortest_path(G, u, s_near, weight='distance')
            for a, b in zip(detour[:-1], detour[1:]):
                kwh -= G.edges[a, b]['distance'] * rate
            kwh = capacity
            planned.extend(detour[1:])
        # 正常行驶
        planned.append(next_node)
        kwh -= need
    return planned


print("[MAIN] Defining simulate()...")
def simulate():
    N, radius = 100, 0.15
    G, pos = generate_connected_graph(N, radius)
    depots = [50, 73, 41, 49, 27]
    fixed_stations = [84, 25, 22, 66, 63, 16, 58, 90, 93]
    extra = random.sample([n for n in range(1, N) if n not in fixed_stations], 0)
    station_nodes = fixed_stations + extra
    stations = {n: EnergyStation(node=n) for n in station_nodes}
    print(f"[SIM] Depots: {depots}, Stations: {station_nodes}")

    trucks = []
    candidates = [n for n in range(N) if n not in station_nodes]
    for i in range(TRUCK_COUNT):
        start = random.choice(depots)
        k = random.randint(SERVICE_MIN, min(SERVICE_MAX, len(candidates)))
        services = random.sample([n for n in candidates if n != start], k)
        tasks = [ServiceTask(node=s,
                             weight=random.uniform(200, 800),
                             due=random.randint(1, STEPS))
                 for s in services]
        route = plan_route_nearest(G, start, tasks)
        # 初始化满载
        init_weight = MAX_PAYLOAD_KG
        trucks.append(Truck(
            id=i + 1,
            capacity_kwh=TRUCK_CAPACITY_KWH,
            current_kwh=TRUCK_CAPACITY_KWH,
            consumption_rate=TRUCK_CONSUMPTION_RATE,
            route=route,
            original_services=services,  # 这里传入原始服务点列表
            cargo_weight=init_weight,
            max_payload=MAX_PAYLOAD_KG,
            tasks = tasks
        ))

    print(f"[SIM] {len(trucks)} trucks initialized")

    service_nodes = [n for t in trucks for n in t.route if n not in depots and n not in station_nodes]
    print(f"[SIM] Service nodes: {len(service_nodes)}")


    record_swaps = {}
    for t in trucks:
        print(f"  sim driving truck {t.id}")
        full_route = insert_stations(t.route, G, station_nodes, t.capacity_kwh, t.consumption_rate)
        t.route = full_route
        t.reset()
        t.drive(G, stations)
        record_swaps[t.id] = t.swap_events
    print("[MAIN] simulate() defined")

    rate_step = {n: stations[n].ev_lambda * (TIME_WINDOW_MIN/60) for n in station_nodes}
    ev_arrivals = {n: [] for n in station_nodes}
    for _ in range(STEPS):
        for n, s in stations.items():
            ev_arrivals[n].append(s.serve_ev(rate_step[n]))

    return G, pos, depots, station_nodes, stations, trucks, record_swaps, ev_arrivals


# 构建并求解混合整数线性规划（MILP）模型，使用 Gurobi 求解器
def solve_with_gurobi(G, depots, stations, service_nodes, trucks):
    # 全节点建模
    V = list(G.nodes())
    K = [t.id for t in trucks]
    # 预计算所有节点对最短距离
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='distance'))

    # 创建模型
    m = Model('vrp_swap')
    # 路径决策变量 x[k,i,j]
    x = m.addVars(K, V, V, vtype=GRB.BINARY, name='x')
    # 换电决策变量 f[k,i]
    f = m.addVars(K, V, vtype=GRB.BINARY, name='f')
    # 电量状态变量 e[k,i]
    C = TRUCK_CAPACITY_KWH
    r = TRUCK_CONSUMPTION_RATE
    e = m.addVars(K, V, lb=0, ub=C, name='e')

    # 禁止非站点换电
    for k in K:
        for i in V:
            if i not in stations:
                m.addConstr(f[k,i] == 0)

    # 目标：最小化总行驶距离
    m.setObjective(
        sum(dist[i][j] * x[k,i,j] for k in K for i in V for j in V if i!=j),
        GRB.MINIMIZE)

    # 流平衡及服务覆盖约束
    for k in K:
        # 流平衡
        for h in service_nodes + depots:
            m.addConstr(sum(x[k,i,h] for i in V if i!=h)
                        == sum(x[k,h,j] for j in V if j!=h))
    for h in service_nodes:
        # 每个服务点被且仅被访问一次
        m.addConstr(sum(x[k,i,h] for k in K for i in V if i!=h) == 1)

    # 能耗与SOC约束
    reserve = 0.25 * C
    # 定义到达SOC变量 s_arr and 离开SOC s_dep
    s_arr = m.addVars(K, V, lb=0, ub=C, name='s_arr')
    s_dep = m.addVars(K, V, lb=0, ub=C, name='s_dep')
    # 离站SOC = 到站SOC + 换电量
    for k in K:
        for i in V:
            m.addConstr(s_dep[k,i] == s_arr[k,i] + C * f[k,i])
    # 电量消耗: s_arr[j] >= s_dep[i] - r*dij*x
    for k in K:
        for i in V:
            for j in V:
                if i!=j:
                    m.addConstr(
                        s_arr[k,j] >= s_dep[k,i] - r*dist[i][j]*x[k,i,j]
                    )
    # 保留电量约束
    for k in K:
        for i in V:
            m.addConstr(s_arr[k,i] >= reserve)
    # 初始电量为满
    for k in K:
        start = random.choice(depots)
        m.addConstr(s_arr[k,start] == C)

    # 换电站容量约束
    for i in stations:
        m.addConstr(sum(f[k,i] for k in K) <= stations[i].swap_batteries)

    # 求解
    m.optimize()
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS(); m.write('model.ilp')
        raise RuntimeError('模型不可行，请检查约束')

    # 提取结果
    routes = {}
    for k in K:
        sel = []
        for i in V:
            for j in V:
                if i!=j and x[k,i,j].Xn > 0.5:
                    sel.append((i,j))
        routes[k] = sel
    return routes


def visualize(G, pos, depots, stations, trucks, record_swaps, ev_arrivals):
    fig, ax = plt.subplots(figsize=(12,9))
    nx.draw(G, pos=pos, ax=ax, node_size=15, edge_color='lightgray', alpha=0.3)
    nx.draw_networkx_labels(G, pos, {n:str(n) for n in G.nodes()}, font_size=4, ax=ax)
    for d in depots:
        ax.scatter(*pos[d], c='gold', marker='*', s=150, edgecolors='black', label=f'Depot {d}')
    sx = [pos[n][0] for n in stations]; sy = [pos[n][1] for n in stations]
    ax.scatter(sx, sy, c='orange', marker='s', s=80, label='Station')
    cmap = plt.get_cmap('hsv', TRUCK_COUNT)
    for i, t in enumerate(trucks):
        xs, ys = zip(*(pos[n] for n in t.route))
        ax.plot(xs, ys, color=cmap(i), linewidth=0.8, alpha=0.6, label=f'Truck {t.id}', zorder=2)
        for s in record_swaps[t.id]:
            ax.scatter(*pos[s], facecolors='none', edgecolors='red', s=150, marker='o', zorder=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=5, ncol=3)
    ax.set_title('HDT Routes & EV Service')
    plt.tight_layout(); plt.show()

    print("\nTruck Specs and Routes:")
    for t in trucks:
        print(f"Truck {t.id}: Cap {t.capacity_kwh} kWh, Cons {t.consumption_rate} kWh/unit, Route {t.route}")
    print("\nSwap events:")
    for tid, sw in record_swaps.items(): print(f"Truck {tid} swapped at: {sw}")
    print("\nEV arrivals per station:")
    for n, arr in ev_arrivals.items(): print(f"Station {n}: {arr}")

# def plot_network_with_stations(G, pos, station_nodes):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     # 画所有边
#     nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5)
#     # 画普通节点
#     nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color='lightblue', label='Node')
#     # 高亮换电站
#     nx.draw_networkx_nodes(
#         G, pos,
#         nodelist=station_nodes,
#         ax=ax,
#         node_size=150,
#         node_color='red',
#         node_shape='s',
#         label='Station'
#     )
#     # 加节点标签
#     nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
#     ax.set_title('Road Network with Stations Highlighted')
#     ax.axis('off')
#     ax.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()

def plot_single_truck(G: nx.Graph,
                      pos: dict[int, tuple[float, float]],
                      depots: list[int],
                      station_nodes: list[int],
                      trucks: list[Truck],
                      truck_id: int) -> None:
    """绘制指定卡车的行驶轨迹"""
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos=pos, ax=ax, node_size=10,
            edge_color='lightgray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=depots,
                           node_color='gold', node_shape='*',
                           node_size=150, edgecolors='black',
                           label='Depot')
    nx.draw_networkx_nodes(G, pos, nodelist=station_nodes,
                           node_color='orange', node_shape='s',
                           node_size=80, label='Station')
    truck = next(t for t in trucks if t.id == truck_id)
    xs, ys = zip(*(pos[n] for n in truck.route))
    ax.plot(xs, ys, '-o', linewidth=2, markersize=5,
            label=f'Truck {truck_id}')
    ax.set_title(f'Trajectory of Truck {truck_id}')
    ax.legend()
    plt.tight_layout()
    plt.show()





def print_route_log(truck: Truck):
    for entry in truck.route_log:
        print(entry)

def print_events(truck: Truck):
    print(f"Truck {truck.id} 换电节点：{truck.swap_events}")
    print(f"Truck {truck.id} 卸货节点：{truck.deliver_events}")



def print_service_tasks(trucks: list[Truck]):
    for t in trucks:
        info = ", ".join(
            f"{tsk.node}(w={tsk.weight:.0f}, due={tsk.due})" for tsk in t.tasks
        )
        print(f"Truck {t.id} tasks: {info}")



if __name__ == '__main__':
    G, pos, depots, station_nodes, stations, trucks, record_swaps, ev_arrivals = simulate()
    print_service_tasks(trucks)
    service_nodes = [n for t in trucks for n in t.route if n not in depots and n not in station_nodes]
    vrp_routes = solve_with_gurobi(G, depots, stations, service_nodes, trucks)
    visualize(G, pos, depots, stations, trucks, record_swaps, ev_arrivals)

    # **先重置并跑一遍第15辆车，填充 history 和 route_log**
    truck15 = next(t for t in trucks if t.id == 15)
    truck15.reset()
    truck15.drive(G, stations)

    # 打印第15辆车的完整节点序列（包含所有中间转弯和换电站）
    print(f"Truck 15 完整路径节点序列：{truck15.route}")

    # 打印日志 & 事件
    print_route_log(truck15)
    print_events(truck15)

    # 最后再画轨迹和三坐标图
    plot_single_truck(G, pos, depots, station_nodes, trucks, 15)
    plot_metrics(truck15)

