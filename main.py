from dataclasses import dataclass
import networkx as nx
import random
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 配置常量
SEED = 42
TRUCK_COUNT = 30
SERVICE_MIN = 10
SERVICE_MAX = 25
STEPS = 48
TIME_WINDOW_MIN = 15
# 重卡参数
TRUCK_CAPACITY_KWH = 200.0         # 电池总容量 (kWh)         # 电池总容量 (kWh)
TRUCK_CONSUMPTION_RATE = 5.0     # 耗电速率 (kWh/单位距离)     # 耗电速率 (kWh/单位距离)

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
        # HDT到站换电，重置电量
        if self.swap_batteries > 0:
            self.swap_batteries -= 1
            truck.current_kwh = truck.capacity_kwh
            return True
        return False

    def serve_ev(self, rate_per_step: float) -> int:
        # EV到站充电，受充电桩数限制
        arrivals = np.random.poisson(rate_per_step)
        served = min(arrivals, self.chargers)
        self.ev_served += served
        return served

@dataclass
class Truck:
    id: int
    capacity_kwh: float
    current_kwh: float
    consumption_rate: float
    route: list[int]

    def drive(self, G, stations) -> list[int]:
        # 按已规划的路线行驶，在需要时换电并记录
        swaps = []
        for u, v in zip(self.route[:-1], self.route[1:]):
            # 若到达 u 是换电站且当前电量不足以完成后续规划，先换电
            if u in stations:
                if stations[u].serve_hdt(self):
                    swaps.append(u)
            # 行驶 u->v 消耗电量
            dist = G.edges[u, v]['distance']
            need = dist * self.consumption_rate
            if need > self.current_kwh:
                raise RuntimeError(f"Truck {self.id} 电量不足，停在节点 {u}")
            self.current_kwh -= need
        return swaps


def generate_connected_graph(N: int, radius: float):
    while True:
        G = nx.random_geometric_graph(N, radius)
        if nx.is_connected(G): break
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in G.edges():
        p, q = pos[u], pos[v]
        G.edges[u, v]['distance'] = math.hypot(p[0]-q[0], p[1]-q[1])
    return G, pos


def insert_stations(route: list[int], G: nx.Graph, stations: list[int], capacity: float, rate: float) -> list[int]:
    # 先展平完整路径
    full_path = [route[0]]
    for u, v in zip(route[:-1], route[1:]):
        seg = nx.shortest_path(G, u, v, weight='distance')
        if full_path[-1] == seg[0]:
            full_path.extend(seg[1:])
        else:
            full_path.extend(seg)
    # 规划换电：保持至少25%电量
    reserve = capacity * 0.25
    kwh = capacity
    planned = [full_path[0]]
    for i in range(len(full_path)-1):
        u = full_path[i]
        v = full_path[i+1]
        dist = G.edges[u, v]['distance']
        need = dist * rate
        # 如果扣减后低于保留量，则寻找最近的前置站点换电
        if kwh - need < reserve:
            # 找到最近的 full_path 中前置换电站节点
            candidates = [(j, full_path[j]) for j in range(i, -1, -1) if full_path[j] in stations]
            if not candidates:
                raise RuntimeError(f"无可达换电站，{u}->{v} 段可能太远")
            # j0, st = 最近
            j0, st = candidates[0]
            planned.append(st)
            kwh = capacity
        # 扣减并添加下一点
        kwh -= need
        planned.append(v)
    return planned


def simulate():
    # 构建图与站点
    N, radius = 100, 0.15
    G, pos = generate_connected_graph(N, radius)
    depots = [50, 73, 41, 49, 27]
    fixed_stations = [84, 25, 22, 66, 63, 16, 58]
    extra = random.sample([n for n in range(1, N) if n not in fixed_stations], 3)
    station_nodes = fixed_stations + extra
    stations = {n: EnergyStation(node=n) for n in station_nodes}

    # 生成车队
    trucks = []
    cand = [n for n in range(N) if n not in station_nodes]
    for i in range(TRUCK_COUNT):
        start = random.choice(depots)
        k = random.randint(SERVICE_MIN, min(SERVICE_MAX, len(cand)))
        services = random.sample([n for n in cand if n != start], k)
        route = [start] + services + [start]
        trucks.append(Truck(
            id=i+1,
            capacity_kwh=TRUCK_CAPACITY_KWH,
            current_kwh=TRUCK_CAPACITY_KWH,
            consumption_rate=TRUCK_CONSUMPTION_RATE,
            route=route
        ))

    # 插站并行驶
    record_swaps = {}
    for t in trucks:
        full_route = insert_stations(t.route, G, station_nodes, t.capacity_kwh, t.consumption_rate)
        t.route = full_route
        record_swaps[t.id] = t.drive(G, stations)

    # EV到达
    rate_step = {n: stations[n].ev_lambda * (TIME_WINDOW_MIN/60) for n in station_nodes}
    ev_arrivals = {n: [] for n in station_nodes}
    for _ in range(STEPS):
        for n, s in stations.items(): ev_arrivals[n].append(s.serve_ev(rate_step[n]))

    return G, pos, depots, stations, trucks, record_swaps, ev_arrivals


def visualize(G, pos, depots, stations, trucks, record_swaps, ev_arrivals):
    fig, ax = plt.subplots(figsize=(12,9))
    nx.draw(G, pos=pos, ax=ax, node_size=15, edge_color='lightgray', alpha=0.3)
    nx.draw_networkx_labels(G, pos, {n:str(n) for n in G.nodes()}, font_size=4, ax=ax)
    # 可视化
    for d in depots: ax.scatter(*pos[d], c='gold', marker='*', s=150, edgecolors='black', label=f'Depot {d}')
    sx = [pos[n][0] for n in stations]; sy = [pos[n][1] for n in stations]
    ax.scatter(sx, sy, c='orange', marker='s', s=80, label='Station')
    cmap = plt.get_cmap('hsv', TRUCK_COUNT)
    for i, t in enumerate(trucks):
        xs, ys = zip(*(pos[n] for n in t.route))
        ax.plot(xs, ys, color=cmap(i), linewidth=0.8, alpha=0.6, label=f'Truck {t.id}', zorder=2)
        for s in record_swaps[t.id]: ax.scatter(*pos[s], facecolors='none', edgecolors='red', s=150, marker='o', zorder=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=5, ncol=3)
    ax.set_title('HDT Routes & EV Service')
    plt.tight_layout(); plt.show()

    # 打印卡车参数、换电与EV统计
    print("Truck Specs and Routes:")
    for t in trucks:
        print(f"Truck {t.id}: Capacity {t.capacity_kwh} kWh, Consumption {t.consumption_rate} kWh/unit, Route {t.route}")
    for t in trucks:
        print(f"Truck {t.id}: Capacity {t.capacity_kwh} kWh, Consumption {t.consumption_rate} kWh/unit")
    print("\nSwap events:")
    for tid, sw in record_swaps.items(): print(f"Truck {tid} swapped at: {sw}")
    print("\nEV arrivals per station:")
    for n, arr in ev_arrivals.items(): print(f"Station {n}: {arr}")

if __name__ == '__main__':
    G, pos, depots, stations, trucks, record_swaps, ev_arrivals = simulate()
    visualize(G, pos, depots, stations, trucks, record_swaps, ev_arrivals)