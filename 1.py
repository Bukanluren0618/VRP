# -*- coding: utf-8 -*-
"""
生成 data.pkl ：中心更密集 + 外围稀疏 + 适度“死路”的城市化路网（含设施/车辆/任务/时间序列/PV/电压）
—— 在你提供的版本基础上整理为“可直接调参”的完整版 ——

你只需改“PARAMS 区”的开关与数值即可：
- 生成密度/中心化：CENTER_BIAS, KNN_* , LONG_EDGE_Q
- 外圈死路强度：LEAF_OUTER_Q, LEAF_FRAC, LEAF_KEEP_K
- 低度兜底：MIN_DEGREE, DENSIFY_RADIUS, ADD_PER_NODE
- 核心补洞：CORE_Q, HOLE_RADIUS, BRIDGES_PER_CC
- 手动微调：PULL_NODES 列表、MANUAL_EDGES 列表
"""

import os, sys, math, pickle, random
import numpy as np
import pandas as pd
import networkx as nx

# ------------------------- 读取配置（唯一入口） -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [BASE_DIR, os.path.join(BASE_DIR, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from importlib import reload
import src.common.config_final as cfg
cfg = reload(cfg)  # 若你刚改过 config_final，确保最新

def _need(name):
    if not hasattr(cfg, name):
        raise ValueError(f"[config_final 缺少必需项] {name}")
    return getattr(cfg, name)

def _opt(name, default):
    return getattr(cfg, name, default)
# 基础参数（全部带硬编码默认值）
TOTAL_STEPS               = int(_need("TOTAL_TIME_STEPS"))
TIME_STEP_HOURS           = float(_need("TIME_STEP_HOURS"))
NUM_DEPOTS                = int(_need("NUM_DEPOTS"))
NUM_STATIONS              = int(_need("NUM_STATIONS"))
NUM_CUSTOMERS             = int(_need("NUM_CUSTOMERS"))
NUM_TRUCKS                = int(_need("NUM_TRUCKS"))
CITY_NODE_COUNT           = int(_need("CITY_NODE_COUNT"))
CITY_SCALE_KM             = float(_need("CITY_SCALE_KM"))
AVG_SPEED_KMH             = float(_need("AVG_SPEED_KMH"))
HDT_BATTERY_CAPACITY_KWH  = float(_need("HDT_BATTERY_CAPACITY_KWH"))
LOADING_UNLOADING_TIME_HOURS = float(_need("LOADING_UNLOADING_TIME_HOURS"))
VOLTAGE_MIN               = float(_need("VOLTAGE_MIN"))
VOLTAGE_MAX               = float(_need("VOLTAGE_MAX"))
PV_PEAK_POWER_KW          = float(_need("PV_PEAK_POWER_KW"))

SEED = 42

print(f"[CFG] steps={TOTAL_STEPS}, dt={TIME_STEP_HOURS}, depots={NUM_DEPOTS}, stations={NUM_STATIONS}, customers={NUM_CUSTOMERS}, trucks={NUM_TRUCKS}")
print(f"[CFG] CITY_NODE_COUNT={CITY_NODE_COUNT}, CITY_SCALE_KM={CITY_SCALE_KM}, AVG_SPEED_KMH={AVG_SPEED_KMH}. HDT_BATTERY_CAPACITY_KWH={HDT_BATTERY_CAPACITY_KWH}")


# 生成器（整体连接度 & 中心化）
CENTER_BIAS  = 7          # ↑越大越靠中心
N_RINGS      = 5
KNN_INNER    = 7          # 核心圈近邻数
KNN_MID      = 6
KNN_OUTER    = 5          # 外圈近邻数
LONG_EDGE_Q  = 0.70       # 删除最远 30% 长边（越大→删得越少）

# 外圈“死路”造型（越小越少死路/越温和）
LEAF_OUTER_Q = 0.72       # 仅半径分位 > 此阈值的外圈参与
LEAF_FRAC    = 0.18       # 参与外圈节点比例
LEAF_KEEP_K  = 4          # 每个保留的最近边条数（1=典型死路，4 基本无死路）

# 低度兜底（保证至少多少条连接）
MIN_DEGREE     = 5
DENSIFY_RADIUS = 0.18
ADD_PER_NODE   = 1

# 核心补洞（仅核心圈内短桥）
CORE_Q          = 0.80
HOLE_RADIUS     = 0.15
BRIDGES_PER_CC  = 3

# 手动微调：把此列表中的节点向中心拉近（0<k<1；越小越近）
PULL_NODES      = [30, 39, 57, 58]
PULL_K          = 0.55

# 手动连边（按 node id 指定）
MANUAL_EDGES = [
    (12, 45), (58, 31), (58, 61), (58, 9), (70, 31), (70, 9), (30, 31),
    (14, 78), (61, 95), (37, 81), (76, 83), (93, 39), (39, 80),
    (8, 64), (8, 43), (93, 8), (48, 16), (16, 8), (25, 8),
    (78, 28), (68, 10), (8, 80), (93, 40), (16, 93), (8, 62),
    (37, 55), (15, 52), (15, 9), (70, 52), (15, 31), (33, 54), (19, 20), (52, 92), (80, 16)
]
# ======================================================================


# ---- 导入配置----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [BASE_DIR, os.path.join(BASE_DIR, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.common import config_final as cfg
    TOTAL_STEPS              = int(getattr(cfg, "TOTAL_TIME_STEPS", TOTAL_STEPS))
    TIME_STEP_HOURS          = float(getattr(cfg, "TIME_STEP_HOURS", TIME_STEP_HOURS))
    NUM_DEPOTS               = int(getattr(cfg, "NUM_DEPOTS", NUM_DEPOTS))
    NUM_STATIONS             = int(getattr(cfg, "NUM_STATIONS", NUM_STATIONS))
    NUM_CUSTOMERS            = int(getattr(cfg, "NUM_CUSTOMERS", NUM_CUSTOMERS))
    NUM_TRUCKS               = int(getattr(cfg, "NUM_TRUCKS", NUM_TRUCKS))
    CITY_SCALE_KM            = float(getattr(cfg, "CITY_SCALE_KM", CITY_SCALE_KM))
    HDT_BATTERY_CAPACITY_KWH = float(getattr(cfg, "HDT_BATTERY_CAPACITY_KWH", HDT_BATTERY_CAPACITY_KWH))
    LOADING_UNLOADING_TIME_HOURS = float(getattr(cfg, "LOADING_UNLOADING_TIME_HOURS", LOADING_UNLOADING_TIME_HOURS))
    PV_PEAK_POWER_KW         = float(getattr(cfg, "PV_PEAK_POWER_KW", PV_PEAK_POWER_KW))
except Exception:
    pass

np.random.seed(SEED)

# ---- 导入路网模块 ----
try:
    import src.simulation.road_network as road_network
except Exception:
    import importlib.util
    path = os.path.join(BASE_DIR, "road_network.py")
    spec = importlib.util.spec_from_file_location("road_network", path)
    road_network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(road_network)


# ================= 工具函数 =================
def generate_concentric_city_graph_fallback(n_nodes, *, center_bias=4.0, n_rings=3,
                                            knn_inner=5, knn_mid=4, knn_outer=3,
                                            long_edge_quantile=0.85, seed=42):
    """当 src.simulation.road_network 中没有 concentric 生成器时使用"""
    rng = np.random.default_rng(seed)
    random.seed(seed); np.random.seed(seed)
    sigma = 0.25 / float(center_bias)
    xs = np.clip(rng.normal(0.5, sigma, size=n_nodes), 0.0, 1.0)
    ys = np.clip(rng.normal(0.5, sigma, size=n_nodes), 0.0, 1.0)
    pos = {i: (float(xs[i]), float(ys[i])) for i in range(n_nodes)}

    def dist(a, b):
        ax, ay = pos[a]; bx, by = pos[b]
        return math.hypot(ax - bx, ay - by)

    rc = np.hypot(xs - 0.5, ys - 0.5)
    q = np.quantile(rc, np.linspace(0, 1, n_rings + 1))
    rings = [[] for _ in range(n_rings)]
    for i in range(n_nodes):
        idx = min(n_rings - 1, max(0, int(np.searchsorted(q, rc[i], side='right') - 1)))
        rings[idx].append(i)

    def knn_for_ring(idx): return knn_inner if idx == 0 else knn_mid if idx == 1 else knn_outer

    G = nx.Graph(); [G.add_node(i) for i in range(n_nodes)]
    for ridx, ring_nodes in enumerate(rings):
        if not ring_nodes: continue
        k = knn_for_ring(ridx)
        for i in ring_nodes:
            cands = set(ring_nodes)
            if ridx - 1 >= 0: cands |= set(rings[ridx - 1])
            if ridx + 1 < n_rings: cands |= set(rings[ridx + 1])
            cands.discard(i)
            neighs = sorted(((j, dist(i, j)) for j in cands), key=lambda t: t[1])[:k]
            for j, dval in neighs:
                if not G.has_edge(i, j): G.add_edge(i, j, distance=dval)
    for ring_nodes in rings:
        if len(ring_nodes) < 3: continue
        order = sorted(ring_nodes, key=lambda i: math.atan2(pos[i][1]-0.5, pos[i][0]-0.5))
        for a, b in zip(order, order[1:]+order[:1]):
            if not G.has_edge(a, b): G.add_edge(a, b, distance=dist(a, b))
    if G.number_of_edges() > 0:
        lens = np.array([edata.get('distance', 1.0) for _,_,edata in G.edges(data=True)])
        cut = float(np.quantile(lens, long_edge_quantile))
        G.remove_edges_from([(u,v) for u,v,edata in G.edges(data=True) if edata.get('distance',1.0) > cut])
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        base = comps[0]
        for comp in comps[1:]:
            best, best_d = None, float('inf')
            for u in base:
                for v in comp:
                    dval = dist(u, v)
                    if dval < best_d: best_d, best = dval, (u, v)
            if best:
                u, v = best
                G.add_edge(u, v, distance=best_d)
                base += comp
    return G, pos


def promote_outer_leaves(G, pos, *, outer_q=0.65, frac=0.3, keep_k=1):
    """让外圈部分节点成为死路（只保留 keep_k 条最近的边），保持连通"""
    G = G.copy()
    r = np.array([math.hypot(pos[i][0]-0.5,pos[i][1]-0.5) for i in G.nodes()])
    rthr = float(np.quantile(r, outer_q))
    outer = [n for n in G.nodes() if r[n] > rthr and G.degree(n) > keep_k]
    if not outer:
        return G
    pick = set(random.sample(outer, max(1, int(len(outer)*frac))))
    for n in pick:
        nbrs = list(G.neighbors(n))
        dlist = [(m, math.hypot(pos[n][0]-pos[m][0], pos[n][1]-pos[m][1])) for m in nbrs]
        dlist.sort(key=lambda t: t[1])
        keep = set(m for m,_ in dlist[:keep_k])
        for m,_ in dlist[keep_k:]:
            if G.has_edge(n,m): G.remove_edge(n,m)
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        base = comps[0]
        for comp in comps[1:]:
            best, best_d = None, float('inf')
            for u in base:
                for v in comp:
                    d = math.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1])
                    if d < best_d: best_d, best = d, (u, v)
            if best: u, v = best; G.add_edge(u, v, distance=best_d); base += comp
    return G


def densify_low_degree(G, pos, *, min_degree=3, radius=0.15, add_per_node=2):
    """让度 < min_degree 的节点在 radius 内补最短边"""
    G = G.copy()
    for n, d in list(G.degree()):
        if d >= min_degree: continue
        need = max(0, min_degree - d)
        cand = []
        for m in G.nodes():
            if m == n or G.has_edge(n, m): continue
            dd = math.hypot(pos[n][0]-pos[m][0], pos[n][1]-pos[m][1])
            if dd <= radius: cand.append((m, dd))
        cand.sort(key=lambda t: t[1])
        for m, dd in cand[:min(need+add_per_node, len(cand))]:
            G.add_edge(n, m, distance=dd)
            if G.degree(n) >= min_degree: break
    return G


def fill_core_holes(G, pos, *, core_q=0.45, hole_radius=0.10, bridges_per_cc=2):
    """核心圈补洞（近邻图分块时，用最短桥接边连通）"""
    G2 = G.copy()
    r = np.array([math.hypot(pos[i][0]-0.5, pos[i][1]-0.5) for i in G2.nodes()])
    rthr = float(np.quantile(r, core_q))
    core_nodes = [n for n in G2.nodes() if r[n] <= rthr]
    core = G2.subgraph(core_nodes).copy()

    H = nx.Graph(); H.add_nodes_from(core.nodes())
    for u, v in core.edges():
        d = G2[u][v].get('distance', math.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]))
        if d <= hole_radius: H.add_edge(u, v, distance=d)

    comps = [list(c) for c in nx.connected_components(H)]
    if len(comps) <= 1: return G2

    bridges = []
    for i in range(len(comps)):
        for j in range(i+1, len(comps)):
            best, best_d = None, float('inf')
            for u in comps[i]:
                for v in comps[j]:
                    d = math.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1])
                    if d < best_d: best_d, best = d, (u, v)
            if best: bridges.append((best_d, best))
    bridges.sort(key=lambda t: t[0])
    added = 0
    for _, (u, v) in bridges:
        if not G2.has_edge(u, v):
            G2.add_edge(u, v, distance=math.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]))
            added += 1
            if added >= bridges_per_cc: break
    return G2


def pull_nodes_toward_center(pos: dict, nodes, k=0.65, center=(0.5, 0.5), clamp=(1e-3, 0.999)): #0.65
    """把 nodes 径向向中心拉近到原距离的 k 倍（仅改坐标）"""
    cx, cy = center
    xmin, xmax = clamp
    for n in nodes:
        if n not in pos: continue
        x, y = pos[n]; vx, vy = x - cx, y - cy
        x2, y2 = cx + k*vx, cy + k*vy
        pos[n] = (min(max(x2, xmin), xmax), min(max(y2, xmin), xmax))
    return pos


# ========================== 生成路网 ==========================
if hasattr(road_network, "generate_concentric_city_graph"):
    G, pos = road_network.generate_concentric_city_graph(
        n_nodes = getattr(cfg, "CITY_NODE_COUNT", 100),
        center_bias = CENTER_BIAS,
        n_rings = N_RINGS,
        knn_inner = KNN_INNER, knn_mid = KNN_MID, knn_outer = KNN_OUTER,
        long_edge_quantile = LONG_EDGE_Q,
        seed = SEED
    )
else:
    G, pos = generate_concentric_city_graph_fallback(
        n_nodes = getattr(cfg, "CITY_NODE_COUNT", 100),
        center_bias = CENTER_BIAS,
        n_rings = N_RINGS,
        knn_inner = KNN_INNER, knn_mid = KNN_MID, knn_outer = KNN_OUTER,
        long_edge_quantile = LONG_EDGE_Q,
        seed = SEED
    )

# 外圈适度“死路”
G = promote_outer_leaves(G, pos, outer_q=LEAF_OUTER_Q, frac=LEAF_FRAC, keep_k=LEAF_KEEP_K)

# 低度兜底
G = densify_low_degree(G, pos, min_degree=MIN_DEGREE, radius=DENSIFY_RADIUS, add_per_node=ADD_PER_NODE)

# 核心补洞
G = fill_core_holes(G, pos, core_q=CORE_Q, hole_radius=HOLE_RADIUS, bridges_per_cc=BRIDGES_PER_CC)

# 手动微调坐标与连边
if PULL_NODES:
    pos = pull_nodes_toward_center(pos, nodes=PULL_NODES, k=PULL_K)
for u, v in MANUAL_EDGES:
    if u in G and v in G:
        d = math.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1])
        G.add_edge(u, v, distance=d)

# 把坐标写回节点属性
for n,(x,y) in pos.items():
    G.nodes[n]["x"], G.nodes[n]["y"] = float(x), float(y)

# ========================== 设施分配（支持“指定位置”） ==========================
FIXED_DEPOT_NODE_IDS   = [83, 53]                 # 固定 2 个仓库
FIXED_STATION_NODE_IDS = [70, 54, 8, 65, 50, 64, 31, 85, 32, 44]            # 固定这几个换电站
# 校验节点是否存在
for nid in FIXED_DEPOT_NODE_IDS + FIXED_STATION_NODE_IDS:
    if nid not in G.nodes():
        raise ValueError(f"❌ 节点 {nid} 不存在，请检查编号是否正确。")

# 不允许重复
if set(FIXED_DEPOT_NODE_IDS) & set(FIXED_STATION_NODE_IDS):
    raise ValueError("❌ 仓库与换电站节点重复，请重新指定。")

# 客户节点为剩余所有未被占用的节点
reserved = set(FIXED_DEPOT_NODE_IDS + FIXED_STATION_NODE_IDS)
customer_nodes = [n for n in G.nodes() if n not in reserved][:NUM_CUSTOMERS]

# 数量检查
if len(FIXED_DEPOT_NODE_IDS) != NUM_DEPOTS:
    print(f"[WARN] 实际仓库数量 {len(FIXED_DEPOT_NODE_IDS)} 与 NUM_DEPOTS={NUM_DEPOTS} 不一致。")
if len(FIXED_STATION_NODE_IDS) != NUM_STATIONS:
    print(f"[WARN] 实际换电站数量 {len(FIXED_STATION_NODE_IDS)} 与 NUM_STATIONS={NUM_STATIONS} 不一致。")

# 组装 locations
locations = {}
for i, n in enumerate(FIXED_DEPOT_NODE_IDS[:NUM_DEPOTS], 1):
    locations[f"Depot_{i}"] = {"type": "Depot", "node_id": n}
for i, n in enumerate(FIXED_STATION_NODE_IDS[:NUM_STATIONS], 1):
    locations[f"Station_{i}"] = {"type": "SwapStation", "node_id": n}
for i, n in enumerate(customer_nodes, 1):
    locations[f"Customer_{i}"] = {"type": "Customer", "node_id": n}

print(f"✅ 已固定 {len(FIXED_DEPOT_NODE_IDS)} 个仓库，{len(FIXED_STATION_NODE_IDS)} 个换电站。")
# ======================== 设施分配结束 ========================




# ========================== 矩阵计算 ==========================
node_id_to_name = {v["node_id"]: k for k, v in locations.items()}
facility_nodes  = [v["node_id"] for v in locations.values()]

# 用现有函数重算最短路 + 距离
dist_raw, path_raw = road_network.get_path_and_distance_matrices(G, facility_nodes)
dist_matrix = dist_raw.rename(index=node_id_to_name, columns=node_id_to_name) * CITY_SCALE_KM
path_matrix = path_raw.rename(index=node_id_to_name, columns=node_id_to_name)
time_matrix = dist_matrix / AVG_SPEED_KMH

# ========================== 车辆 / 任务 / 时间序列 ==========================
depots = [k for k, v in locations.items() if v["type"]=="Depot"]
vehicles = {f"HDT_{i+1}": {"initial_soc":HDT_BATTERY_CAPACITY_KWH, "depot_id": depots[i % len(depots)]}
            for i in range(NUM_TRUCKS)}

customers = [k for k, v in locations.items() if v["type"]=="Customer"]
def nearest_depot(c): return min(depots, key=lambda d: time_matrix.loc[d, c])

tasks = {}
for i, c in enumerate(customers, 1):
    d = nearest_depot(c)
    one_way = float(time_matrix.loc[d, c])
    due = round(one_way + LOADING_UNLOADING_TIME_HOURS + float(np.random.uniform(1.0,4.0)), 3)
    tasks[f"Task_{i}"] = {"delivery_to": c, "demand": round(float(np.random.uniform(1.0,3.0)), 3),
                          "due_time": due, "depot": d}

stations = [k for k, v in locations.items() if v["type"]=="SwapStation"]
stations_info = {s: {"initial_full":10, "initial_empty":5, "bus_id": i+1} for i, s in enumerate(stations)}
station_to_bus_map = {s: info["bus_id"] for s, info in stations_info.items()}

# ========================== IEEE118 电网映射 ==========================
import pandapower as pp

net = pp.networks.case118()
print(f"🔌 已载入 IEEE118 系统，共 {len(net.bus)} 个母线。")

# 为每个站创建：storage（BESS）+ sgen（PV）+ load（EV/站内负荷）
# 同时记录各自的表索引，便于后续快速更新功率
load_index_map    = {}   # {station: load_idx}
sgen_index_map    = {}   # {station: sgen_idx}
storage_index_map = {}   # {station: storage_idx}
bus_map           = {}   # {station: bus_id}

for i, s in enumerate(stations):
    bus_id = int(i % len(net.bus))          # 你也可以改成自己想绑定的 bus
    bus_map[s] = bus_id

    stor_idx = pp.create_storage(
        net, bus=bus_id, p_mw=0.0, max_e_mwh=0.5,
        soc_percent=50.0, min_e_mwh=0.0, max_p_mw=0.3,
        controllable=True, name=f"{s}_BESS"
    )
    sgen_idx = pp.create_sgen(
        net, bus=bus_id, p_mw=0.0, q_mvar=0.0,
        name=f"{s}_PV", type="PV"
    )
    load_idx = pp.create_load(
        net, bus=bus_id, p_mw=0.0, q_mvar=0.0,
        name=f"{s}_EVload"
    )
    storage_index_map[s] = int(stor_idx)
    sgen_index_map[s]    = int(sgen_idx)
    load_index_map[s]    = int(load_idx)

print(f"✅ 已将 {len(stations)} 个换电站挂载到 IEEE118 网络。")

# 用 IEEE118 的 bus 映射覆盖/更新 station_to_bus_map（以后以电网为准）
station_to_bus_map = bus_map


time_steps = list(range(TOTAL_STEPS))
electricity_prices = pd.Series(0.9 + 0.3*np.sin(np.linspace(0, 10*math.pi, TOTAL_STEPS)), index=time_steps)

pv_generation = {}
for s in stations:
    base = np.maximum(0.0, np.sin((np.array(time_steps)*TIME_STEP_HOURS - 6.0)/12.0*np.pi))**1.5
    pv   = base/(base.max()+1e-9)*PV_PEAK_POWER_KW
    noise = np.clip(np.random.normal(0, 0.03, TOTAL_STEPS), -0.1, 0.1)
    pv_generation[s] = [float(max(0.0, p*(1.0+e))) for p, e in zip(pv, noise)]

v_mid = (VOLTAGE_MIN + VOLTAGE_MAX)/2.0
voltage_pre = {s: [float(v_mid)]*TOTAL_STEPS for s in stations}
ev_demand_timestep = {s: [0.0]*TOTAL_STEPS for s in stations}



# ========================== 写出 data.pkl（新增网元索引） ==========================
data = {
    "traffic_graph": G, "locations": locations, "tasks": tasks, "vehicles": vehicles,
    "stations": stations_info, "dist_matrix": dist_matrix, "time_matrix": time_matrix,
    "path_matrix": path_matrix, "power_grid_net": net,
    "station_to_bus_map": station_to_bus_map,
    "pp_load_index_map": load_index_map,
    "pp_sgen_index_map": sgen_index_map,
    "pp_storage_index_map": storage_index_map,
    "time_steps": time_steps, "electricity_prices": electricity_prices,
    "pv_generation": pv_generation, "ev_demand_timestep": ev_demand_timestep,
    "voltage_pre": voltage_pre
}
OUT_PATH = os.path.join(BASE_DIR, "data.pkl")
with open(OUT_PATH, "wb") as f:
    pickle.dump(data, f)
print(f"✅ 已生成 data.pkl（步长={TOTAL_STEPS}）")
print(f"   Depots={NUM_DEPOTS}, Stations={NUM_STATIONS}, Customers={NUM_CUSTOMERS}, Trucks={NUM_TRUCKS}")
print("📁", OUT_PATH)
