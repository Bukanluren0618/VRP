# -*- coding: utf-8 -*-
"""
从 data.pkl 读取并绘制“路网”并进行规模/时间一致性检查：
- 若 data['traffic_graph'] 存在（NetworkX 图）：按真实路网绘制；
- 若不存在（None）：仅绘制设施散点；可选构造 kNN 近邻图近似显示。
- 额外输出：
  * 估计的 km-per-unit 比例尺（应≈生成时 CITY_SCALE_KM）
  * 坐标包络尺度与其 km 粗估
  * 设施层最短路距离/时间统计
  * 由 dist/time 反推的有效速度（应≈ AVG_SPEED_KMH）
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"  # 或改为 "Qt5Agg"
import matplotlib
matplotlib.use("TkAgg")  # 或 "Qt5Agg"
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from itertools import islice
import math

# ========= 配置：修改为你的 data.pkl 路径 =========
PKL_PATH = Path(r"C:\PY3\pythonProject5\pythonProject5\pythonProject5\data.pkl")

# 若没有真实路网时，是否用 dist_matrix 临时构造 kNN 近邻“伪路网”
PLOT_KNN_IF_NO_GRAPH = False
K_FOR_KNN = 5

# ========= 工具函数 =========
def _get_positions_from_graph(G: nx.Graph):
    """优先用节点属性 (x,y) 或 pos；否则 spring_layout"""
    if all(("x" in G.nodes[n] and "y" in G.nodes[n]) for n in G.nodes):
        return {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    if all(("pos" in G.nodes[n]) for n in G.nodes):
        return {n: G.nodes[n]["pos"] for n in G.nodes}
    return nx.spring_layout(G, seed=42)

def _draw_facilities_on_axes(ax, pos_dict, locations):
    # 分类设施（注意：这里 pos_dict 的 key 是“路网节点 id”，locations 的 node_id 也是“路网节点 id”
    depot_nodes    = [v["node_id"] for _, v in locations.items() if v.get("type") == "Depot"]
    station_nodes  = [v["node_id"] for _, v in locations.items() if v.get("type") == "SwapStation"]
    customer_nodes = [v["node_id"] for _, v in locations.items() if v.get("type") == "Customer"]

    if depot_nodes:
        ax.scatter([pos_dict[n][0] for n in depot_nodes],
                   [pos_dict[n][1] for n in depot_nodes],
                   s=140, c="#1f77b4", marker="s", label="Depot", zorder=3)
    if station_nodes:
        ax.scatter([pos_dict[n][0] for n in station_nodes],
                   [pos_dict[n][1] for n in station_nodes],
                   s=140, c="#d62728", marker="^", label="Swap Station", zorder=3)
    if customer_nodes:
        ax.scatter([pos_dict[n][0] for n in customer_nodes],
                   [pos_dict[n][1] for n in customer_nodes],
                   s=26,  c="#7f7f7f", marker="o", label="Customer", zorder=2, alpha=0.9)

def _plot_true_graph(G: nx.Graph, locations: dict):
    pos = _get_positions_from_graph(G)

    # === 统计 ===
    edge_count = G.number_of_edges()
    node_count = G.number_of_nodes()
    print(f"[STATS] nodes={node_count}, edges={edge_count}")

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- 底图：边 + 点 ---
    nx.draw_networkx_edges(G, pos, edge_color="#d0d0d0", width=0.8, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="#bfbfbf", alpha=0.7, ax=ax)

    # --- 标注节点编号 ---
    dy = 0.002
    label_pos = {n: (p[0], p[1] + dy) for n, p in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos,
        labels={n: str(n) for n in G.nodes()},
        font_size=7, font_color="#444",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.6),
        ax=ax
    )

    # 叠加设施（仓库/站/客户）
    _draw_facilities_on_axes(ax, pos, locations)

    ax.set_title(f"Road Network from data.pkl (true graph)\nEdges={edge_count}, Nodes={node_count}")
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

def _plot_facility_scatter_only(locations: dict):
    # 没有真实路网时，locations 里没有坐标；做一个可重复的随机散点仅用于“看到分布”
    names = list(locations.keys())
    rng = np.random.default_rng(42)
    xy = {n: (float(rng.uniform(0, 50)), float(rng.uniform(0, 50))) for n in names}

    fig, ax = plt.subplots(figsize=(12, 10))
    # 仅设施：按类型绘制
    depot = [n for n, v in locations.items() if v["type"] == "Depot"]
    stat  = [n for n, v in locations.items() if v["type"] == "SwapStation"]
    cust  = [n for n, v in locations.items() if v["type"] == "Customer"]

    if cust:
        ax.scatter([xy[n][0] for n in cust], [xy[n][1] for n in cust],
                   s=26, c="#7f7f7f", marker="o", label="Customer", alpha=0.9)
    if stat:
        ax.scatter([xy[n][0] for n in stat], [xy[n][1] for n in stat],
                   s=140, c="#d62728", marker="^", label="Swap Station")
    if depot:
        ax.scatter([xy[n][0] for n in depot], [xy[n][1] for n in depot],
                   s=140, c="#1f77b4", marker="s", label="Depot")

    ax.set_title("No true road network in data.pkl (traffic_graph=None)\nShowing facility scatter only")
    ax.set_axis_off()
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

def _plot_knn_from_dist(locations: dict, dist_matrix: dict, k: int = 5):
    # 用距离矩阵在设施层面构 kNN 图（仅近似，可视化连通关系）
    names = list(locations.keys())
    # 随机稳定坐标（可替换为 MDS/TSNE 以更“像距离”）
    rng = np.random.default_rng(42)
    xy = {n: (float(rng.uniform(0, 50)), float(rng.uniform(0, 50))) for n in names}

    G = nx.Graph()
    for n in names: G.add_node(n)
    for a in names:
        dists = sorted(((b, dist_matrix[a][b]) for b in names if b != a), key=lambda t: t[1])[:k]
        for b, d in dists:
            G.add_edge(a, b, length=float(d))

    pos = {n: xy[n] for n in names}

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, edge_color="#d0d0d0", width=0.8, alpha=0.6, ax=ax)

    depot = [n for n, v in locations.items() if v["type"] == "Depot"]
    stat  = [n for n, v in locations.items() if v["type"] == "SwapStation"]
    cust  = [n for n, v in locations.items() if v["type"] == "Customer"]

    if cust:
        ax.scatter([pos[n][0] for n in cust], [pos[n][1] for n in cust],
                   s=20, c="#999999", marker="o", label="Customer")
    if stat:
        ax.scatter([pos[n][0] for n in stat], [pos[n][1] for n in stat],
                   s=100, c="#d62728", marker="^", label="Station")
    if depot:
        ax.scatter([pos[n][0] for n in depot], [pos[n][1] for n in depot],
                   s=100, c="#1f77b4", marker="s", label="Depot")

    ax.set_title(f"No true road network; kNN({k}) approximation from dist_matrix")
    ax.set_axis_off()
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

# ========= 规模/时间一致性检查 =========
def _scale_and_time_checks(data: dict):
    """
    打印：
      - 估计 km-per-unit（应≈生成时的 CITY_SCALE_KM）
      - 坐标包络宽/高/对角及其 km 粗估
      - 设施层最短路距离/时间统计
      - 由 dist/time 反推速度分布的中位数（应≈ AVG_SPEED_KMH）
    """
    G: nx.Graph = data.get("traffic_graph", None)
    locations: dict = data.get("locations", {})
    dist_matrix = data.get("dist_matrix", None)   # 已是 km
    time_matrix = data.get("time_matrix", None)   # 小时；若未保存则可能为 None

    if not isinstance(G, nx.Graph) or not locations or not isinstance(dist_matrix, (pd.DataFrame, dict)):
        print("[CHECK] 跳过：缺少 G/locations/dist_matrix")
        return

    if isinstance(dist_matrix, dict):
        dist_df = pd.DataFrame(dist_matrix)
    else:
        dist_df = dist_matrix

    # 位置与包络
    pos = _get_positions_from_graph(G)
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    diag_unit = math.hypot(w, h)

    # —— 估计 km-per-unit：用设施节点对的原图最短路 'distance'（单位）与 dist_df(km) 的比值的中位数
    #      取若干随机设施对，鲁棒些
    loc_items = list(locations.items())
    node_ids = [v["node_id"] for _, v in loc_items]
    loc_names = [k for k, _ in loc_items]

    rng = np.random.default_rng(123)
    pairs = set()
    if len(loc_names) >= 2:
        while len(pairs) < min(200, len(loc_names) * 3):
            a, b = rng.choice(loc_names, 2, replace=False)
            pairs.add(tuple(sorted((a, b))))
    pairs = list(pairs)

    km_per_unit_samples = []
    for a, b in pairs:
        na = locations[a]["node_id"]
        nb = locations[b]["node_id"]
        try:
            raw_len = nx.shortest_path_length(G, na, nb, weight="distance")  # 单位长度
        except Exception:
            raw_len = None
        km_len = dist_df.loc[a, b] if (a in dist_df.index and b in dist_df.columns) else None
        if raw_len is not None and km_len is not None and np.isfinite(raw_len) and raw_len > 0 and np.isfinite(km_len) and km_len > 0:
            km_per_unit_samples.append(float(km_len) / float(raw_len))

    if km_per_unit_samples:
        km_per_unit = float(np.median(km_per_unit_samples))
    else:
        km_per_unit = float("nan")

    # —— 设施层最短路统计（km）
    vals_km = dist_df.values
    mask = np.isfinite(vals_km) & (vals_km > 0)
    km_stats = {}
    if mask.any():
        valid = vals_km[mask]
        km_stats = {
            "mean_km": float(valid.mean()),
            "p50_km":  float(np.percentile(valid, 50)),
            "p90_km":  float(np.percentile(valid, 90)),
            "max_km":  float(valid.max()),
        }

    # —— 时间与速度（若有 time_matrix）
    speed_stats = {}
    if isinstance(time_matrix, (pd.DataFrame, dict)):
        time_df = pd.DataFrame(time_matrix) if isinstance(time_matrix, dict) else time_matrix
        if time_df.shape == dist_df.shape and all(time_df.index == dist_df.index) and all(time_df.columns == dist_df.columns):
            time_vals = time_df.values
            m = mask & np.isfinite(time_vals) & (time_vals > 0)
            if m.any():
                speeds = vals_km[m] / time_vals[m]  # km/h
                speed_stats = {
                    "median_speed": float(np.median(speeds)),
                    "mean_speed":   float(speeds.mean()),
                    "p10_speed":    float(np.percentile(speeds, 10)),
                    "p90_speed":    float(np.percentile(speeds, 90)),
                }

    # —— 打印
    print("\n========== SCALE/TIME CHECK ==========")
    print(f"[Coords] bbox_w={w:.4f} unit, bbox_h={h:.4f} unit, diag={diag_unit:.4f} unit")
    if not np.isnan(km_per_unit):
        print(f"[Scale ] Est. km-per-unit ≈ {km_per_unit:.3f} km/unit  （应≈ 你的 CITY_SCALE_KM）")
        print(f"[Coords] bbox_w≈{w*km_per_unit:.2f} km, bbox_h≈{h*km_per_unit:.2f} km, diag≈{diag_unit*km_per_unit:.2f} km")
    else:
        print("[Scale ] 无法估计 km-per-unit（请检查 dist_matrix/time_matrix 是否齐全）")

    if km_stats:
        print(f"[Dist  ] 设施对最短路：mean={km_stats['mean_km']:.2f} km, P50={km_stats['p50_km']:.2f} km, "
              f"P90={km_stats['p90_km']:.2f} km, max={km_stats['max_km']:.2f} km")
    else:
        print("[Dist  ] 无有效设施间最短路样本")

    if speed_stats:
        print(f"[Speed ] 由 dist/time 反推：median={speed_stats['median_speed']:.2f} km/h, "
              f"mean={speed_stats['mean_speed']:.2f}, P10={speed_stats['p10_speed']:.2f}, P90={speed_stats['p90_speed']:.2f}")
        print("         （若你未做额外时间倍率，应≈ AVG_SPEED_KMH，例如 40 km/h）")
    else:
        print("[Speed ] 未找到匹配的 time_matrix，或其形状与 dist_matrix 不一致")
    print("======================================\n")

# ========= 主逻辑 =========
if not PKL_PATH.exists():
    raise FileNotFoundError(f"找不到 data.pkl：{PKL_PATH}")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

G = data.get("traffic_graph", None)
locations = data.get("locations", {})
dist_matrix = data.get("dist_matrix", {})

# —— 先做规模/时间一致性检查
_scale_and_time_checks(data)

# —— 再绘图
if isinstance(G, nx.Graph):
    print("[INFO] 读取到真实路网（NetworkX Graph）")
    _plot_true_graph(G, locations)
else:
    print("[WARN] data.pkl 中没有真实路网（traffic_graph=None）")
    if PLOT_KNN_IF_NO_GRAPH and isinstance(dist_matrix, (pd.DataFrame, dict)):
        _plot_knn_from_dist(locations, dist_matrix, k=K_FOR_KNN)
    else:
        _plot_facility_scatter_only(locations)
