# -*- coding: utf-8 -*-
"""
从 data.pkl 读取并绘制“路网”：
- 若 data['traffic_graph'] 存在（NetworkX 图）：按真实路网绘制；
- 若不存在（None）：仅绘制设施散点；可选构造 kNN 近邻图近似显示。
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ========= 配置：修改为你的 data.pkl 路径 =========
PKL_PATH = Path(r"C:\Users\user\Desktop\Wangsenyao\VRP\pythonProject5\pythonProject5\data.pkl")

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
    # 轻微上移一点，避免文字压住点（可按需调 dy）
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

# ========= 主逻辑 =========
if not PKL_PATH.exists():
    raise FileNotFoundError(f"找不到 data.pkl：{PKL_PATH}")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

G = data.get("traffic_graph", None)
locations = data.get("locations", {})
dist_matrix = data.get("dist_matrix", {})

if isinstance(G, nx.Graph):
    print("[INFO] 读取到真实路网（NetworkX Graph）")
    _plot_true_graph(G, locations)
else:
    print("[WARN] data.pkl 中没有真实路网（traffic_graph=None）")
    if PLOT_KNN_IF_NO_GRAPH and dist_matrix:
        _plot_knn_from_dist(locations, dist_matrix, k=K_FOR_KNN)
    else:
        _plot_facility_scatter_only(locations)
