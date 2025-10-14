# src/simulation/road_network.py

import networkx as nx
import random
import pandas as pd
import osmnx as ox
import math
import numpy as np

# ---------------- Waxman 城市路网（默认） ----------------
def generate_city_like_graph(n_nodes, *, alpha=0.45, beta=0.10, seed=42):
    """
    生成近似城市路网的 Waxman 图（均匀分布）。
    - alpha ↑ ：短边概率↑，整体更稠密
    - beta  ↓ ：长边概率↓，远距离边更少
    若不连通：最小化补边以连通（保留原有 Waxman 连边特征）。
    返回: (G, pos) 其中 pos 为 {node: (x, y)}，G 的边含 'distance'（欧氏距离）。
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"--- 正在生成 Waxman 城市模拟路网 (alpha={alpha}, beta={beta}) ---")
    # 1) 生成 Waxman 随机几何图（节点自带 'pos'）
    G = nx.waxman_graph(n=n_nodes, alpha=alpha, beta=beta, domain=(0, 0, 1, 1), seed=seed)
    pos = nx.get_node_attributes(G, "pos")

    # 2) 给已有边写入欧氏距离
    for u, v in G.edges():
        pu, pv = pos[u], pos[v]
        G[u][v]["distance"] = float(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))

    # 3) 检查连通性，若不连通则最小化补边
    if not nx.is_connected(G):
        print("  -> 路网不连通，正在进行最小化补边...")
        components = [list(c) for c in nx.connected_components(G)]
        # 逐个连接到最大的分量
        base = components[0]
        for comp in components[1:]:
            c1 = random.choice(base)
            c2 = random.choice(comp)
            G.add_edge(c1, c2)
            pu, pv = pos[c1], pos[c2]
            G[c1][c2]["distance"] = float(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))
            base += comp

    print(f"Waxman 路网生成成功，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G, pos


def generate_concentric_city_graph(
    n_nodes: int,
    *,
    center_bias: float = 4.0,     # 节点向中心聚集强度（越大越集中）
    n_rings: int = 3,             # 同心环数量
    knn_inner: int = 5,           # 内圈近邻数
    knn_mid: int = 4,             # 中圈近邻数
    knn_outer: int = 3,           # 外圈近邻数
    long_edge_quantile: float = 0.85,  # 删长边的分位数（越小删得越狠）
    seed: int = 42
):
    """
    生成“中心更密、外围更疏、长边更少、具有环向结构”的城市型路网。
    步骤：
      1) 节点：以 (0.5,0.5) 为中心的 2D 高斯；center_bias 控制集中度。
      2) KNN 局部连接：内>中>外（让中心更稠密，郊区更稀疏）。
      3) 同心环：按半径分 n_rings 个圈，每圈按角度排序连成“环路”。
      4) 删长边：按全图边长分位数裁剪，减少“跨城长边”。
      5) 若不连通：按分量最近点做最小桥接，保证整图连通。
    返回：
      - G: networkx.Graph（边属性 'distance'）
      - pos: {node: (x,y)}，归一化到 [0,1]^2
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1) 节点坐标（中心偏置）
    sigma = 0.25 / float(center_bias)
    xs = np.clip(rng.normal(0.5, sigma, size=n_nodes), 0.0, 1.0)
    ys = np.clip(rng.normal(0.5, sigma, size=n_nodes), 0.0, 1.0)
    pos = {i: (float(xs[i]), float(ys[i])) for i in range(n_nodes)}

    def dist(a, b):
        ax, ay = pos[a]; bx, by = pos[b]
        return math.hypot(ax - bx, ay - by)

    # 2) 半径 & 环分层
    rc = np.array([math.hypot(pos[i][0] - 0.5, pos[i][1] - 0.5) for i in range(n_nodes)])
    q = np.quantile(rc, np.linspace(0, 1, n_rings + 1))
    rings = [[] for _ in range(n_rings)]
    for i in range(n_nodes):
        r = rc[i]
        idx = min(n_rings - 1, max(0, int(np.searchsorted(q, r, side='right') - 1)))
        rings[idx].append(i)

    # 3) KNN 局部连接（内圈更多近邻，且只在本环+相邻环中选邻居）
    def knn_for_ring(idx):
        if idx == 0: return knn_inner
        if idx == 1: return knn_mid
        return knn_outer

    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)

    for ridx, ring_nodes in enumerate(rings):
        if not ring_nodes:
            continue
        k = knn_for_ring(ridx)
        for i in ring_nodes:
            candidates = set(ring_nodes)
            if ridx - 1 >= 0: candidates |= set(rings[ridx - 1])
            if ridx + 1 < n_rings: candidates |= set(rings[ridx + 1])
            candidates.discard(i)
            if not candidates:
                continue
            neighs = sorted(((j, dist(i, j)) for j in candidates), key=lambda t: t[1])[:k]
            for j, dval in neighs:
                if not G.has_edge(i, j):
                    G.add_edge(i, j, distance=dval)

    # 4) 同心环“环路”（按角度连成圈）
    for ring_nodes in rings:
        if len(ring_nodes) < 3:
            continue
        ring_nodes_sorted = sorted(
            ring_nodes, key=lambda i: math.atan2(pos[i][1] - 0.5, pos[i][0] - 0.5)
        )
        for a, b in zip(ring_nodes_sorted, ring_nodes_sorted[1:] + ring_nodes_sorted[:1]):
            dval = dist(a, b)
            if not G.has_edge(a, b):
                G.add_edge(a, b, distance=dval)

    # 5) 删长边
    if G.number_of_edges() > 0:
        lengths = np.array([edata.get('distance', 1.0) for _, _, edata in G.edges(data=True)])
        cutoff = float(np.quantile(lengths, long_edge_quantile))
        to_remove = [(u, v) for u, v, edata in G.edges(data=True) if edata.get('distance', 1.0) > cutoff]
        G.remove_edges_from(to_remove)

    # 6) 若不连通，最小桥接
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        base = comps[0]
        for comp in comps[1:]:
            best_pair, best_d = None, float('inf')
            for u in base:
                for v in comp:
                    dval = dist(u, v)
                    if dval < best_d:
                        best_d, best_pair = dval, (u, v)
            if best_pair:
                u, v = best_pair
                G.add_edge(u, v, distance=best_d)
                base += comp

    return G, pos



def generate_real_road_network(city_name="Piedmont, California, USA"):
    """
    使用OSMnx从OpenStreetMap下载指定城市的真实世界街道网络。
    """
    print(f"--- 正在从OpenStreetMap下载 '{city_name}' 的真实路网... ---")
    G = ox.graph_from_place(city_name, network_type='drive')
    G = nx.Graph(G)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    print(f"真实路网 '{city_name}' 加载成功，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G


def get_path_and_distance_matrices(G, node_ids):
    """
    计算指定节点列表之间所有点对的最短路径和距离。
    返回：
      dist_matrix: DataFrame(index=node_ids, columns=node_ids, dtype=float)
      path_matrix: DataFrame(index=node_ids, columns=node_ids, dtype=object), 值为节点序列(list)
    """
    print("正在计算所有关键节点对之间的最短路径和距离矩阵...")
    dist_matrix = pd.DataFrame(index=node_ids, columns=node_ids, dtype=float)
    path_matrix = pd.DataFrame(index=node_ids, columns=node_ids, dtype=object)

    # 兼容空图或无边图
    edges_list = list(G.edges(data=True))
    if len(edges_list) == 0:
        # 无边：仅对自身为 0，其他为 inf，路径为空
        for s in node_ids:
            for t in node_ids:
                if s == t:
                    dist_matrix.loc[s, t] = 0.0
                    path_matrix.loc[s, t] = [s]
                else:
                    dist_matrix.loc[s, t] = float('inf')
                    path_matrix.loc[s, t] = []
        print("图无边，已返回空路径矩阵。")
        return dist_matrix, path_matrix

    # 权重使用 'distance' (for Waxman/本地图) or 'length' (for OSMnx)
    first_edge_attr = edges_list[0][2]
    if 'distance' in first_edge_attr:
        weight_key = 'distance'
    elif 'length' in first_edge_attr:
        weight_key = 'length'
    else:
        weight_key = None  # 无权重

    # 预先计算所有对最短路
    if weight_key:
        all_pairs_path_length = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight_key))
        all_pairs_path = dict(nx.all_pairs_dijkstra_path(G, weight=weight_key))
    else:
        all_pairs_path_length = dict(nx.all_pairs_shortest_path_length(G))
        all_pairs_path = dict(nx.all_pairs_shortest_path(G))

    for start_node in node_ids:
        for end_node in node_ids:
            if start_node == end_node:
                dist_matrix.loc[start_node, end_node] = 0.0
                path_matrix.loc[start_node, end_node] = [start_node]
            else:
                dist = all_pairs_path_length.get(start_node, {}).get(end_node, float('inf'))
                dist_matrix.loc[start_node, end_node] = float(dist)
                path = all_pairs_path.get(start_node, {}).get(end_node, [])
                path_matrix.loc[start_node, end_node] = list(path) if path else []

    print("路径和距离矩阵计算完成。")
    return dist_matrix, path_matrix
