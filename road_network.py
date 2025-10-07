# src/simulation/road_network.py

import networkx as nx
import random
import pandas as pd
import osmnx as ox
import math


# ---------------- Waxman 城市路网（中心密、边缘疏） ----------------
def generate_city_like_graph(n_nodes, *, alpha=0.45, beta=0.10):
    """
    生成近似城市路网的 Waxman 图。
    - alpha ↑ ：短边概率↑，中心更稠密
    - beta  ↓ ：长边概率↓，远距离边更少
    若不连通：仅最小补边以连通（保留原有 Waxman 连边特征）。
    返回: (G, pos) 其中 pos 为 {node: (x, y)}，G 的边含 'distance'（欧氏距离）。
    """
    print(f"--- 正在生成 Waxman 城市模拟路网 (alpha={alpha}, beta={beta}) ---")
    # 1) 生成 Waxman 随机几何图（节点自带 'pos'）
    G = nx.waxman_graph(n=n_nodes, alpha=alpha, beta=beta, domain=(0, 0, 1, 1))
    pos = nx.get_node_attributes(G, "pos")

    # 2) 给已有边写入欧氏距离
    for u, v in G.edges():
        pu, pv = pos[u], pos[v]
        G[u][v]["distance"] = float(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))

    # 3) 检查连通性，若不连通则最小化补边
    if not nx.is_connected(G):
        print("  -> 路网不连通，正在进行最小化补边...")
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            c1 = random.choice(list(components[i]))
            c2 = random.choice(list(components[i + 1]))
            G.add_edge(c1, c2)
            pu, pv = pos[c1], pos[c2]
            G[c1][c2]["distance"] = float(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))

    print(f"Waxman 路网生成成功，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
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
    """
    print("正在计算所有关键节点对之间的最短路径和距离矩阵...")
    dist_matrix = pd.DataFrame(index=node_ids, columns=node_ids, dtype=float)
    path_matrix = pd.DataFrame(index=node_ids, columns=node_ids, dtype=object)

    # 权重使用 'distance' (for Waxman) or 'length' (for OSMnx)
    weight_key = 'distance' if 'distance' in list(G.edges(data=True))[0][2] else 'length'

    all_pairs_path_length = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight_key))
    all_pairs_path = dict(nx.all_pairs_dijkstra_path(G))

    for start_node in node_ids:
        for end_node in node_ids:
            if start_node == end_node:
                dist_matrix.loc[start_node, end_node] = 0.0
                path_matrix.loc[start_node, end_node] = [start_node]
            else:
                dist = all_pairs_path_length.get(start_node, {}).get(end_node, float('inf'))
                # OSMnx 距离单位是米, Waxman是相对距离，都需要后续在loader中缩放
                dist_matrix.loc[start_node, end_node] = dist
                path_matrix.loc[start_node, end_node] = all_pairs_path.get(start_node, {}).get(end_node, [])

    print("路径和距离矩阵计算完成。")
    return dist_matrix, path_matrix