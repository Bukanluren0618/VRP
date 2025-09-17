# HDT_Swapping_Optimization/src/simulation/road_network.py

import networkx as nx
import numpy as np
import pandas as pd
from src.common import config_final as config


def generate_connected_graph(N, radius):
    """
    生成一个随机但保证连通的几何图。
    """
    while True:
        G = nx.random_geometric_graph(N, radius)
        if nx.is_connected(G):
            break
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in G.edges():
        p, q = pos[u], pos[v]
        G.edges[u, v]['distance'] = np.hypot(p[0] - q[0], p[1] - q[1])
    return G, pos


def get_path_and_distance_matrices(G, key_locations_nodes):
    """
    【最终修复版】: 计算所有关键位置节点对之间的最短路径和距离。
    此版本更健壮，能处理节点子集计算。
    """
    print("正在计算所有关键节点对之间的最短路径和距离矩阵...")

    # 确保所有关键位置都在图中
    missing_nodes = [node for node in key_locations_nodes if node not in G]
    if missing_nodes:
        raise ValueError(f"以下关键节点在路网图中不存在: {missing_nodes}")

    # --- 1. 距离矩阵计算 (更健壮的方式) ---
    # 先计算图中所有节点之间的距离
    all_nodes_list = list(G.nodes())
    dist_matrix_full = pd.DataFrame(nx.floyd_warshall_numpy(G, nodelist=all_nodes_list, weight='distance'),
                                    index=all_nodes_list, columns=all_nodes_list)

    # 然后从完整矩阵中提取我们需要的关键节点的部分
    dist_matrix = dist_matrix_full.loc[key_locations_nodes, key_locations_nodes]

    # --- 2. 路径矩阵计算 ---
    path_matrix = pd.DataFrame(index=key_locations_nodes, columns=key_locations_nodes, dtype=object)

    all_paths_gen = nx.all_pairs_dijkstra_path(G, weight='distance')
    paths_dict = {source: targets for source, targets in all_paths_gen}

    for start_node in key_locations_nodes:
        for end_node in key_locations_nodes:
            if start_node in paths_dict and end_node in paths_dict[start_node]:
                path_matrix.loc[start_node, end_node] = paths_dict[start_node][end_node]
            else:
                path_matrix.loc[start_node, end_node] = []

    print("路径和距离矩阵计算完成。")
    return dist_matrix, path_matrix