# HDT_Swapping_Optimization/src/simulation/road_network.py

import networkx as nx
import numpy as np
import pandas as pd
from src.common import config_final as config


def create_road_network_from_file(locations_df, road_network_df):
    """
    从数据文件创建一个带权重的路网图。
    """
    print("正在从文件创建城市路网图...")
    G = nx.Graph()

    for _, row in locations_df.iterrows():
        node_id = row['id']
        try:
            pos = (float(row.get('pos_x', 0)), float(row.get('pos_y', 0)))
        except (ValueError, TypeError):
            pos = (0, 0)
        G.add_node(node_id, pos=pos, type=row.get('type', 'location'))

    for _, row in road_network_df.iterrows():
        u, v, dist = row['start_node'], row['end_node'], row['distance']
        road_condition_factor = np.random.uniform(1.0, 1.05)
        G.add_edge(u, v, weight=dist * road_condition_factor)

    if not nx.is_connected(G):
        print("=" * 50)
        print("警告: 从文件加载的路网不是完全连通的。")
        print("=" * 50)

    print(f"路网创建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G


def get_path_and_distance_matrices(G, key_locations):
    """
    计算所有关键位置节点对之间的最短路径和距离。
    """
    print("正在计算所有关键节点对之间的最短路径和距离矩阵...")

    missing_nodes = [node for node in key_locations if node not in G]
    if missing_nodes:
        raise ValueError(f"以下关键节点在路网图中不存在: {missing_nodes}")

    dist_matrix = pd.DataFrame(nx.floyd_warshall_numpy(G, nodelist=key_locations, weight='weight'),
                               index=key_locations, columns=key_locations)

    path_matrix = pd.DataFrame(index=key_locations, columns=key_locations, dtype=object)
    all_paths_gen = nx.all_pairs_dijkstra_path(G, weight='weight')
    paths_dict = {source: targets for source, targets in all_paths_gen}

    for start_node in key_locations:
        for end_node in key_locations:
            if start_node in paths_dict and end_node in paths_dict[start_node]:
                path_matrix.loc[start_node, end_node] = paths_dict[start_node][end_node]
            else:
                path_matrix.loc[start_node, end_node] = []

    print("路径和距离矩阵计算完成。")
    return dist_matrix, path_matrix