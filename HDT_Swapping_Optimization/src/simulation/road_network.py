# HDT_Swapping_Optimization/src/simulation/road_network.py

import networkx as nx
import numpy as np
import pandas as pd
from src.common import config_final as config


def create_road_network(locations):
    """
    创建一个带权重的、更真实的随机城市路网图。
    """
    print("正在创建城市路网图...")
    G = nx.Graph()
    all_locations = list(locations.keys())

    # 1. 添加给定的关键位置节点
    for name, pos in locations.items():
        G.add_node(name, pos=pos, type='location')

    # 2. 添加一些随机的交叉路口节点来丰富路网
    for i in range(config.ROAD_NETWORK_INTERSECTIONS):
        node_name = f"Junction_{i}"
        G.add_node(node_name, pos=(np.random.randint(-100, 100), np.random.randint(-100, 100)), type='junction')

    all_nodes = list(G.nodes())

    # 3. 连接节点形成路网
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            node1 = all_nodes[i]
            node2 = all_nodes[j]
            pos1 = G.nodes[node1]['pos']
            pos2 = G.nodes[node2]['pos']
            dist = np.linalg.norm(np.array(pos1) - np.array(pos2))

            if dist < 80:
                road_condition_factor = np.random.uniform(1.0, 1.3)
                G.add_edge(node1, node2, weight=dist * road_condition_factor)

    # 4. 确保图是连通的
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            c1 = list(components[i])
            c2 = list(components[i + 1])
            min_dist = np.inf
            node_to_connect1, node_to_connect2 = None, None
            for n1 in c1:
                for n2 in c2:
                    dist = np.linalg.norm(np.array(G.nodes[n1]['pos']) - np.array(G.nodes[n2]['pos']))
                    if dist < min_dist:
                        min_dist = dist
                        node_to_connect1, node_to_connect2 = n1, n2
            G.add_edge(node_to_connect1, node_to_connect2, weight=min_dist)

    print(f"路网创建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G


def get_path_and_distance_matrices(G, key_locations):
    """
    使用A*算法计算所有关键位置节点对之间的最短路径和距离。
    """
    print("正在使用A*算法计算所有关键节点对之间的最短路径...")
    num_locations = len(key_locations)
    dist_matrix = pd.DataFrame(np.zeros((num_locations, num_locations)), index=key_locations, columns=key_locations)
    path_matrix = pd.DataFrame(index=key_locations, columns=key_locations, dtype=object)

    def heuristic(u, v):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        return np.linalg.norm(np.array(pos_u) - np.array(pos_v))

    for start_node in key_locations:
        for end_node in key_locations:
            if start_node == end_node:
                dist_matrix.loc[start_node, end_node] = 0
                path_matrix.loc[start_node, end_node] = [start_node]
            else:
                try:
                    path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')
                    distance = nx.astar_path_length(G, start_node, end_node, heuristic=heuristic, weight='weight')
                    dist_matrix.loc[start_node, end_node] = distance
                    path_matrix.loc[start_node, end_node] = path
                except nx.NetworkXNoPath:
                    dist_matrix.loc[start_node, end_node] = np.inf
                    path_matrix.loc[start_node, end_node] = None

    print("路径和距离矩阵计算完成。")
    return dist_matrix, path_matrix