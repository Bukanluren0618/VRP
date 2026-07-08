import heapq
from typing import List, Tuple, Dict, Optional
import pandas as pd

class RoadNetwork:
    def __init__(self):
        """初始化一个有向图，存储邻接表"""
        self.graph: Dict[int, List[Tuple[int, float, float]]] = {}
        # 键：起始节点，值：[(终点, 长度(km), 时间(h)), ...]

    def add_edge(self, from_node: int, to_node: int, length_km: float, time_h: float):
        """添加一条有向边"""
        if from_node not in self.graph:
            self.graph[from_node] = []
        self.graph[from_node].append((to_node, length_km, time_h))

    def build_from_data(self, edges: List[Tuple[int, int, int, float, float]]):
        """
        从边数据批量构建
        edges: 列表，每个元素为 (arc_id, from_node, to_node, length_km, t0_h)
        假设 arc_id 不使用，仅用于索引
        """
        for arc_id, from_node, to_node, length_km, t0_h in edges:
            self.add_edge(from_node, to_node, length_km, t0_h)

    def shortest_path(self, start: int, end: int, mode: str = 'distance') -> Tuple[Optional[List[int]], float]:
        """
        寻找从 start 到 end 的最短路径
        :param start: 起点节点
        :param end:   终点节点
        :param mode:  'distance' 或 'time'
        :return: (路径节点列表, 总权重) 若不存在路径则返回 (None, inf)
        """
        if start not in self.graph and start != end:
            return None, float('inf')

        # 选择权重：0 表示 length_km，1 表示 time_h
        weight_idx = 0 if mode == 'distance' else 1

        # Dijkstra 算法
        visited = set()
        dist = {start: 0.0}
        prev = {start: None}
        heap = [(0.0, start)]

        while heap:
            cur_dist, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == end:
                break

            # 处理所有邻居
            for v, length, time in self.graph.get(u, []):
                w = length if weight_idx == 0 else time
                new_dist = cur_dist + w
                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        # 根据 prev 重建路径
        if end not in dist:
            return None, float('inf')

        path = []
        node = end
        while node is not None:
            path.append(int(node))
            node = prev.get(node)
        path.reverse()

        return path, dist[end]


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 原始数据（arc_id, from_node, to_node, length_km, t0_h）
    data = pd.read_pickle('./raw_data_bpr.pkl')
    arcs_bpr_df = data['arc_df']
    arc_df = arcs_bpr_df[['arc_id', 'from_node', 'to_node', 'length_km', 't0_h']]
    edges_data = arc_df.values.tolist()
    # 构建路网
    network = RoadNetwork()
    network.build_from_data(edges_data)

    # 测试：从节点 0 到节点 52，按距离最短
    path, total_dist = network.shortest_path(0, 52, mode='distance')
    print(f"距离最短路径: {path}, 总距离: {total_dist:.4f} km")

    # 测试：从节点 0 到节点 52，按时间最短
    path, total_time = network.shortest_path(0, 52, mode='time')
    print(f"时间最短路径: {path}, 总时间: {total_time:.4f} h")

    # 测试：从节点 0 到节点 61（无直接连接，但可通过中间节点？这里没有连接0和1的边）
    path, total = network.shortest_path(0, 0, mode='distance')
    print(f"从0到61: 路径={path}, 总距离={total}")
    path, total = network.shortest_path(0, 0, mode='time')
    print(f"从0到61: 路径={path},  总时间: {total_time:.4f} h")