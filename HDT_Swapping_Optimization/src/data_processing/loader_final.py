# src/data_processing/loader_final.py

import os
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
from scipy.stats import norm

from src.simulation import road_network


# ---------------- Waxman 城市路网（中心密、边缘疏） ----------------
def generate_city_like_graph(n_nodes, *, alpha=0.45, beta=0.10):
    """
    生成近似城市路网的 Waxman 图。
    - alpha ↑ ：短边概率↑，中心更稠密
    - beta  ↓ ：长边概率↓，远距离边更少
    若不连通：仅最小补边以连通（保留原有 Waxman 连边特征）。
    返回: (G, pos) 其中 pos 为 {node: (x, y)}，G 的边含 'distance'（欧氏距离）。
    """
    # 1) 生成 Waxman 随机几何图（节点自带 'pos'）
    G = nx.waxman_graph(n=n_nodes, alpha=alpha, beta=beta, domain=(0, 0, 1, 1))
    pos = nx.get_node_attributes(G, "pos")

    # 2) 给已有边写入欧氏距离
    for u, v in G.edges():
        pu, pv = pos[u], pos[v]
        G[u][v]["distance"] = float(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))

    # 3) 若不连通：按“最近跨分量点对”逐步补边，直到连通（不替换、不清空原边）
    if not nx.is_connected(G):
        comps = [set(c) for c in nx.connected_components(G)]
        # 贪心：每次把两个最近的分量连起来
        while len(comps) > 1:
            best = None  # (dist, u, v, ia, ib)
            for ia in range(len(comps)):
                for ib in range(ia + 1, len(comps)):
                    Ca, Cb = comps[ia], comps[ib]
                    # 在 Ca 与 Cb 间找最近点对
                    for u in Ca:
                        pu = pos[u]
                        for v in Cb:
                            pv = pos[v]
                            d = (pu[0] - pv[0]) ** 2 + (pu[1] - pv[1]) ** 2
                            if best is None or d < best[0]:
                                best = (d, u, v, ia, ib)
            _, u, v, ia, ib = best
            duv = float(math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]))
            G.add_edge(u, v, distance=duv)
            # 合并分量
            merged = comps[ia] | comps[ib]
            comps = [comps[k] for k in range(len(comps)) if k not in (ia, ib)]
            comps.append(merged)

    # 4) 把 pos 写回节点属性，保证可视化使用
    for n in G.nodes():
        G.nodes[n]["pos"] = tuple(pos[n])
    return G, pos


# ---------------- 备用：MST + kNN 稀疏图（避免全连接） ----------------
def build_sparse_graph_from_pos(pos, *, k_neighbors=4, max_geom=None):
    """
    基于节点坐标构建稀疏路网：MST 保证连通 + 每节点 k 近邻补边（可选几何距离上限）。
    返回: 带 'distance' 属性的图 G。
    """
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))

    # 1) 完全图上做 MST（距为欧氏距离）
    complete = nx.Graph()
    complete.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            complete.add_edge(nodes[i], nodes[j], distance=float(D[i, j]))
    mst = nx.minimum_spanning_tree(complete, weight="distance")

    # 2) 在 MST 基础上为每个节点补 k 近邻边
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(mst.edges(data=True))

    for i, u in enumerate(nodes):
        order = np.argsort(D[i])
        added = 0
        for j in order[1:]:  # 跳过自身
            v = nodes[j]
            d = float(D[i, j])
            if max_geom is not None and d > max_geom:
                continue
            if not G.has_edge(u, v):
                G.add_edge(u, v, distance=d)
                added += 1
            if added >= k_neighbors:
                break

    for n in nodes:
        G.nodes[n]["pos"] = tuple(pos[n])
    return G


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.model_data = {}
        self.road_network = None

    # ---------------- PV / EV 随机曲线 ----------------
    def get_stochastic_pv_generation(self, time_steps, time_step_hours, peak_power, noise_level):
        hours = np.array([t * time_step_hours for t in time_steps])
        sin_wave = np.sin(np.maximum(0, hours - 6) / 12 * np.pi)
        base_pv = np.maximum(0, sin_wave) ** 1.5 * peak_power
        cloud_noise = pd.Series(np.random.normal(0, noise_level, len(time_steps))).rolling(window=4, min_periods=1).mean()
        cloud_factor = np.clip(1 - cloud_noise, 1 - noise_level, 1.0)
        stochastic_pv = np.maximum(0, base_pv * cloud_factor)
        return pd.Series(stochastic_pv, index=time_steps)

    def get_stochastic_ev_demand(self, time_steps, time_step_hours, peak_kw, noise_level):
        if peak_kw == 0:
            return pd.Series(0, index=time_steps)
        hours = np.array([t * time_step_hours for t in time_steps])
        peak1 = norm.pdf(hours, loc=8.5, scale=1.5)
        peak2 = norm.pdf(hours, loc=18, scale=2.5)
        base = (peak1 * 0.6 + peak2)
        base_demand = base / np.max(base) * peak_kw
        noise = np.random.normal(0, peak_kw * noise_level, len(time_steps))
        stochastic = np.maximum(0, base_demand + noise)
        return pd.Series(stochastic, index=time_steps)

    # ---------------- 主加载流程 ----------------
    def load_all(self):
        print("=" * 30)
        print("开始创建全新的【城市配送+配网交互】场景...")
        cfg = self.config
        random.seed(42); np.random.seed(42)

        # 选择路网生成方式：优先使用 Waxman（城市风格）
        road_model = getattr(cfg, "ROAD_MODEL", "waxman").lower()
        if road_model == "waxman":
            alpha = getattr(cfg, "WAXMAN_ALPHA", 0.55)
            beta  = getattr(cfg, "WAXMAN_BETA", 0.08)
            G, pos = generate_city_like_graph(cfg.CITY_NODE_COUNT, alpha=alpha, beta=beta)
        else:
            # 兼容旧流程：用你已有的随机连通点集 + 稀疏化
            base_graph, pos = road_network.generate_connected_graph(cfg.CITY_NODE_COUNT, cfg.CITY_GRAPH_RADIUS)
            # 如果你希望限制几何连边上限（单位：pos 的原始尺度），可把 km 换算回 pos 尺度
            max_geom = None
            if hasattr(cfg, "ROADS_MAX_KM") and cfg.ROADS_MAX_KM is not None:
                # pos 是 unit-square；dist_matrix 之后才乘 CITY_SCALE_KM，所以这里把 km 换回几何距离
                max_geom = cfg.ROADS_MAX_KM / max(1e-9, cfg.CITY_SCALE_KM)
            G = build_sparse_graph_from_pos(pos, k_neighbors=getattr(cfg, "ROADS_KNN", 4), max_geom=max_geom)

        # 确保每条边都有 'distance'，每个点有 'pos'、'info'
        for u, v in G.edges():
            if "distance" not in G[u][v]:
                pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
                G[u][v]["distance"] = float(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))
        for n in G.nodes():
            if "pos" not in G.nodes[n]:
                G.nodes[n]["pos"] = tuple(pos[n])
            if "info" not in G.nodes[n]:
                G.nodes[n]["info"] = {"name": str(n)}

        self.road_network = G

        # IEEE118 负荷节点用于电网映射
        try:
            net = pp.networks.case118()
            load_buses = net.load.bus.to_list()
            print(f"成功加载IEEE 118节点系统，找到 {len(load_buses)} 个可用的负荷节点。")
        except Exception as e:
            print(f"加载pandapower case118失败: {e}。将使用备用节点列表。")
            load_buses = list(range(1, 119))
            net = None

        # 随机分配设施点
        all_nodes = list(G.nodes())
        random.shuffle(all_nodes)
        depot_nodes = all_nodes[:cfg.NUM_DEPOTS]
        station_nodes = all_nodes[cfg.NUM_DEPOTS: cfg.NUM_DEPOTS + cfg.NUM_STATIONS]
        customer_nodes = all_nodes[cfg.NUM_DEPOTS + cfg.NUM_STATIONS:
                                   cfg.NUM_DEPOTS + cfg.NUM_STATIONS + cfg.NUM_CUSTOMERS]

        locations_data = {}
        for i, node in enumerate(depot_nodes):
            name = f"Depot_{i + 1}"
            locations_data[name] = {"type": "Depot", "pos": G.nodes[node]["pos"], "node_id": node}
            G.nodes[node]["info"] = {"name": name, "type": "Depot"}

        for i, node in enumerate(customer_nodes):
            name = f"Customer_{i + 1}"
            locations_data[name] = {"type": "Customer", "pos": G.nodes[node]["pos"], "node_id": node}
            G.nodes[node]["info"] = {"name": name, "type": "Customer"}

        stations_info = {}
        station_to_bus_map = {}
        random.shuffle(load_buses)
        for i, node in enumerate(station_nodes):
            station_name = f"Station_{i + 1}"
            bus_id = load_buses[i % len(load_buses)]
            locations_data[station_name] = {"type": "SwapStation", "pos": G.nodes[node]["pos"], "node_id": node}
            G.nodes[node]["info"] = {"name": station_name, "type": "SwapStation"}
            stations_info[station_name] = {"initial_full": 10, "initial_empty": 5, "bus_id": bus_id}
            station_to_bus_map[station_name] = bus_id

        print(f"场景设施点分配完毕: {len(depot_nodes)}仓库, {len(stations_info)}换电站, {len(customer_nodes)}客户。")

        # 距离/路径矩阵（基于“物理边”的最短路）
        all_location_node_ids = [d["node_id"] for d in locations_data.values()]
        dist_matrix_raw, path_matrix_raw = road_network.get_path_and_distance_matrices(G, all_location_node_ids)

        # 把索引/列改为地点名称
        node_id_to_name_map = {d["node_id"]: name for name, d in locations_data.items()}
        dist_matrix_named = dist_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)
        path_matrix = path_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)

        # 尺度放大到 km（pos 是 unit-square），time_df = 距离 / 速度
        dist_matrix = dist_matrix_named * cfg.CITY_SCALE_KM
        print(f"路网已缩放以模拟真实的城市环境，城市尺度约为 {cfg.CITY_SCALE_KM} 公里。")
        avg_speed_kmh = 30.0
        time_df = dist_matrix / avg_speed_kmh

        # 车辆
        vehicles = {}
        depot_names = [name for name, info in locations_data.items() if info["type"] == "Depot"]
        for i in range(cfg.NUM_TRUCKS):
            truck_id = f"HDT_{i + 1}"
            vehicles[truck_id] = {"initial_soc": cfg.HDT_BATTERY_CAPACITY_KWH, "depot_id": random.choice(depot_names)}

        # 任务
        tasks = {}
        task_id_counter = 1
        customer_names = [name for name, info in locations_data.items() if info["type"] == "Customer"]
        for cust_name in customer_names:
            task_id = f"Task_{task_id_counter}"
            closest_depot = min(depot_names, key=lambda d: dist_matrix.loc[d, cust_name])
            one_way_time = time_df.loc[closest_depot, cust_name]
            earliest_due = one_way_time + cfg.LOADING_UNLOADING_TIME_HOURS + 1.0
            latest_due = earliest_due + 8.0
            due_time = round(random.uniform(earliest_due, latest_due), 1)
            tasks[task_id] = {
                "delivery_to": cust_name,
                "demand": round(random.uniform(1.0, 2.5), 2),
                "due_time": due_time,
                "depot": closest_depot,
            }
            task_id_counter += 1

        print(f"已生成 {len(vehicles)} 辆卡车和 {len(tasks)} 个任务的中央任务池。")

        # 时间维度 & 价格/出力/需求
        time_steps = range(cfg.TOTAL_TIME_STEPS)
        electricity_prices = pd.Series(
            [
                0.4 if 7 <= t * cfg.TIME_STEP_HOURS < 11 or 14 <= t * cfg.TIME_STEP_HOURS < 19
                else (1.2 if 11 <= t * cfg.TIME_STEP_HOURS < 14 or 19 <= t * cfg.TIME_STEP_HOURS < 21 else 0.8)
                for t in time_steps
            ],
            index=time_steps,
        )
        pv_generation = {
            s: self.get_stochastic_pv_generation(time_steps, cfg.TIME_STEP_HOURS, cfg.PV_PEAK_POWER_KW, cfg.PV_CLOUD_NOISE_LEVEL)
            for s in stations_info
        }
        ev_demand_timestep = {
            s: self.get_stochastic_ev_demand(time_steps, cfg.TIME_STEP_HOURS, cfg.EV_DEMAND_PEAK_KW, cfg.EV_DEMAND_NOISE_LEVEL)
            for s in stations_info
        }

        # 汇总
        self.model_data = {
            "traffic_graph": G,
            "physical_edges": list(G.edges()),
            "locations": locations_data,
            "tasks": tasks,
            "vehicles": vehicles,
            "stations": stations_info,
            "dist_matrix": dist_matrix,
            "time_matrix": time_df,
            "path_matrix": path_matrix,
            "power_grid_net": net,
            "station_to_bus_map": station_to_bus_map,
            "time_steps": list(time_steps),
            "electricity_prices": electricity_prices,
            "pv_generation": pv_generation,
            "ev_demand_timestep": ev_demand_timestep,
        }

        print("=" * 30)
        print("全新的【城市配送+配网交互】场景数据创建完成！")
        return self.model_data
