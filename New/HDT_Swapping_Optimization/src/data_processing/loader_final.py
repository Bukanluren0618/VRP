# HDT_Swapping_Optimization/src/data_processing/loader_final.py

import pandas as pd
import numpy as np
import os
import json
import networkx as nx
from src.common import config_final as config
from src.simulation import road_network


def create_scenario_from_manual_data():
    """
    【最终修复版】: 修正了文件读取编码问题，并直接从硬编码的Python字典创建场景。
    所有数据均来自您在最后一次提示中提供的文本。
    """
    print("=" * 30)
    print("开始从最终手动定义的场景数据创建模型输入...")

    # --- 1. 定义节点、弧段和路网 (根据您的数据) ---
    print("正在创建交通路网...")

    # a. 定义所有节点及其坐标 (使用随机坐标进行可视化)
    locations_data = {
        '广西壮族自治区百色市田东县': {'type': 'Customer',
                                       'pos': (np.random.randint(-150, -50), np.random.randint(50, 150))},
        '广西壮族自治区百色市右江区龙景街道': {'type': 'Customer',
                                               'pos': (np.random.randint(-150, -50), np.random.randint(-50, 50))},
        '河北省保定市定州市庞村镇S76': {'type': 'Depot', 'pos': (0, 0)},  # 假设河北是仓库/枢纽
        '河北省保定市高阳县庞口镇': {'type': 'Customer',
                                     'pos': (np.random.randint(50, 150), np.random.randint(50, 150))},
        '海南省海口市秀英区长流镇': {'type': 'Customer',
                                     'pos': (np.random.randint(-50, 50), np.random.randint(-150, -50))},
        '海南省儋州市三都街道': {'type': 'Customer', 'pos': (np.random.randint(50, 150), np.random.randint(-150, -50))},
        '换电站ID1': {'type': 'SwapStation', 'pos': (np.random.randint(-50, 50), np.random.randint(50, 150))},
        '换电站ID2': {'type': 'SwapStation', 'pos': (np.random.randint(50, 150), np.random.randint(-50, 50))},
        '换电站ID3': {'type': 'SwapStation', 'pos': (np.random.randint(-50, 50), np.random.randint(-150, -50))},
        '换电站ID4': {'type': 'SwapStation', 'pos': (np.random.randint(-150, -50), np.random.randint(-50, 50))}
    }

    # b. 构建交通网络图
    traffic_graph = nx.Graph()
    for loc, attrs in locations_data.items():
        traffic_graph.add_node(loc, **attrs)

    # c. 添加您定义的弧段
    arc_data = {
        '1': {'nodes': ('广西壮族自治区百色市田东县', '广西壮族自治区百色市右江区龙景街道'), 'length': 66,
              'time_min': 158.83},
        '2': {'nodes': ('河北省保定市定州市庞村镇S76', '河北省保定市高阳县庞口镇'), 'length': 103.68,
              'time_min': 192.81},
        '3': {'nodes': ('海南省海口市秀英区长流镇', '海南省儋州市三都街道'), 'length': 128.71, 'time_min': 232.09}
    }
    for arc in arc_data.values():
        u, v = arc['nodes']
        traffic_graph.add_edge(u, v, weight=arc['length'])

    # d. 增加连接以保证图的连通性 (非常重要)
    depot_node = '河北省保定市定州市庞村镇S76'
    traffic_graph.add_edge('广西壮族自治区百色市右江区龙景街道', depot_node, weight=1500)
    traffic_graph.add_edge('海南省儋州市三都街道', depot_node, weight=2000)
    for station_id in ['换电站ID1', '换电站ID2', '换电站ID3', '换电站ID4']:
        traffic_graph.add_edge(station_id, depot_node, weight=np.random.uniform(50, 150))

    key_locations_list = list(locations_data.keys())
    dist_matrix, path_matrix = road_network.get_path_and_distance_matrices(traffic_graph, key_locations_list)
    time_df = dist_matrix / 50.0  # 假设平均时速50km/h
    dist_matrix = dist_matrix / 50.0

    # --- 2. 解析任务、车辆、站点和电池信息 ---
    print("正在解析任务、车辆和站点数据...")

    # a. 站点数据
    stations = {
        '换电站ID1': {'initial_full': 7, 'initial_empty': 5, 'bus_id': 'Bus_1'},
        '换电站ID2': {'initial_full': 4, 'initial_empty': 6, 'bus_id': 'Bus_2'},
        '换电站ID3': {'initial_full': 7, 'initial_empty': 3, 'bus_id': 'Bus_3'},
        '换电站ID4': {'initial_full': 4, 'initial_empty': 6, 'bus_id': 'Bus_4'}
    }

    # b. 电池数据
    battery_specs = {
        '电池ID1': {'rated_capacity': 400.0, 'max_c_rate_charge': 1.5, 'max_c_rate_discharge': 1.5},
        '电池ID2': {'rated_capacity': 400.0, 'max_c_rate_charge': 2.0, 'max_c_rate_discharge': 2.0},
        '电池ID3': {'rated_capacity': 282.0, 'max_c_rate_charge': 1.0, 'max_c_rate_discharge': 1.0}
    }

    # c. 【假设】定义示例任务 (从河北仓库出发)
    tasks = {
        'Task_Guangxi': {'delivery_to': '广西壮族自治区百色市田东县', 'demand': 5.0, 'due_time': 20.0,
                         'depot': depot_node},
        'Task_Hebei': {'delivery_to': '河北省保定市高阳县庞口镇', 'demand': 8.0, 'due_time': 5.0, 'depot': depot_node},
        'Task_Hainan': {'delivery_to': '海南省海口市秀英区长流镇', 'demand': 6.0, 'due_time': 22.0, 'depot': depot_node}
    }

    # d. 【假设】定义示例车辆
    vehicles = {
        'HDT_1': {'initial_soc': battery_specs['电池ID1']['rated_capacity'], 'depot_id': depot_node,
                  'battery_model': '电池ID1'},
        'HDT_2': {'initial_soc': battery_specs['电池ID3']['rated_capacity'], 'depot_id': depot_node,
                  'battery_model': '电池ID3'},
    }

    # --- 3. 创建配电网络和时间序列数据 ---
    # a. 配电网络
    print("正在创建配电网络拓扑...")
    power_grid = nx.DiGraph()
    power_buses = ['Bus_Sub'] + [s_info['bus_id'] for s_info in stations.values()]
    power_grid.add_nodes_from(power_buses)
    power_lines = [('Bus_Sub', s_info['bus_id']) for s_info in stations.values()]
    power_grid.add_edges_from(power_lines)
    line_params = {line: {'r_pu': 0.01, 'x_pu': 0.05} for line in power_lines}
    station_to_bus_map = {s_id: s_info['bus_id'] for s_id, s_info in stations.items()}

    # b. 时间序列数据 (从上传的文件读取)
    print("正在从文件加载时序EV需求...")
    data_dir = config.DATA_DIR
    time_steps = range(config.TOTAL_TIME_STEPS)
    ev_demand_timestep = {s: pd.Series(0.0, index=time_steps) for s in stations}
    try:
        # 【关键修复】: 添加 encoding='gbk' 来正确读取包含中文的CSV文件
        charge_load_df = pd.read_csv(os.path.join(data_dir, config.CHARGE_LOAD_FILENAME), encoding='gbk')

        charge_load_df['日期'] = pd.to_datetime(charge_load_df['日期'])
        sample_day_load = charge_load_df[charge_load_df['日期'] == charge_load_df['日期'].max()].copy()

        # 尝试多种时间格式以增加鲁棒性
        try:
            sample_day_load['hour'] = pd.to_datetime(
                sample_day_load['开始时间'], format='%H:%M:%S'
            ).dt.hour
        except ValueError:
            sample_day_load['hour'] = pd.to_datetime(
                sample_day_load['开始时间'], format='mixed'
            ).dt.hour


        for s_name_raw, group in sample_day_load.groupby('电站名称'):
            s_id = None
            for an_id in stations.keys():
                if an_id in s_name_raw:
                    s_id = an_id
                    break
            if s_id:
                hourly_load = group.groupby('hour')['电量'].sum()
                for t in time_steps:
                    hour_of_step = int(t * config.TIME_STEP_HOURS) % 24
                    if hour_of_step in hourly_load.index:
                        ev_demand_timestep[s_id][t] = hourly_load[hour_of_step]
        print("  -> 已成功从文件生成时序EV需求。")
    except Exception as e:
        print(f"  -> 警告：处理充电负荷文件失败 ({e})，将使用随机EV需求。")
        ev_demand_timestep = {s: pd.Series(np.random.uniform(50, 200, len(time_steps)), index=time_steps) for s in
                              stations}

    prices = [0.4 if 23 <= t * config.TIME_STEP_HOURS or t * config.TIME_STEP_HOURS < 7 else 1.0 for t in time_steps]
    electricity_prices = pd.Series(prices, index=time_steps)
    pv_generation = {s: pd.Series(np.sin(np.linspace(0, np.pi, len(time_steps))) ** 2 * 300, index=time_steps) for s in
                     stations}

    # --- 4. 组装最终数据字典 ---
    model_data = {
        'traffic_graph': traffic_graph, 'locations': locations_data, 'tasks': tasks,
        'vehicles': vehicles, 'stations': stations, 'battery_specs': battery_specs,
        'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': path_matrix,
        'power_grid': power_grid, 'power_buses': list(set(power_buses)), 'power_lines': power_lines,
        'line_params': line_params, 'station_to_bus_map': station_to_bus_map,
        'substation_bus': 'Bus_Sub',
        'time_steps': list(time_steps), 'electricity_prices': electricity_prices,
        'pv_generation': pv_generation, 'ev_demand_timestep': ev_demand_timestep
    }

    print("=" * 30)
    print("场景数据根据最终手动输入创建完成！")
    return model_data