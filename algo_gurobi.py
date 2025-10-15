import pandas as pd
import numpy as np
import os
import random
import networkx as nx
import pandapower as pp
from scipy.stats import norm
import osmnx as ox
import math

import src.common.config_final as config
import road_network
from collections import defaultdict
import pickle

#导入pygpopt求解器
import gurobipy as gp
from gurobipy import GRB


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.model_data = {}
        self.road_network = None

    def get_voltage_pre(self,time_steps, peak): #默认按最大计算
        voltage_max = np.array([peak*0.8] * len(time_steps))
        return pd.Series(voltage_max, index=time_steps).tolist()

    def get_voltage_next(self,time_steps, peak): #默认按最大计算
        voltage_max = np.array([peak * 0.8] * len(time_steps))
        return pd.Series(voltage_max, index=time_steps).tolist()

    def get_stochastic_pv_generation(self, time_steps, time_step_hours, peak_power, noise_level):
        hours = np.array([t * time_step_hours for t in time_steps])
        sin_wave = np.sin(np.maximum(0, hours - 6) / 12 * np.pi)
        base_pv = np.maximum(0, sin_wave) ** 1.5 * peak_power
        cloud_noise = pd.Series(np.random.normal(0, noise_level, len(time_steps))).rolling(window=4,
                                                                                           min_periods=1).mean()
        cloud_factor = np.clip(1 - cloud_noise, 1 - noise_level, 1.0)
        stochastic_pv = np.maximum(0, base_pv * cloud_factor)
        return pd.Series(stochastic_pv, index=time_steps).tolist()

    def get_stochastic_ev_demand(self, time_steps, time_step_hours, peak_kw, noise_level):
        if peak_kw == 0: return pd.Series(0, index=time_steps)
        hours = np.array([t * time_step_hours for t in time_steps])
        peak1 = norm.pdf(hours, loc=8.5, scale=1.5)
        peak2 = norm.pdf(hours, loc=18, scale=2.5)
        base_demand = (peak1 * 0.6 + peak2) / np.max(peak1 * 0.6 + peak2) * peak_kw
        noise = np.random.normal(0, peak_kw * noise_level, len(time_steps))
        stochastic_demand = np.maximum(0, base_demand + noise)
        return pd.Series(stochastic_demand, index=time_steps)

    def load_all(self):
        print("=" * 30);
        print("开始创建场景...")
        config = self.config
        random.seed(42);
        np.random.seed(42)

        # --- MODIFIED: 根据config选择路网模型 ---
        if config.ROAD_MODEL == 'waxman':
            G, pos = road_network.generate_city_like_graph(config.CITY_NODE_COUNT, alpha=config.WAXMAN_ALPHA,
                                                           beta=config.WAXMAN_BETA)
            scaling_factor = config.CITY_SCALE_KM
        elif config.ROAD_MODEL == 'real_world':
            G = road_network.generate_real_road_network(city_name=config.CITY_NAME)
            pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
            scaling_factor = 1  # OSMnx距离单位是米，会在get_path_and_distance_matrices中转换为km
        else:
            raise ValueError(f"未知的路网模型: {config.ROAD_MODEL}")

        self.road_network = G

        try:
            net = pp.networks.case118()
            load_buses = net.load.bus.to_list();
            print(f"成功加载IEEE 118节点系统，找到 {len(load_buses)} 个负荷节点。")
        except Exception as e:
            print(f"加载pandapower case118失败: {e}。将使用备用节点列表。");
            load_buses = list(range(1, 119))

        all_nodes = list(G.nodes())
        random.shuffle(all_nodes)

        num_facilities = config.NUM_DEPOTS + config.NUM_STATIONS + config.NUM_CUSTOMERS
        if len(all_nodes) < num_facilities:
            raise ValueError(f"路网节点数 ({len(all_nodes)}) 不足以容纳所有设施点 ({num_facilities})")

        depot_nodes = all_nodes[:config.NUM_DEPOTS]
        station_nodes = all_nodes[config.NUM_DEPOTS: config.NUM_DEPOTS + config.NUM_STATIONS]
        customer_nodes = all_nodes[
                         config.NUM_DEPOTS + config.NUM_STATIONS: config.NUM_DEPOTS + config.NUM_STATIONS + config.NUM_CUSTOMERS]

        locations_data = {};
        node_id_to_name_map = {}
        for i, node_id in enumerate(depot_nodes):
            name = f"Depot_{i + 1}";
            locations_data[name] = {'type': 'Depot', 'node_id': node_id};
            node_id_to_name_map[node_id] = name
        for i, node_id in enumerate(customer_nodes):
            name = f"Customer_{i + 1}";
            locations_data[name] = {'type': 'Customer', 'node_id': node_id};
            node_id_to_name_map[node_id] = name

        stations_info = {};
        station_to_bus_map = {}
        random.shuffle(load_buses)
        for i, node_id in enumerate(station_nodes):
            name = f"Station_{i + 1}";
            locations_data[name] = {'type': 'SwapStation', 'node_id': node_id};
            node_id_to_name_map[node_id] = name
            bus_id = load_buses[i % len(load_buses)];
            stations_info[name] = {'initial_full': 10, 'initial_empty': 5, 'bus_id': bus_id};
            station_to_bus_map[name] = bus_id

        print(f"场景设施点已放置: {len(depot_nodes)}仓库, {len(station_nodes)}换电站, {len(customer_nodes)}客户。")

        all_location_node_ids = [d['node_id'] for d in locations_data.values()]
        dist_matrix_raw, path_matrix_nodes = road_network.get_path_and_distance_matrices(G, all_location_node_ids)

        dist_matrix_renamed = dist_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)
        dist_matrix = dist_matrix_renamed * scaling_factor

        avg_speed_kmh = 40.0
        time_df = dist_matrix / avg_speed_kmh

        vehicles = {};
        depot_names = [name for name, info in locations_data.items() if info['type'] == 'Depot']
        for i in range(config.NUM_TRUCKS):
            truck_id = f"HDT_{i + 1}";
            vehicles[truck_id] = {'initial_soc': config.HDT_BATTERY_CAPACITY_KWH,
                                  'depot_id': random.choice(depot_names)}
        tasks = {};
        task_id_counter = 1;
        customer_names = [name for name, info in locations_data.items() if info['type'] == 'Customer']
        for cust_name in customer_names:
            task_id = f"Task_{task_id_counter}";
            closest_depot = min(depot_names, key=lambda d: dist_matrix.loc[d, cust_name])
            one_way_time = time_df.loc[closest_depot, cust_name];
            earliest_due = one_way_time + config.LOADING_UNLOADING_TIME_HOURS + 1.0
            latest_due = earliest_due + 8.0;
            due_time = round(random.uniform(earliest_due, latest_due), 1)
            tasks[task_id] = {'delivery_to': cust_name, 'demand': round(random.uniform(1.0, 2.5), 2),
                              'due_time': due_time, 'depot': closest_depot}
            task_id_counter += 1
        print(f"已生成 {len(vehicles)} 辆卡车和 {len(tasks)} 个任务的中央任务池。")

        time_steps = range(config.TOTAL_TIME_STEPS)
        electricity_prices = pd.Series([
                                           0.4 if 7 <= t * config.TIME_STEP_HOURS < 11 or 14 <= t * config.TIME_STEP_HOURS < 19 else (
                                               1.2 if 11 <= t * config.TIME_STEP_HOURS < 14 or 19 <= t * config.TIME_STEP_HOURS < 21 else 0.8)
                                           for t in time_steps], index=time_steps)
        pv_generation = {
            s: self.get_stochastic_pv_generation(time_steps, config.TIME_STEP_HOURS, config.PV_PEAK_POWER_KW,
                                                 config.PV_CLOUD_NOISE_LEVEL) for s in stations_info}
        ev_demand_timestep = {
            s: self.get_stochastic_ev_demand(time_steps, config.TIME_STEP_HOURS, config.EV_DEMAND_PEAK_KW,
                                             config.EV_DEMAND_NOISE_LEVEL) for s in stations_info}

        # 电压安全约束：需要充电站上下节点的v[i,t]
        voltage_pre =  {
            s: self.get_voltage_pre(time_steps, config.VOLTAGE_MAX) for s in stations_info}

        # voltage_next =  {
        #     s: self.get_voltage_next(time_steps, config.VOLTAGE_MAX) for s in stations_info}
        self.model_data = {
            'traffic_graph': G, 'locations': locations_data, 'tasks': tasks, 'vehicles': vehicles,
            'stations': stations_info,
            'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': None,
            'power_grid_net': net, 'station_to_bus_map': station_to_bus_map, 'time_steps': list(time_steps),
            'electricity_prices': electricity_prices, 'pv_generation': pv_generation,
            'ev_demand_timestep': ev_demand_timestep,'voltage_pre': voltage_pre,
            # 'voltage_next': voltage_next
        }
        print("=" * 30);
        print("场景数据创建完成！");
        return self.model_data

def run_power_flow_over_time(
    data,
    swap_g_e_df: pd.DataFrame,   # 列: ['station','time','grid_to_bess_kw']   —— 你已有
    swap_pv_g_df: pd.DataFrame,  # 列: ['station','time','pv_to_grid_kw']     —— 你已有
    swap_vs_df: pd.DataFrame = None,  # 可选，若你想用 BESS p_mw = 站内换电量/持续时间
    bess_discharge_kw_col: str = 'swap_bess_kwh',  # swap_vs_df 的功率/能量列名（按你表命名调整）
    time_step_hours: float = 0.25
):
    """
    将 MILP 的功率决策写入 pandapower 网络，按时间步运行潮流，返回电压与线路负荷结果。
    说明：
      - grid_to_bess_kw : 购电（负荷，写到 load.p_mw，单位转换 kW->MW）
      - pv_to_grid_kw   : 售电（分布式发电，写到 sgen.p_mw，注意发电为正，故取 +）
      - 可选：BESS 的 p_mw 可用 swap_vs_df 近似，或留 0 由能量约束自己管
    返回：
      - bus_vm_df: 列 ['time','bus','vm_pu']
      - line_loading_df: 列 ['time','line','loading_percent']
    """
    net = data["power_grid_net"]
    if net is None:
        raise RuntimeError("power_grid_net 为空，请确认在 data.pkl 中已保存 pandapower 网络。")

    load_map    = data["pp_load_index_map"]
    sgen_map    = data["pp_sgen_index_map"]
    storage_map = data.get("pp_storage_index_map", {})

    time_steps = sorted(data["time_steps"])
    # 为避免 query 性能问题，先构建透视便于快速读取
    g2e = swap_g_e_df.pivot(index='time', columns='station', values='grid_to_bess_kw').fillna(0.0)
    pvg = swap_pv_g_df.pivot(index='time', columns='station', values='pv_to_grid_kw').fillna(0.0)
    if swap_vs_df is not None and bess_discharge_kw_col in swap_vs_df.columns:
        vs  = swap_vs_df.pivot(index='time', columns='station', values=bess_discharge_kw_col).fillna(0.0)
    else:
        vs  = None

    # 结果收集
    vm_records = []
    ld_records = []

    # 可选：指定潮流选项提高鲁棒性（配电网常用 BFS 等）
    pp.set_user_pf_options(net, calculate_voltage_angles=False, init="results", tolerance_mva=1e-6)

    for t in time_steps:
        # 1) 写负荷（购电为正）
        for s, idx in load_map.items():
            p_kw = float(g2e.at[t, s]) if (t in g2e.index and s in g2e.columns) else 0.0
            net.load.at[idx, 'p_mw'] = max(0.0, p_kw) / 1000.0

        # 2) 写分布式发电（售电为正）
        for s, idx in sgen_map.items():
            p_kw = float(pvg.at[t, s]) if (t in pvg.index and s in pvg.columns) else 0.0
            net.sgen.at[idx, 'p_mw'] = max(0.0, p_kw) / 1000.0

        # 3)（可选）写 BESS 功率，正为放电
        if vs is not None:
            for s, idx in storage_map.items():
                p_kw = float(vs.at[t, s]) if (t in vs.index and s in vs.columns) else 0.0
                net.storage.at[idx, 'p_mw'] = p_kw / 1000.0

        # 4) 运行潮流
        try:
            pp.runpp(net)
        except Exception:
            # 如果失败，降级用更稳健初始化再跑一次
            pp.runpp(net, init="flat", enforce_q_lims=True, tolerance_mva=1e-6)

        # 5) 记录母线电压与线路负荷
        for bus_idx, row in net.res_bus.iterrows():
            vm_records.append((t, int(bus_idx), float(row.vm_pu)))
        for line_idx, row in net.res_line.iterrows():
            loading = float(row.loading_percent) if not np.isnan(row.loading_percent) else 0.0
            ld_records.append((t, int(line_idx), loading))

    bus_vm_df = pd.DataFrame(vm_records, columns=['time', 'bus', 'vm_pu'])
    line_loading_df = pd.DataFrame(ld_records, columns=['time', 'line', 'loading_percent'])
    return bus_vm_df, line_loading_df

def test_scale(data, coff =10):
    data['dist_matrix'] = {pre: val.to_dict() for pre,val in data['dist_matrix'].items()}
    data['time_matrix'] = {pre: val.to_dict() for pre, val in data['time_matrix'].items()}
    dist_matrix = data['dist_matrix']
    time_matrix = data['time_matrix']
    for pre in dist_matrix:
        for next in dist_matrix[pre]:
            dist_matrix[pre][next]  *= coff
            time_matrix[pre][next] *= coff

def buildModel_Case1(data, config):
    # 默认一个任务就是一个节点
    test_scale(data, coff=8)
    # 时间字典
    time_steps = range(config.TOTAL_TIME_STEPS)
    hours = [t * config.TIME_STEP_HOURS for t in time_steps]
    timetonum = {h: t for t, h in enumerate(hours)}
    numtotime = {t: h for t, h in enumerate(hours)}
    # 根据配送任务，过滤仓库
    depot_nodes = set([node_info['depot']  for node_id, node_info in data['tasks'].items() ])
    depot_vehicles, vehicel_depot = defaultdict(list),{}
    for vehicle_id, vehicle_info in data['vehicles'].items():
        if vehicle_info['depot_id'] in depot_nodes:
            depot_vehicles[vehicle_info['depot_id']].append(vehicle_id)
            vehicel_depot[vehicle_id] = vehicle_info['depot_id']
    task_depot, depot_tasks = dict(), defaultdict(list)
    for task_id, task_info in data['tasks'].items():
        task_depot[task_id] = task_info['depot']
        depot_tasks[task_info['depot']].append(task_id)
    # PV功率
    pv_dict, voltage_pre_dict = {},{}
    for station in data['pv_generation']:
        for step in time_steps:
            pv_dict[station, step] = data['pv_generation'][station][step]
            voltage_pre_dict[station,step] = data['voltage_pre'][station][step]
    #电价
    grid_price = data['electricity_prices']
    stations = list(data['stations'].keys())
    M = 1e6

    model = gp.Model("Model")
    # model.setParam("TimeLimit", 60) # 设置求解时间
    model.setParam("MipGap", 0.20)  #设置求解gap
    #建立变量
    varDict = {}
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        varDict['v',vehicle_id] = model.addVar(vtype=GRB.BINARY, name="v_%s" % vehicle_id) #车辆v是否启用
        varDict['vmt',vehicle_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name="vmt_%s" % vehicle_id) #v的最大工作时长
        # 仓库需要虚拟为两个节点，出一次，进一次
        depots = [depot +'_out', depot +'_in']
        for task_id in depot_tasks[depot] + depots + stations:
            varDict['vv',(vehicle_id,task_id)] = model.addVar(vtype=GRB.BINARY, name="vv_%s_%s" % (vehicle_id, task_id)) #v车是否服务任务t
            varDict['Wt',(vehicle_id,task_id)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="Wt_%s_%s" % (vehicle_id, task_id)) #v过任务点t时的载重量
            if task_id in stations:
                varDict['SOCt',(vehicle_id,task_id+'_in')] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name="SOCt_%s_%s" % (vehicle_id, task_id+'_in')) #v到达任务点t时的SOC
                varDict['SOCt',(vehicle_id,task_id+'_out')] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name="SOCt_%s_%s" % (vehicle_id, task_id+'_out')) #v到达任务点t时的SOC
            else:
                varDict['SOCt',(vehicle_id,task_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name="SOCt_%s_%s" % (vehicle_id, task_id)) #v到达任务点t时的SOC
            varDict['vt',(vehicle_id,task_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name="vt_%s_%s" % (vehicle_id, task_id)) #v过任务点t时的到达时间
            for node in depot_tasks[depot] + [depot +'_in'] + stations:
                if node == depot +'_in' and task_id == depot + '_out':
                    continue
                if  task_id == depot + '_in':
                    continue
                if node != task_id :
                    varDict['vx',(vehicle_id,task_id,node)] = model.addVar(vtype=GRB.BINARY, name="vx_%s_%s_%s" % (vehicle_id, task_id, node))

    for task_id in task_depot:
        varDict['df',task_id] = model.addVar(vtype=GRB.BINARY, name="df_%s" % task_id) #任务t是否完成

    for station in stations:
        for t in numtotime:
            for vehicle_id in vehicel_depot:
                varDict['swap_v',(vehicle_id,station,t)] = model.addVar(vtype=GRB.BINARY, name="swap_v_%s_%s_%s" % (vehicle_id, station, t))
                varDict['swap_vs',(vehicle_id,station,t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_vs_%s_%s_%s" % (vehicle_id, station, t))
            varDict['swap_s',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_s_%s_%s" % (station, t)) # 充电站t时刻的换电需求量
            varDict['swap_pv_e',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_pv_e_%s_%s" % (station, t))# 充电站t时刻PV冲电池的功率
            varDict['swap_pv_g',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_pv_g_%s_%s" % (station, t))# 充电站t时刻PV卖电网的功率
            varDict['swap_pv_g_f',(station, t)] = model.addVar(vtype=GRB.BINARY, name="swap_pv_g_f_%s_%s" % (station, t))# 充电站t时刻PV是否向电网卖电
            varDict['swap_pv_s',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_pv_s_%s_%s" % (station, t))# 充电站t时刻PV浪费的功率
            varDict['swap_g_e',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_g_e_%s_%s" % (station, t)) #充电站t时刻向电网的买电功率
            varDict['swap_e',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_e_%s_%s" % (station, t)) #充电站t时刻的电量水位迭代(仅与PV，grid的电量有关)
            varDict['vol_pre',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="vol_pre_%s_%s" % (station, t)) #充电站电网前驱节点电压
            varDict['vol',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="vol_%s_%s" % (station, t)) #充电站电网节点电压
            varDict['vol_next',(station, t)] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="vol_next_%s_%s" % (station, t)) #充电站电网后继节点电压
        varDict['swap_e',(station, len(numtotime))] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name="swap_e_%s_%s" % (station, t)) #充电站t时刻的电量水位迭代(仅与PV，grid的电量有关)
    # 车辆换电时间离散化
    for station in stations:
        for vehicle_id in vehicel_depot:
            model.addConstr(gp.quicksum(varDict['swap_v',(vehicle_id, station, t)] for t in numtotime) == varDict['vv',(vehicle_id,station)])
        for t in numtotime:
            for vehicle_id in vehicel_depot:
                model.addConstr(varDict['swap_v',(vehicle_id,  station, t)] * numtotime[t] <= varDict['vt',(vehicle_id, station)])
                model.addConstr(varDict['swap_v',(vehicle_id,  station, t)] * numtotime.get(t+1, max(list(timetonum.keys()))) >= varDict['vt',(vehicle_id, station)] - M * (1 - varDict['swap_v',(vehicle_id,  station, t)]))
                model.addConstr(varDict['swap_vs',(vehicle_id,  station, t)] <= M * varDict['swap_v',(vehicle_id,station,t)])
                model.addConstr(varDict['swap_vs',(vehicle_id,  station, t)] <= varDict['SOCt',(vehicle_id,station+'_out')] - varDict['SOCt',(vehicle_id,station+'_in')])
                model.addConstr(varDict['swap_vs',(vehicle_id,  station, t)] >= varDict['SOCt',(vehicle_id,station+'_out')] - varDict['SOCt',(vehicle_id,station+'_in')]- M * (1 - varDict['swap_v',(vehicle_id,  station, t)]))
            # 计算station t时刻总换电量
            model.addConstr( varDict['swap_s',(station, t)] == gp.quicksum( varDict['swap_vs',(vehicle_id,  station, t)] for vehicle_id in vehicel_depot))
            # t时刻PV的使用量
            model.addConstr(varDict['swap_pv_e',(station,t)] + varDict['swap_pv_g',(station,t)] + varDict['swap_pv_s',(station,t)] == pv_dict.get((station,t),0))
            # t时刻，PV卖电就不能买电
            model.addConstr(varDict['swap_pv_g',(station,t)]  <= M * varDict['swap_pv_g_f',(station, t)])
            model.addConstr(varDict['swap_g_e',(station,t)]  <= M * (1 - varDict['swap_pv_g_f',(station, t)]))
            # 电量水位迭代
            if t == 0:
                model.addConstr( varDict['swap_e',(station,t)]  == data['stations'][station]['initial_empty'])
            else:
                model.addConstr(varDict['swap_e',(station,t)] == varDict['swap_e',(station,t-1)] + (varDict['swap_pv_e',(station,t-1)] +
                            varDict['swap_g_e',(station,t-1)]) * config.SWAP_DURATION_HOURS - varDict['swap_s',(station,t-1)])
            # 电压压降方程

            model.addConstr(varDict['vol_pre',(station, t)] ==voltage_pre_dict.get((station, t), 0) -
                          2*config.VOLTAGE_DESCEND*varDict.get(('swap_g_e',(station,t)),0) )
            model.addConstr(varDict['vol',(station, t)] == voltage_pre_dict.get((station, t), 0) -
                          2*config.VOLTAGE_DESCEND*(varDict.get(('swap_g_e',(station,t)),0) - varDict.get(('swap_pv_g',(station,t)),0)))
            model.addConstr(varDict['vol_next',(station, t)] == varDict['vol',(station, t)] + 2 * config.VOLTAGE_DESCEND * varDict.get(('swap_pv_g',(station,t)),0))

            # 电压安全约束
            model.addConstr(varDict['vol',(station, t)] >= config.VOLTAGE_MIN)
            model.addConstr(varDict['vol',(station, t)] <= config.VOLTAGE_MAX)
            model.addConstr(varDict['vol_next',(station, t)] >= config.VOLTAGE_MIN)
            model.addConstr(varDict['vol_next',(station, t)] <= config.VOLTAGE_MAX)
            model.addConstr(varDict['vol_pre',(station, t)] >= config.VOLTAGE_MIN)
            model.addConstr(varDict['vol_pre',(station, t)] <= config.VOLTAGE_MAX)
            # 热稳定性
            model.addConstr(varDict.get(('swap_g_e',(station,t)),0) <= config.THERMAL_STABILITY_MAX)
        # model.addConstr(varDict['swap_e',(station,len(numtotime))] == 0)
        model.addConstr(varDict['swap_e',(station,len(numtotime))] == varDict['swap_e',(station,len(numtotime)-1)] + (varDict['swap_pv_e',(station,len(numtotime)-1)] +
                            varDict['swap_g_e',(station,len(numtotime)-1)]) * config.SWAP_DURATION_HOURS - varDict['swap_s',(station,len(numtotime)-1)])



    # 车辆路过节点才能提供服务
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        depots = [depot +'_out', depot +'_in']
        var_lists = []
        for task_id in depot_tasks[depot] + [depot +'_out'] + stations:
            var_lists = []
            for node in depot_tasks[depot] + [depot +'_in'] + stations:
                if node != task_id :
                    var_lists.append(varDict.get(('vx',(vehicle_id,task_id,node)),0))
            model.addConstr(gp.quicksum(v for v in var_lists) == varDict['vv',(vehicle_id,task_id)])
        var_lists = []
        for task_id in depot_tasks[depot] + [depot +'_out'] + stations:
            var_lists.append(varDict.get(('vx',(vehicle_id,task_id,depot +'_in')),0))
        model.addConstr(gp.quicksum(v for v in var_lists) == varDict['vv',(vehicle_id,depot +'_in')])

    # 每个需求点最多被访问一次
    for task_id in task_depot:
        model.addConstr(gp.quicksum(varDict['vv',(vehicle_id,task_id)] for vehicle_id in depot_vehicles[task_depot[task_id]]) == varDict['df',task_id])
        model.addConstr(varDict['df',task_id] == 1)
    # 每台车对节点进度等于出度, 且度=r[v,d]
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        depots = [depot +'_out', depot +'_in']
        for task_id in depot_tasks[depot] + stations + [depot+'_out']:
            var_list_entrys, var_list_outputs = [],[]
            for node in depot_tasks[depot] + depots + stations:
                if node != task_id :
                    var_list_outputs.append(varDict.get(('vx',(vehicle_id,task_id,node)),0))
                    var_list_entrys.append(varDict.get(('vx',(vehicle_id,node,task_id)),0))
            if task_id != depot + '_out' :
                model.addConstr(gp.quicksum(v for v in var_list_entrys) == gp.quicksum(v for v in var_list_outputs))
            model.addConstr(gp.quicksum(v for v in var_list_outputs) == varDict['vv',(vehicle_id,task_id)])
        # 仓库出度 = 车辆是否被启用
        var_list_entrys, var_list_outputs = [],[]
        for node in depot_tasks[depot] :
            var_list_outputs.append(varDict['vx',(vehicle_id,depot+'_out',node)])
            var_list_entrys.append(varDict['vx',(vehicle_id,node,depot+'_in')])
        model.addConstr(gp.quicksum(v for v in var_list_entrys) == gp.quicksum(v for v in var_list_outputs))
        model.addConstr(gp.quicksum(v for v in var_list_entrys) == varDict['v',vehicle_id])
        model.addConstr(gp.quicksum(varDict['vv',(vehicle_id,task_id)] for task_id in depot_tasks[depot] + depots + stations) <= varDict['v',vehicle_id] * M)


    # 车辆载重迭代&最后空车到仓库
    # Wt[v,j] = sum_i (Wt[v,i] - d[i]) * vx[v,i,j]  -> Wt[v,j] >= (Wt[v,i] - d[i])  - (1-x[v,i,j]) * M & Wt[v,j] <= vx[v,i,j] * M
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        depots = [depot +'_out', depot +'_in']
        for node in depot_tasks[depot] + stations + [depot +'_in']: #next
            if node == depot + '_in': #回到仓库，为0
                model.addConstr(varDict['Wt',(vehicle_id,node)] == config.HDT_EMPTY_WEIGHT_TON * varDict['v', vehicle_id])
            demand_qty = data['tasks'].get(node,{}).get('demand',0)
            for task_id in depot_tasks[depot] + stations + [depot + '_out']: # pre
                if node != task_id:
                    model.addConstr(varDict['Wt',(vehicle_id,task_id)] >= varDict['Wt',(vehicle_id,node)] + demand_qty - (1-varDict.get(('vx',(vehicle_id,task_id,node)),0)) * M)
                    model.addConstr(varDict['Wt',(vehicle_id,task_id)] <= varDict['Wt',(vehicle_id,node)] + demand_qty + (1-varDict.get(('vx',(vehicle_id,task_id,node)),0)) * M)
                model.addConstr(varDict['Wt',(vehicle_id,task_id)] <= varDict['vv', (vehicle_id, task_id)] * M )
            model.addConstr(varDict['Wt',(vehicle_id,node)] <= varDict['vv', (vehicle_id, node)] * M )
    def nodetoLoc(node, task_depot, data):
        loc = 'defaultLOC'
        if node in task_depot:
            loc = data['tasks'][node]['delivery_to']
        elif node in data['stations']:
            loc = node
        else:
            loc = node.split('_')[0]  + '_' + node.split('_')[1]
        return loc

    # 车辆到达节点时间
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        depots = [depot +'_out', depot +'_in']
        for node in depot_tasks[depot] + stations + depots: #i
            if node == depot + '_out': #出仓库，为0
                model.addConstr(varDict['vt',(vehicle_id,node)] == 0)
                # continue
            s_loc = nodetoLoc(node, task_depot, data)
            for task_id in depot_tasks[depot] + stations + [depot + '_in']:
                t_loc = nodetoLoc(task_id, task_depot, data)
                dur = data['time_matrix'].get(s_loc,{}).get(t_loc,0)
                if node != task_id:
                    model.addConstr(varDict['vt',(vehicle_id,task_id)] >= varDict['vt',(vehicle_id,node)] + dur - (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                model.addConstr(varDict['vt',(vehicle_id,task_id)] <= varDict['vv', (vehicle_id, task_id)] * M)
            model.addConstr(varDict['vt',(vehicle_id,node)] <= varDict['vmt',vehicle_id])
            # if node in task_depot:
            #     model.addConstr(varDict['vt',(vehicle_id,node)] <= data['tasks'][node]['due_time'] * varDict['df',node])
        # 车辆如果被启用，必须满足最小服务数量&最大服务数量
        model.addConstr(gp.quicksum(varDict['vv',(vehicle_id,task_id)] for task_id in depot_tasks[depot]) <= config.MAX_TASKS_PER_TRUCK * varDict['v',vehicle_id])
        model.addConstr(gp.quicksum(varDict['vv',(vehicle_id,task_id)] for task_id in depot_tasks[depot]) >= config.MIN_TASKS_PER_TRUCK * varDict['v',vehicle_id])
    # 车辆SOC的迭代
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        depots = [depot +'_out', depot +'_in']
        for node in depot_tasks[depot] + [depot +'_out']: #pre
            if node == depot + '_out': #出仓库，为初始电力
                model.addConstr(varDict['SOCt',(vehicle_id,node)] == data['vehicles'][vehicle_id]['initial_soc'])
            s_loc = nodetoLoc(node, task_depot, data)
            for task_id in depot_tasks[depot] + stations + [depot + '_in']: # next
                t_loc = nodetoLoc(task_id, task_depot, data)
                dist = data['dist_matrix'].get(s_loc,{}).get(t_loc,0)
                if node != task_id:
                    if task_id in stations:
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id+'_in')] >= varDict['SOCt',(vehicle_id,node)] - dist * (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) - (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id+'_in')] <= varDict['SOCt',(vehicle_id,node)] - dist * (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) + (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id+'_in')] >= config.HDT_MIN_SOC_KWH * varDict['vv',(vehicle_id,task_id)])
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id+'_in')] <= config.HDT_BATTERY_CAPACITY_KWH * varDict['vv',(vehicle_id,task_id)])
                    else:
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id)] >= varDict['SOCt',(vehicle_id,node)] - dist *
                                        (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) - (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id)] <= varDict['SOCt',(vehicle_id,node)] - dist *
                                        (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) + (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id)] >= config.HDT_MIN_SOC_KWH * varDict['vv',(vehicle_id,task_id)])
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id)] <= config.HDT_BATTERY_CAPACITY_KWH * varDict['vv',(vehicle_id,task_id)])
        for node in stations:
            s_loc = nodetoLoc(node, task_depot, data)
            for task_id in depot_tasks[depot] + stations + [depot + '_in']:
                t_loc = nodetoLoc(task_id, task_depot, data)
                dist = data['dist_matrix'].get(s_loc,{}).get(t_loc,0)
                if node != task_id:
                    if task_id in stations:
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id+'_in')] >= varDict['SOCt',(vehicle_id,node +'_out')] - dist *
                                            (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) - (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id+'_in')] <= varDict['SOCt',(vehicle_id,node+'_out')] - dist *
                                            (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) + (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                    else:
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id)] >= varDict['SOCt',(vehicle_id,node +'_out')] - dist *
                                            (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) - (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                        model.addConstr(varDict['SOCt',(vehicle_id,task_id)] <= varDict['SOCt',(vehicle_id,node+'_out')] - dist *
                                            (config.HDT_BASE_CONSUMPTION_KWH_PER_KM + config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON * varDict['Wt',(vehicle_id,node)]) + (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
            model.addConstr(varDict['SOCt',(vehicle_id,node+'_out')] == config.HDT_BATTERY_CAPACITY_KWH * varDict['vv',(vehicle_id,node)])
    obj1 = gp.quicksum(varDict['vmt',vehicle_id] * config.MANPOWER_COST_PER_HOUR for vehicle_id in vehicel_depot) #人力成本
    obj2 = gp.quicksum(gp.quicksum(varDict['vv',(vehicle_id, node)] * config.FIXED_SWAP_COST for node in stations )for vehicle_id in vehicel_depot) #换电成本
    obj3 = gp.quicksum((1-varDict['df',task_id]) * config.UNASSIGNED_TASK_PENALTY for task_id in depot_tasks[depot])
    obj4 = gp.quicksum(gp.quicksum( (varDict['swap_g_e',(station,t)] - varDict['swap_pv_g',(station,t)])* grid_price[t] for t in numtotime) for station in stations)
    model.setObjective(obj1 + obj2 + obj3 + obj4,sense=GRB.MINIMIZE)
    # model.writeProblem("D:/model.lp")
    # model.setRealParam('limits/time', 300)
    # model.setRealParam('limits/gap', 0.1)
    # ==== 求解 ====
    # ==== 求解 ====
    # model.Params.TimeLimit = 300
    model.optimize()

    status = model.Status
    if status in [GRB.INF_OR_UNBD, GRB.UNBOUNDED]:
        model.setParam("DualReductions", 0)
        model.optimize()
        status = model.Status

    if status == GRB.INFEASIBLE:
        # 计算并导出 IIS（注意：不要写 .iis 扩展）
        model.computeIIS()
        # 推荐同时导出 LP 和 MPS，LP 内会标出 IIS 成员
        model.write("model.ilp")
        model.write("model.lp")
        model.write("model.mps")
        # 打印 IIS 详情，定位具体冲突
        print("\n===== IIS DETAILS (Constraints) =====")
        for c in model.getConstrs():
            try:
                if c.IISConstr:
                    print(f"[IIS-CONSTR] {c.ConstrName}")
            except Exception:
                pass
        print("\n===== IIS DETAILS (Variable Bounds) =====")
        for v in model.getVars():
            try:
                if v.IISLB:
                    print(f"[IIS-BOUND-LB] {v.VarName} at LB = {v.LB}")
                if v.IISUB:
                    print(f"[IIS-BOUND-UB] {v.VarName} at UB = {v.UB}")
            except Exception:
                pass
        raise RuntimeError("模型不可行(INFEASIBLE)。已导出 model.ilp / model.lp / model.mps，并打印 IIS 详情。")

    # if status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    #     raise RuntimeError(f"求解结束但无可行解可读，Gurobi 状态码：{status}")

    # ==== 安全读取变量值 ====
    def _safe_val(v):
        try:
            return v.X
        except Exception:
            try:
                return v.Xn
            except Exception:
                return 0.0

    init_sol = {key: _safe_val(var) for key, var in varDict.items()}

    v_values, vv_values, wt_values, soct_values, vt_values, vx_values,vmt_values, df_values = [],[],[],[],[],[],[],[]
    swap_s,swap_pv_e, swap_pv_g,swap_g_e,swap_v,swap_vs,swap_pv_g_f, swap_pv_s,swap_e,vol,vol_next = [],[],[],[],[],[],[],[],[],[],[]
    for task_id in task_depot:
        demand = data['tasks'][task_id]['demand']
        due_time = data['tasks'][task_id]['due_time']
        df_values.append((task_id, due_time,demand,init_sol[('df',task_id)]))
    df_df = pd.DataFrame(df_values,columns=['task_id', 'due_time', 'demand', 'is_assigned'])
    for vehicle_id in vehicel_depot:
        depot = vehicel_depot[vehicle_id]
        init_soc = data['vehicles'][vehicle_id]['initial_soc']
        init_weight = config.HDT_EMPTY_WEIGHT_TON
        v_values.append((vehicle_id,init_soc,init_weight,init_sol['v',vehicle_id]))
        vmt_values.append((vehicle_id,init_sol[('vmt',vehicle_id)]))
        depots = [depot +'_out', depot +'_in']
        for task_id in depot_tasks[depot] + depots + stations:
            vv_values.append((vehicle_id,task_id,init_sol[('vv',(vehicle_id,task_id))]))
            wt_values.append((vehicle_id,task_id,init_sol[('Wt',(vehicle_id,task_id))]))
            if task_id in stations:
                soct_values.append((vehicle_id,task_id+'_in',init_sol[('SOCt',(vehicle_id,task_id+'_in'))]))
                soct_values.append((vehicle_id,task_id+'_out',init_sol[('SOCt',(vehicle_id,task_id+'_out'))]))
            else:
                soct_values.append((vehicle_id,task_id,init_sol[('SOCt',(vehicle_id,task_id))]))
            vt_values.append((vehicle_id,task_id,init_sol[('vt',(vehicle_id,task_id))]))
            for node in depot_tasks[depot] + [depot +'_in'] + stations:
                if node == depot +'_in' and task_id == depot + '_out':
                    continue
                if  task_id == depot + '_in':
                    continue
                if node != task_id :
                    vx_values.append((vehicle_id,task_id,node,init_sol[('vx',(vehicle_id,task_id,node))]))

    for station in stations:
        for t in numtotime:
            for vehicle_id in vehicel_depot:
                swap_v.append((vehicle_id,station,t, init_sol[('swap_v',(vehicle_id,station,t))]))
                swap_vs.append((vehicle_id,station,t, init_sol[('swap_vs',(vehicle_id,station,t))]))
            swap_s.append((station,t, init_sol[('swap_s',(station,t))]))
            swap_e.append((station,t, init_sol[('swap_e',(station,t))]))
            swap_pv_e.append((station,t, init_sol[('swap_pv_e',(station,t))]))
            swap_pv_g.append((station,t, init_sol[('swap_pv_g',(station,t))]))
            swap_g_e.append((station,t, init_sol[('swap_g_e',(station,t))]))
            swap_pv_g_f.append((station,t, init_sol[('swap_pv_g_f',(station,t))]))
            swap_pv_s.append((station,t, init_sol[('swap_pv_s',(station,t))]))
            vol.append((station,t, init_sol[('vol',(station,t))]))
            vol_next.append((station,t, init_sol[('vol_next',(station,t))]))
        swap_e.append((station,len(numtotime), init_sol[('swap_e',(station,len(numtotime)))]))
    swap_v_df = pd.DataFrame(swap_v,columns=['vehicle_id', 'station', 'time', 'is_swap_bess'])
    swap_vs_df = pd.DataFrame(swap_vs,columns=['vehicle_id', 'station', 'time', 'swap_bess_kwh'])
    swap_pv_e_df = pd.DataFrame(swap_pv_e,columns=['station', 'time', 'pv_to_bess_kw'])
    swap_pv_g_df = pd.DataFrame(swap_pv_g,columns=['station', 'time', 'pv_to_grid_kw'])
    swap_g_e_df = pd.DataFrame(swap_g_e,columns=['station', 'time', 'grid_to_bess_kw'])
    swap_pv_g_f_df = pd.DataFrame(swap_pv_g_f,columns=['station', 'time', 'pv_to_grid_flag'])
    swap_pv_s_df = pd.DataFrame(swap_pv_s,columns=['station', 'time', 'pv_to_waste'])
    swap_s_df = pd.DataFrame(swap_s,columns=['station', 'time', 'totalSwapKWH'])
    swap_e_df = pd.DataFrame(swap_e,columns=['station', 'time', 'StationKWH'])
    vol_df = pd.DataFrame(vol,columns=['station', 'time', 'vol'])
    vol_next_df = pd.DataFrame(vol_next,columns=['station', 'time', 'vol_next'])

    v_df = pd.DataFrame(v_values,columns=['vehicle_id', 'soc', 'weight', 'is_assigned'])
    vmt_df = pd.DataFrame(vmt_values,columns=['vehicle_id', 'vehicle_max_time'])
    vt_df = pd.DataFrame(vt_values,columns=['vehicle_id', 'task_id', 'arrive_time'])
    vv_df = pd.DataFrame(vv_values,columns=['vehicle_id', 'task_id', 'is_assigned'])
    wt_df = pd.DataFrame(wt_values,columns=['vehicle_id', 'task_id', 'weight'])
    soct_df = pd.DataFrame(soct_values,columns=['vehicle_id', 'task_id', 'soc'])
    # vs_df = pd.DataFrame(vs_values,columns=['vehicle_id', 'task_id', 'is_swap_bess'])
    vx_df = pd.DataFrame(vx_values,columns=['vehicle_id', 'task_id', 'node', 'is_connected'])
    otput_dict = {'v_df':v_df, 'vmt_df':vmt_df, 'vt_df':vt_df,
                  'vv_df':vv_df, 'wt_df':wt_df, 'soct_df':soct_df,
                   'vx_df':vx_df, 'df_df':df_df,
                   'swap_v_df':swap_v_df, 'swap_vs_df':swap_vs_df,
                   'swap_pv_e_df':swap_pv_e_df, 'swap_pv_g_df':swap_pv_g_df,
                   'swap_g_e_df':swap_g_e_df, 'swap_pv_g_f_df':swap_pv_g_f_df,
                   'swap_pv_s_df':swap_pv_s_df, 'swap_s_df':swap_s_df,
                   'vol_df':vol_df, 'vol_next_df':vol_next_df, 'swap_e_df':swap_e_df}
    return otput_dict
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    pkl_path = os.path.join(BASE_DIR, 'data.pkl')
    if os.path.exists(pkl_path):
        print(f"✅ 读取已有 data.pkl: {pkl_path}")
        data = pd.read_pickle(pkl_path)
    else:
        print("⚠️ 未找到 data.pkl，正在生成最小示例数据并保存...")
        # data = make_minimal_data(
        #     total_time_steps=8,
        #     time_step_hours=1.0,
        #     n_customers=6,
        #     n_stations=2,
        #     n_depots=2,
        #     n_vehicles=3
        # )
        data_loader = DataLoader(config)
        data = data_loader.load_all()
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ 已写入: {pkl_path}")


    output_dict = buildModel_Case1(data, config)

    # # ==== 优化后：调用潮流 ====
    # bus_vm_df, line_loading_df = run_power_flow_over_time(
    #     data,
    #     swap_g_e_df=output_dict['swap_g_e_df'],
    #     swap_pv_g_df=output_dict['swap_pv_g_df'],
    #     swap_vs_df=output_dict.get('swap_vs_df', None),
    #     bess_discharge_kw_col='swap_bess_kwh',
    #     time_step_hours=(data['TIME_STEP_HOURS'] if 'TIME_STEP_HOURS' in data
    #                      else (data['time_steps'][1] - data['time_steps'][0] if len(data['time_steps']) > 1 else 0.25))
    # )
    #
    # # 写Excel（把潮流结果也写进去）
    # with pd.ExcelWriter('algo_res.xlsx') as writer:
    #     for k, v in output_dict.items():
    #         v.to_excel(writer, sheet_name=k[:31], index=False)  # Excel sheet名≤31字符
    #     bus_vm_df.to_excel(writer, sheet_name='pf_bus_vm', index=False)
    #     line_loading_df.to_excel(writer, sheet_name='pf_line_loading', index=False)
    #
    # print("✅ 已写入 algo_res.xlsx（含潮流结果）")
    #
    # # 输出Excel
    with pd.ExcelWriter('algo_res.xlsx') as writer:
        for k, v in output_dict.items():
            v.to_excel(writer, sheet_name=k, index=False)
    print("✅ 结果已写入 algo_res.xlsx")
