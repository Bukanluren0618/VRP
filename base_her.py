import pandas as pd
import numpy as np
import os
import random
import pickle
from collections import defaultdict
import config as config
import pyscipopt as scip


class Her:
    def __init__(self, data, grid_data, config):
        self.data = data
        self.grid_input_data = grid_data
        self.config = config
        # self.preProcess() #预处理一下路网的数据

    def preProcess(self):
        # 预先根据空车载重，计算两点之间是否可以直连
        empty_weight = self.config.HDT_EMPTY_WEIGHT_TON
        base_consump_kwh_per_km = self.config.HDT_BASE_CONSUMPTION_KWH_PER_KM
        weight_consump_kwh_per_km = self.config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
        full_soc = self.config.HDT_BATTERY_CAPACITY_KWH
        min_soc = self.config.HDT_MIN_SOC_KWH
        # 修正距离矩阵
        time_matrix = self.data['time_matrix']
        dist_matrix = self.data['dist_matrix']
        stations = self.data['stations']
        for pre in dist_matrix:
            for next in dist_matrix[pre]:
                dist = dist_matrix.get(pre, {}).get(next, 0)
                if dist * (base_consump_kwh_per_km + weight_consump_kwh_per_km * empty_weight) >= full_soc - min_soc:
                    dist_matrix[pre].pop(next)
                    time_matrix[pre].pop(next)
        # 计算每个点最近的station
        nearestStation = {}
        for pre in dist_matrix:
            next_nodes = sorted(dist_matrix[pre], key=lambda x: x[1])
            for next in next_nodes:
                if next in stations:
                    nearestStation[pre] = dist_matrix.get(pre, {}).get(next, 0)
                    break
        # 计算两步距离: d[i,j] + nearestStation[j] <= full_soc - min_soc
        del_edges = []
        for pre in dist_matrix:
            for next in dist_matrix[pre]:
                dist_pre_next = dist_matrix.get(pre, {}).get(next, 0) + nearestStation[next]
                if dist_pre_next * (
                        base_consump_kwh_per_km + weight_consump_kwh_per_km * empty_weight) * 1.1 >= full_soc - min_soc:
                    del_edges.append((pre, next))
        for pre, next in del_edges:
            dist_matrix[pre].pop(next)
            time_matrix[pre].pop(next)
        print('del edge: ', len(del_edges))
        print('total edge: ', len(dist_matrix) * (len(dist_matrix) - 1))

    def vrp_route_schedule(self):  # 路网侧调度
        time_matrix = {key: val.to_dict() for key, val in self.data['time_matrix'].items()}
        dist_matrix = {key: val.to_dict() for key, val in self.data['dist_matrix'].items()}
        random.seed(10)
        np.random.seed(10)
        # 根据配送任务，过滤仓库
        depot_nodes = set([node_info['depot'] for node_id, node_info in self.data['tasks'].items()])
        depot_vehicles, vehicel_depot = defaultdict(list), {}
        for vehicle_id, vehicle_info in self.data['vehicles'].items():
            if vehicle_info['depot_id'] in depot_nodes:
                depot_vehicles[vehicle_info['depot_id']].append(vehicle_id)
                vehicel_depot[vehicle_id] = vehicle_info['depot_id']
        task_depot, depot_tasks = dict(), defaultdict(list)
        for task_id, task_info in self.data['tasks'].items():
            task_depot[task_id] = task_info['depot']
            depot_tasks[task_info['depot']].append(task_id)
        paths = {}
        vehicel_init_soc = {vehicle_id: self.data['vehicles'][vehicle_id]['initial_soc'] for vehicle_id in
                            vehicel_depot}
        full_soc = self.config.HDT_BATTERY_CAPACITY_KWH
        min_soc = self.config.HDT_MIN_SOC_KWH
        max_tasks_num = self.config.MAX_TASKS_PER_TRUCK
        empty_weight = self.config.HDT_EMPTY_WEIGHT_TON
        base_consump_kwh_per_km = self.config.HDT_BASE_CONSUMPTION_KWH_PER_KM
        weight_consump_kwh_per_km = self.config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
        unSatTasks = []

        def task_to_custom(task, data):
            if task in data['stations']:
                return task
            else:
                return data['tasks'][task]['delivery_to']

        for depot in depot_vehicles:
            cur_all_vehicles = depot_vehicles[depot].copy()
            cur_all_tasks = depot_tasks[depot].copy()
            all_stations = list(self.data['stations'].keys())
            nearestStation = {}
            for pre in cur_all_tasks:
                next_nodes = sorted(dist_matrix[task_to_custom(pre, data)], key=lambda x: x[1])
                for next in next_nodes:
                    if next in all_stations:
                        nearestStation[pre] = (next, dist_matrix.get(task_to_custom(pre, data), {}).get(next, 0),
                                               time_matrix.get(task_to_custom(pre, data), {}).get(next, 0))
                        break
            for vehicle_id in cur_all_vehicles:
                # 调度策略，如果当前节点i的SOC < (min_soc + soc_cost[i,ns]) * 1.2 -> 换电; 每个电站都换过一次后，
                # 并且soc[i] - soc_cost[i,j] - soc_cost[j,depot] >= min_soc * 1.5 或者达到最大配送数量 就回到仓库
                curpath = []
                cur_init_soc = vehicel_init_soc.get(vehicle_id, 0)
                if len(cur_all_tasks) == 0:
                    paths[vehicle_id] = curpath
                    break
                next_task = random.choice(cur_all_tasks)
                sim_soc = cur_init_soc
                sim_weight = empty_weight
                scale_coff = 1.3
                while len(curpath) < max_tasks_num:
                    if len(curpath) == 0:
                        dist = dist_matrix.get(depot, {}).get(task_to_custom(next_task, data), 0)
                        time = time_matrix.get(depot, {}).get(task_to_custom(next_task, data), 0)
                        load = self.data['tasks'].get(next_task, {}).get('demand', 0)
                        cur_weight = sim_weight + load
                        cur_cost_soc = dist * (base_consump_kwh_per_km + weight_consump_kwh_per_km * cur_weight)
                        next_dist = max(dist_matrix.get(task_to_custom(next_task, data), {}).get(depot, 0),
                                        nearestStation[next_task][1])
                        next_cost_soc = next_dist * (base_consump_kwh_per_km + weight_consump_kwh_per_km * cur_weight)
                        if sim_soc < min_soc + cur_cost_soc + next_cost_soc:
                            break
                        curpath.append((depot, 0, 0, 0))
                        curpath.append((next_task, dist, time, load))
                        cur_all_tasks.remove(next_task)
                        sim_weight += load
                        sim_soc -= cur_cost_soc
                    else:
                        if next_task in self.data['stations']:
                            pre_task = curpath[-1][0]
                            dist = dist_matrix.get(task_to_custom(pre_task, self.data), {}).get(
                                task_to_custom(next_task, self.data), 0)
                            time = time_matrix.get(task_to_custom(pre_task, self.data), {}).get(
                                task_to_custom(next_task, self.data), 0)
                            curpath.append((next_task, dist, time, 0))
                            sim_soc = full_soc
                        else:
                            pre_task = curpath[-1][0]
                            dist = dist_matrix.get(task_to_custom(pre_task, self.data), {}).get(
                                task_to_custom(next_task, self.data), 0)
                            time = time_matrix.get(task_to_custom(pre_task, self.data), {}).get(
                                task_to_custom(next_task, self.data), 0)
                            load = self.data['tasks'].get(next_task, {}).get('demand', 0)
                            cur_weight = sim_weight + load
                            cur_cost_soc = dist * (
                                        base_consump_kwh_per_km + weight_consump_kwh_per_km * cur_weight) * scale_coff
                            next_dist = max(dist_matrix.get(next_task, {}).get(depot, 0), nearestStation[next_task][1])
                            next_cost_soc = next_dist * (
                                        base_consump_kwh_per_km + weight_consump_kwh_per_km * cur_weight) * scale_coff
                            if sim_soc < min_soc + cur_cost_soc + next_cost_soc:
                                # 从上个节点直接返回仓库
                                next_dist = dist_matrix.get(pre_task, {}).get(depot, 0)
                                next_time = time_matrix.get(pre_task, {}).get(depot, 0)
                                curpath.append((depot, next_dist, next_time, 0))
                                break
                            curpath.append((next_task, dist, time, load))
                            cur_all_tasks.remove(next_task)
                            sim_weight += load
                            sim_soc -= cur_cost_soc
                    if len(curpath) >= max_tasks_num:  # 到达最大任务点，回程结束
                        next_depot_dist = dist_matrix.get(task_to_custom(next_task, self.data), {}).get(depot, 0)
                        next_cost_soc = next_depot_dist * (
                                    base_consump_kwh_per_km + weight_consump_kwh_per_km * sim_weight) * scale_coff
                        if sim_soc - next_cost_soc >= min_soc:
                            curpath.append((depot, next_depot_dist,
                                            time_matrix.get(task_to_custom(next_task, data), {}).get(depot, 0), 0))
                            break
                        else:  # 先去一趟换电站，在返回仓库
                            next_station, next_station_dist, next_station_time = nearestStation[next_task]
                            curpath.append((next_station, next_station_dist, next_station_time, 0))
                            next_depot_dist = dist_matrix.get(next_station, {}).get(depot, 0)
                            next_depot_time = time_matrix.get(next_station, {}).get(depot, 0)
                            curpath.append((depot, next_depot_dist, next_depot_time, 0))
                            break
                    # 当前车辆在next_task节点，目前要决策是否下一节点：(最近的任务点，回家，去换电)
                    # 如果 当前电量满足 前往下一任务点+去最近的电站；就去下一任务点，否则去换电
                    # 随机选择下一任务点
                    if len(cur_all_tasks) == 0:
                        if len(curpath) > 0:  # 把返回仓库添加到路径中
                            pre_task = curpath[-1][0]
                            dist = dist_matrix.get(task_to_custom(pre_task, self.data), {}).get(depot, 0)
                            time = time_matrix.get(task_to_custom(pre_task, self.data), {}).get(depot, 0)
                            curpath.append((depot, dist, time, 0))
                        break
                    next_task_ = random.choice(cur_all_tasks)
                    next_task_dist = dist_matrix.get(task_to_custom(next_task, self.data), {}).get(
                        task_to_custom(next_task_, self.data), 0)
                    next_task_station_dist = nearestStation[next_task_][1]
                    next_task_soc_cost = next_task_dist * (
                                base_consump_kwh_per_km + weight_consump_kwh_per_km * cur_weight) * scale_coff
                    next_task_station_soc_cost = next_task_station_dist * (
                                base_consump_kwh_per_km + weight_consump_kwh_per_km * cur_weight) * 1.1
                    if sim_soc < min_soc + next_task_soc_cost + next_task_station_soc_cost and next_task not in \
                            self.data['stations']:  # 去station,然后前往下个任务点
                        next_task, dist, time = nearestStation[next_task]
                        # curpath.append((next_task, dist, time,0))
                        # sim_soc = full_soc
                    else:
                        next_task = next_task_
                paths[vehicle_id] = curpath
            if len(cur_all_tasks) != 0:
                unSatTasks += cur_all_tasks
        # 根据path，更新每条路线的时刻表&载重表&soc表
        print('vrp schedule over')
        v_df, vmt_df, vt_df, vv_df, wt_df, soct_df, vx_df, df_df = [], [], [], [], [], [], [], []
        for vehicle_id in paths:
            curPath = paths[vehicle_id]
            if len(curPath) > 0:
                v_df.append((vehicle_id, full_soc, empty_weight, 1))
            else:
                v_df.append((vehicle_id, full_soc, empty_weight, 0))
            # 记录重量&soc消耗&时间
            sim_time, sim_soc, sim_weight = 0, full_soc, empty_weight
            curWeights, record_station_seq = [], {}
            # 逆序计算weight
            for next_point, dist, time, load in curPath[::-1]:
                sim_weight += load
                curWeights.insert(0, sim_weight)
            for next_point, dist, time, load in curPath:
                vv_df.append((vehicle_id, next_point, 1))
                sim_time += time
                if next_point in self.data['stations']:
                    record_station_seq[next_point] = record_station_seq.get(next_point, 0) + 1
                    vt_df.append((vehicle_id, next_point + '_' + str(record_station_seq[next_point]), sim_time))
                else:
                    vt_df.append((vehicle_id, next_point, sim_time))
            vmt_df.append((vehicle_id, sim_time))
            # 将weight逆序
            record_station_seq = {}
            pre_point = None
            for i, (next_point, dist, time, load) in enumerate(curPath):
                wt_df.append((vehicle_id, next_point, curWeights[i]))
                sim_soc -= dist * (base_consump_kwh_per_km + weight_consump_kwh_per_km * curWeights[i])
                if next_point in self.data['stations']:
                    record_station_seq[next_point] = record_station_seq.get(next_point, 0) + 1
                    soct_df.append(
                        (vehicle_id, next_point + '_' + str(record_station_seq[next_point]) + '_in', sim_soc))
                    soct_df.append(
                        (vehicle_id, next_point + '_' + str(record_station_seq[next_point]) + '_out', full_soc))
                    sim_soc = full_soc
                else:
                    soct_df.append((vehicle_id, next_point, sim_soc))
                if pre_point is not None:
                    vx_df.append((vehicle_id, pre_point, next_point, 1))
                pre_point = next_point
        # 输出未满足的订单
        for task in task_depot:
            demand = self.data['tasks'][task_id]['demand']
            due_time = self.data['tasks'][task_id]['due_time']
            if task in unSatTasks:
                df_df.append((task, due_time, demand, 0))
            else:
                df_df.append((task, due_time, demand, 1))
        df_df = pd.DataFrame(df_df, columns=['task_id', 'due_time', 'demand', 'is_assigned'])
        v_df = pd.DataFrame(v_df, columns=['vehicle_id', 'soc', 'weight', 'is_assigned'])
        vmt_df = pd.DataFrame(vmt_df, columns=['vehicle_id', 'vehicle_max_time'])
        vt_df = pd.DataFrame(vt_df, columns=['vehicle_id', 'task_id', 'arrive_time'])
        vv_df = pd.DataFrame(vv_df, columns=['vehicle_id', 'task_id', 'is_assigned'])
        wt_df = pd.DataFrame(wt_df, columns=['vehicle_id', 'task_id', 'weight'])
        soct_df = pd.DataFrame(soct_df, columns=['vehicle_id', 'task_id', 'soc'])
        vx_df = pd.DataFrame(vx_df, columns=['vehicle_id', 'task_id', 'node', 'is_connected'])
        # 统计换电信息
        station_vt_df = vt_df.copy()
        station_vt_df['station_flag'] = station_vt_df['task_id'].apply(lambda x: int('Station' in x))
        station_vt_df = station_vt_df[station_vt_df['station_flag'] == 1]
        station_soc_df = soct_df.copy()
        station_soc_df['station_flag'] = station_soc_df['task_id'].apply(lambda x: int('Station' in x))
        station_soc_df = station_soc_df[station_soc_df['station_flag'] == 1]
        station_soc_dict = dict(
            zip(zip(station_soc_df['vehicle_id'], station_soc_df['task_id']), station_soc_df['soc']))
        station_vt_df['in'] = station_vt_df.apply(
            lambda x: station_soc_dict.get((x['vehicle_id'], x['task_id'] + '_in'), 0), axis=1)
        station_vt_df['out'] = station_vt_df.apply(
            lambda x: station_soc_dict.get((x['vehicle_id'], x['task_id'] + '_out'), 0), axis=1)
        station_vt_df['swap_kwh'] = station_vt_df['out'] - station_vt_df['in']
        station_vt_df['swap_kw'] = station_vt_df['swap_kwh'] / config.TIME_STEP_HOURS
        station_vt_df['time_num'] = station_vt_df['arrive_time'].apply(lambda x: x // config.TIME_STEP_HOURS)
        station_vt_df['is_swap_bess'] = 1
        station_vt_df['station'] = station_vt_df['task_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        swap_v_df = station_vt_df[['vehicle_id', 'station', 'time_num', 'is_swap_bess']]
        swap_v_df.columns = ['vehicle_id', 'station', 'time', 'is_swap_bess']
        swap_vs_df = station_vt_df[['vehicle_id', 'station', 'time_num', 'swap_kwh']]
        swap_vs_df.columns = ['vehicle_id', 'station', 'time', 'swap_bess_kwh']
        # 聚合电站数据，剔除车辆维度
        station_s_df = []
        for (s, t), mod in station_vt_df.groupby(['station', 'time_num']):
            swapKWH = mod['swap_kwh'].sum()
            station_s_df.append((s, t, swapKWH))
        station_s_df = pd.DataFrame(station_s_df, columns=['station', 'time', 'totalSwapKWH'])
        station_s_df['totalSwapKW'] = station_s_df['totalSwapKWH'] / config.TIME_STEP_HOURS
        station_s_dict = dict(zip(zip(station_s_df['station'], station_s_df['time']), station_s_df['totalSwapKW']))
        # 充电站与光伏的冲减
        pv_dict = {}
        time_steps = range(config.TOTAL_TIME_STEPS)
        for station in self.data['pv_generation']:
            for step in time_steps:
                pv_dict[station, step] = self.data['pv_generation'][station][step]
        swap_pv_e_df, swap_pv_g_df, swap_g_e_df = [], [], []  # 不允许有PV浪费的情况
        for station, step in pv_dict:
            pv_val = pv_dict.get((station, step), 0)
            swap_need = station_s_dict.get((station, step), 0)
            pv_g, pv_e, g_e = 0, 0, 0
            if pv_val > 0:
                if swap_need == 0:
                    pv_g = pv_val
                elif swap_need <= pv_val:
                    pv_g = pv_val - swap_need
                    pv_e = swap_need
                else:
                    pv_e = pv_val
                    g_e = swap_need - pv_val
            else:
                if swap_need > 0:
                    g_e = swap_need
            swap_g_e_df.append((station, step, g_e))
            swap_pv_e_df.append((station, step, pv_e))
            swap_pv_g_df.append((station, step, pv_g))
        swap_pv_e_df = pd.DataFrame(swap_pv_e_df, columns=['station', 'time', 'pv_to_bess_kw'])
        swap_pv_g_df = pd.DataFrame(swap_pv_g_df, columns=['station', 'time', 'pv_to_grid_kw'])
        swap_g_e_df = pd.DataFrame(swap_g_e_df, columns=['station', 'time', 'grid_to_bess_kw'])

        # 换电站与电网的交互提取出来
        swap_g_df = swap_pv_g_df.copy()
        swap_g_df = pd.merge(swap_g_df, swap_g_e_df, how='left', on=['station', 'time'])
        swap_g_df['swap_kw'] = swap_g_df.apply(lambda x: (x['grid_to_bess_kw'] - x['pv_to_grid_kw']) / 1000., axis=1)
        # 电网节点处理
        stationName_busID = {stationName: self.data['stations'][stationName]['bus_id'] for stationName in
                             self.data['stations']}
        station_bus = list(stationName_busID.values())
        swap_g_df['bus'] = swap_g_df['station'].map(stationName_busID)
        swap_g_dict = dict(zip(zip(swap_g_df['bus'], swap_g_df['time']), swap_g_df['swap_kw']))
        # 使用模型计算电网的流向，去除smax和电压的限制，保证完全满足所有负荷；
        buses_df = self.grid_input_data['gridcat_buses']
        edged_df = self.grid_input_data['gridcat_ldf']
        edged_df['r_pu'] = edged_df['r_pu'].apply(lambda x: min(x, 0.0035))
        load_df = self.grid_input_data['gridcat_ts_load_long']
        raw_edge_info_df = self.grid_input_data['gridcat_lines']
        gen_node = [7]  # 连接主电网的节点
        bus_node = buses_df[(buses_df['bus'].isin(edged_df['i'])) | (buses_df['bus'].isin(edged_df['j']))][
            'bus'].unique().tolist()
        v_init = self.grid_input_data['gridcat_vm_init_pu'].set_index('bus').to_dict()['vm_pu']
        vn_max_dict = self.grid_input_data['gridcat_vlimits_pu'].set_index('bus').to_dict()['vmax_pu']
        vn_min_dict = self.grid_input_data['gridcat_vlimits_pu'].set_index('bus').to_dict()['vmin_pu']
        r = dict(zip(zip(edged_df['i'], edged_df['j']), edged_df['r_pu']))

        input_edge = edged_df.groupby('j').apply(lambda x: set(list(zip(x['i'], x['j'])))).to_dict()
        output_edge = edged_df.groupby('i').apply(lambda x: set(list(zip(x['i'], x['j'])))).to_dict()
        out_set = edged_df['i'].unique().tolist()
        in_set = edged_df['j'].unique().tolist()
        only_out_node = edged_df[~edged_df['i'].isin(in_set)]['i'].unique().tolist()
        only_in_node = edged_df[~edged_df['j'].isin(out_set)]['j'].unique().tolist()
        Smax = dict(zip(zip(edged_df['i'], edged_df['j']), edged_df['Smax_MVA']))
        load_node = load_df.set_index(['bus_id', 'time']).to_dict()['p_mw']
        model = scip.Model('Grid_model')
        varDict = {}
        M = 1e6

        def getItem(dict_, i, j):
            if (i, j) in dict_:
                return dict_[i, j]
            elif (j, i) in dict_:
                return dict_[j, i]
            else:
                return 1

        for i, j in r:
            for t in time_steps:  # 热稳定性 |P[i,j]| <= Smax[i,j]
                varDict['P', (i, j, t)] = model.addVar(vtype='C', lb=-M, ub=M,
                                                       name='P_%s_%s_%s' % (i, j, t))  # >0 , i-> j
        for i in gen_node:
            for t in time_steps:
                varDict['P', ('grid', i, t)] = model.addVar(vtype='C', lb=-M, ub=M, name='P_grid_%s_%s' % (i, t))
        for i in bus_node:
            for t in time_steps:
                varDict['v', (i, t)] = model.addVar(vtype='C', lb=-M, name='v_%s_%s' % (i, t))
                varDict['S', (i, t)] = model.addVar(vtype='C', lb=-M,
                                                    name='S_%s_%s' % (i, t))  # 每个节点的功率满足量， >0,买电， <0 卖电
        # 有出度和入度的节点功率平衡
        for i in bus_node:
            for t in time_steps:
                if i not in only_in_node and i not in only_out_node:  # 有入边和出边
                    if i in gen_node:
                        model.addCons(varDict['S', (i, t)] - varDict['P', ('grid', i, t)]
                                      == scip.quicksum(varDict['P', (pre, next, t)] for pre, next in input_edge[i])
                                      - scip.quicksum(varDict['P', (pre, next, t)] for pre, next in output_edge[i]))
                    else:
                        model.addCons(varDict['S', (i, t)]
                                      == scip.quicksum(varDict['P', (pre, next, t)] for pre, next in input_edge[i])
                                      - scip.quicksum(varDict['P', (pre, next, t)] for pre, next in output_edge[i]))
                elif i in only_in_node:  # 仅有入度
                    if i in gen_node:
                        model.addCons(
                            scip.quicksum(varDict['P', (pre, next, t)] for pre, next in input_edge[i]) == varDict[
                                'S', (i, t)] - varDict['P', ('grid', i, t)])
                    else:
                        model.addCons(
                            scip.quicksum(varDict['P', (pre, next, t)] for pre, next in input_edge[i]) == varDict[
                                'S', (i, t)])
                else:  # 只有出度,
                    if i in gen_node:
                        model.addCons(
                            scip.quicksum(varDict['P', (pre, next, t)] for pre, next in output_edge[i]) + varDict[
                                'S', (i, t)] - varDict['P', ('grid', i, t)] == 0)
                    else:
                        model.addCons(
                            scip.quicksum(varDict['P', (pre, next, t)] for pre, next in output_edge[i]) + varDict[
                                'S', (i, t)] == 0)
                curLoad = load_node.get((i, t), 0)
                model.addCons(varDict['S', (i, t)] == curLoad + swap_g_dict.get((i, t), 0))
        # 电压压降
        for i in bus_node:
            for t in time_steps:
                if i in gen_node:
                    model.addCons(varDict['v', (i, t)] == v_init[i])
                    continue
                for pre, next in input_edge.get(i, []):
                    model.addCons(varDict['v', (i, t)] == varDict['v', (pre, t)] - 2 * getItem(r, pre, next) * varDict[
                        'P', (pre, next, t)])
        # 目标： 尽量满足其余节点的负荷、
        exprs = []
        for i, t in load_node:
            cur_load = load_node.get((i, t), 0)
            if cur_load > 0:
                exprs.append(load_node[i, t] - varDict['S', (i, t)])
            elif cur_load < 0:
                exprs.append(-varDict['S', (i, t)] + load_node[i, t])
        obj = scip.quicksum(vars for vars in exprs)
        model.setObjective(obj, sense='minimize')
        # model.writeProblem('model.lp')
        model.setRealParam('limits/time', 120)
        model.setRealParam('limits/gap', 0.1)
        n_vars = model.getNVars()
        n_cons = model.getNConss()
        print('变量: ', n_vars, "约束: ", n_cons)
        model.optimize()
        init_sol = {key: model.getVal(var) for key, var in varDict.items()}
        p_i_j_t, s_i_t, v_i_t = [], [], []
        for i, j in r:
            for t in time_steps:
                p_i_j_t.append((i, j, t, init_sol.get(('P', (i, j, t)), 0)))
        for i in gen_node:
            for t in time_steps:
                p_i_j_t.append(('grid', i, t, init_sol.get(('P', ('grid', i, t)))))
        for i in bus_node:
            for t in time_steps:
                s_i_t.append((i, t, init_sol.get(('S', (i, t)), 0), init_sol.get(('v', (i, t)), 0)))
        p_i_j_t_df = pd.DataFrame(p_i_j_t, columns=['i', 'j', 'time', 'P_PW'])
        p_i_j_t_df['P_PW'] = p_i_j_t_df['P_PW'].round(8)
        s_i_t_df = pd.DataFrame(s_i_t, columns=['bus', 'time', 'Sat_PW', 'voltage_KV'])
        s_i_t_df['Sat_PW'] = s_i_t_df['Sat_PW'].round(8)
        ## 判断是否违反约束
        # 1. 车辆SOC小于最低soc
        soct_df['violate'] = soct_df['soc'].apply(lambda x: int(x < min_soc))
        # 2. 电网侧违背功率smax
        p_i_j_t_df['violate'] = p_i_j_t_df.apply(
            lambda x: int((x['P_PW'] < -Smax[x['i'], x['j']])) + int(x['P_PW'] > Smax[x['i'], x['j']]) if x[
                                                                                                              'i'] != 'grid' else 0,
            axis=1)
        # 3. 电压违背上下界
        s_i_t_df['violate'] = s_i_t_df.apply(
            lambda x: int((x['voltage_KV'] < vn_min_dict[x['bus']])) + int(x['voltage_KV'] > vn_max_dict[x['bus']]),
            axis=1)
        output_dict = {}
        output_dict = {'v_df': v_df, 'vmt_df': vmt_df, 'vt_df': vt_df,
                       'vv_df': vv_df, 'wt_df': wt_df, 'soct_df': soct_df,
                       'vx_df': vx_df, 'df_df': df_df, 'swap_v_df': swap_v_df, 'swap_vs_df': swap_vs_df,
                       'swap_pv_e_df': swap_pv_e_df, 'swap_pv_g_df': swap_pv_g_df,
                       'swap_g_e_df': swap_g_e_df, 'swap_s_df': station_s_df,
                       'p_i_j_t': p_i_j_t_df, 's_i_t': s_i_t_df}

        return output_dict


if __name__ == '__main__':
    data = pd.read_pickle('data.pkl')
    raw_grid_data = pd.read_excel('grid_impedance_catalog3.xlsx', sheet_name=None)
    grid_input_data = {}
    for sheetname, df_data in raw_grid_data.items():
        grid_input_data[sheetname] = df_data
    her = Her(data, grid_input_data, config)
    output_dict = her.vrp_route_schedule()
    with pd.ExcelWriter(f'algo_her_res.xlsx') as writer:
        for k in output_dict.keys():
            output_dict[k].to_excel(writer, sheet_name=k, index=False)