import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import gurobipy as gp
import config as config
from collections import defaultdict
import random
import pickle
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

def sweep_clustering_with_coords(
    task_coords,       # 任务点坐标：二维数组，shape=(n_tasks, 2)，每行是(x, y)
    demand,            # 任务点载重需求：一维数组，shape=(n_tasks,)，对应每个任务点的需求
    depot_coords,      # 仓库/换电站坐标：一维数组，shape=(2,)，如[0, 0]
    vehicle_capacity,  # 单车载重上限
):
    """
    基于真实坐标的Sweep扫描法分组（带载重约束）
    :return: grouped_tasks: dict，{组ID: [任务点索引列表], ...}
    """
    # 1. 基础校验
    n_tasks = len(task_coords)
    if n_tasks == 0:
        return {}
    if len(demand) != n_tasks:
        raise ValueError(f"任务点数量({n_tasks})与载重需求数量({len(demand)})不匹配")
    
    # 2. 计算每个任务点相对于仓库的极角（核心：真实坐标直接计算）
    # 仓库坐标拆分
    depot_x, depot_y = depot_coords
    # 任务点相对于仓库的偏移坐标
    dx = task_coords[:, 0] - depot_x
    dy = task_coords[:, 1] - depot_y
    # 计算极角（弧度，范围：-π ~ π → 转换为0 ~ 2π）
    polar_angles = np.arctan2(dy, dx)  # arctan2(y, x) 正确计算象限
    polar_angles = np.where(polar_angles < 0, polar_angles + 2 * np.pi, polar_angles)  # 转换为0~2π
    
    # 3. 按极角从小到大排序任务点（顺时针扫描）
    sorted_task_indices = np.argsort(polar_angles)  # 排序后的任务点索引
    
    # 4. 扫描分组（带载重约束）
    grouped_tasks = {}
    current_group_id = 0
    current_group_indices = []
    current_group_demand = 0
    
    for task_idx in sorted_task_indices:
        task_d = demand[task_idx]
        
        # 检查：加入当前点是否超载，且分组数未超车辆数
        if (current_group_demand + task_d <= vehicle_capacity):
            current_group_indices.append(int(task_idx))  # 转为int避免numpy类型
            current_group_demand += task_d
        else:
            # 保存当前组，新建组
            if current_group_indices:
                grouped_tasks[current_group_id] = current_group_indices
                current_group_id += 1
            
            # 校验单个任务点是否超载（关键：避免无法分配的任务）
            if task_d > vehicle_capacity:
                raise ValueError(f"任务点{task_idx}需求({task_d})超过单车载重上限({vehicle_capacity})")
            
            # 初始化新组
            current_group_indices = [int(task_idx)]
            current_group_demand = task_d
    
    # 保存最后一个组（确保不遗漏）
    if current_group_indices:
        grouped_tasks[current_group_id] = current_group_indices

    return grouped_tasks

def sweep_kmeans_clustering_with_coords(
    task_coords,       # 任务点坐标：二维数组，shape=(n_tasks, 2)，每行是(x, y)
    demand,            # 任务点载重需求：一维数组，shape=(n_tasks,)，对应每个任务点的需求
    depot_coords,      # 仓库/换电站坐标：一维数组，shape=(2,)，如[0, 0]
    vehicle_capacity,  # 单车载重上限
):
    """
    基于真实坐标的Sweep扫描法分组（带载重约束）
    :return: grouped_tasks: dict，{组ID: [任务点索引列表], ...}
    """
    # 1. 基础校验
    n_tasks = len(task_coords)
    if n_tasks == 0:
        return {}
    if len(demand) != n_tasks:
        raise ValueError(f"任务点数量({n_tasks})与载重需求数量({len(demand)})不匹配")
    
    # 2. 计算每个任务点相对于仓库的极角（核心：真实坐标直接计算）
    # 仓库坐标拆分
    depot_x, depot_y = depot_coords
    # 任务点相对于仓库的偏移坐标
    dx = task_coords[:, 0] - depot_x
    dy = task_coords[:, 1] - depot_y
    # 计算极角（弧度，范围：-π ~ π → 转换为0 ~ 2π）
    polar_angles = np.arctan2(dy, dx)  # arctan2(y, x) 正确计算象限
    polar_angles = np.where(polar_angles < 0, polar_angles + 2 * np.pi, polar_angles)  # 转换为0~2π
    
    # 3. 按极角从小到大排序任务点（顺时针扫描）
    sorted_task_indices = np.argsort(polar_angles)  # 排序后的任务点索引
    
    # 4. 扫描分组（带载重约束）
    grouped_tasks = {}
    current_group_id = 0
    current_group_indices = []
    current_group_demand = 0
    
    for task_idx in sorted_task_indices:
        task_d = demand[task_idx]
        
        # 检查：加入当前点是否超载，且分组数未超车辆数
        if (current_group_demand + task_d <= vehicle_capacity):
            current_group_indices.append(int(task_idx))  # 转为int避免numpy类型
            current_group_demand += task_d
        else:
            # 保存当前组，新建组
            if current_group_indices:
                grouped_tasks[current_group_id] = current_group_indices
                current_group_id += 1
            
            # 校验单个任务点是否超载（关键：避免无法分配的任务）
            if task_d > vehicle_capacity:
                raise ValueError(f"任务点{task_idx}需求({task_d})超过单车载重上限({vehicle_capacity})")
            
            # 初始化新组
            current_group_indices = [int(task_idx)]
            current_group_demand = task_d
    
    # 保存最后一个组（确保不遗漏）
    if current_group_indices:
        grouped_tasks[current_group_id] = current_group_indices
    #K-Means聚类
    kmeans_subgroups = {}
    current_final_id = 0  # 最终组ID计数器
    
    for sweep_group_id, sweep_task_indices in grouped_tasks.items():
        if len(sweep_task_indices) == 0:
            continue
        
        # 提取当前Sweep大区域的任务点坐标
        sweep_coords = task_coords[sweep_task_indices]
        
        # 确定当前大区域的K-Means聚类数（目标：最终总分组数≈n_vehicles）
        n_kmeans = max(1, round(len(sweep_task_indices) / 10))
        # 避免聚类数超过任务点数
        n_kmeans = min(n_kmeans, len(sweep_task_indices))
        # K-Means聚类（基于真实坐标，无需MDS降维）
        kmeans = KMeans(n_clusters=n_kmeans, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(sweep_coords)
        
        # 保存细聚类结果
        for label in range(n_kmeans):
            # 映射回原始任务点索引
            sub_task_indices = [sweep_task_indices[i] for i in range(len(sweep_task_indices)) 
                               if cluster_labels[i] == label]
            kmeans_subgroups[current_final_id] = sub_task_indices
            current_final_id += 1

    return kmeans_subgroups


BUS_SPEED = 40
GRID_SAVE_NODES = 118 #电网保留节点数

def print_station_grid_mapping(data, grid_input_data=None):
    print("\n" + "=" * 80)
    print("【路网节点 -> 电网节点 映射】")
    print("=" * 80)

    traffic_graph = data['traffic_graph']
    locations = data['locations']
    stations_info = data['stations']

    for station_name, station_meta in stations_info.items():
        road_node_id = locations[station_name]['node_id']
        x = traffic_graph.nodes[road_node_id].get('x', None)
        y = traffic_graph.nodes[road_node_id].get('y', None)
        grid_bus_id = station_meta.get('bus_id', None)

        print(
            f"{station_name} -> 路网node_id={road_node_id}, "
            f"coord=({x:.4f}, {y:.4f}), 电网bus_id={grid_bus_id}"
        )

    if grid_input_data is not None and 'gridcat_buses' in grid_input_data:
        print("-" * 80)
        bus_df = grid_input_data['gridcat_buses']
        bus_set = set(bus_df['nidou'].astype(int).tolist())
        for station_name, station_meta in stations_info.items():
            grid_bus_id = station_meta.get('bus_id', None)
            print(f"{station_name} 的 bus {grid_bus_id} 是否在 gridcat_buses 中: {grid_bus_id in bus_set}")

    print("=" * 80 + "\n")

def dataloarder(data_file='data_new.pkl'):
    #此数据，仅路网可用，任务点均重新生成
    data = pd.read_pickle(data_file)
    traffic_graph = data['traffic_graph']
    locations = data['locations']
    #提取station,depot的node
    depots = [node for node, node_info in data['locations'].items() if node_info['type']=='Depot']
    new_location = defaultdict(defaultdict)
    occupy_idx = set()
    for node, node_info in locations.items():
        if node_info['type'] in ['Depot','SwapStation']:
            new_location[node] = node_info
            occupy_idx.add(node_info['node_id'])
    #重新生成任务点
    task_seq = 1
    tasks_dict = defaultdict(defaultdict)
    idx_coor_dict = defaultdict(tuple)
    for idx, corrs in traffic_graph._node.items():
        idx_coor_dict[idx] = (corrs['x'], corrs['y'])
        if idx not in occupy_idx:
            new_location[f'Customer_{task_seq}'] = {'type':'Customer', 'node_id':idx}
            tasks_dict[f'Task_{task_seq}'] = {'delivery_to':f'Customer_{task_seq}', 'demand':round(np.random.uniform(1,5),2),'depot':random.choice(depots) }
            task_seq += 1
    data['locations'] = new_location
    data['tasks'] = tasks_dict
    def dis(corrx, corry):
        return np.sqrt((corrx[0] - corry[0])**2 + (corrx[1] - corry[1])**2) * config.CITY_SCALE_KM
    # 重新生成距离矩阵
    dist_matrix = defaultdict(defaultdict)
    time_matrix = defaultdict(defaultdict)
    for node1 in new_location:
        idx1 = new_location[node1]['node_id']
        for node2 in new_location:
            idx2 = new_location[node2]['node_id']
            dist = dis(idx_coor_dict[idx1], idx_coor_dict[idx2])
            dist_matrix[node1][node2] = dist
            time_matrix[node1][node2] = dist / BUS_SPEED
    data['dist_matrix'] = dist_matrix
    data['time_matrix'] = time_matrix
    vehicles_dict = defaultdict(defaultdict)  #{'initial_soc': 282.0, 'depot_id': 'Depot_1'}
    vehicle_id = 1
    for depot in depots:
        for idx in range(50):
            vehicles_dict[f"HDT_{vehicle_id}"] = {'initial_soc': 282.0, 'depot_id': depot}
            vehicle_id += 1
    data['vehicles']  = vehicles_dict
    with open('data_new_0.pkl','wb') as f:
        pickle.dump(data, f)

def preProcess(data_file='data_new_0.pkl'):
    data = pd.read_pickle(data_file)
    dist_matrix = data['dist_matrix']
    stations = data['stations']
    depots = [node for node, node_info in data['locations'].items() if node_info['type']=='Depot']
    node_station_dict  = defaultdict()
    station_depot_dict = defaultdict()
    station_nodes_dict = defaultdict(list)
    for node in dist_matrix:
        min_dist = 1e5
        for station in stations:
            if node == station: continue
            dist_node = dist_matrix.get(node,{}).get(station,1e4)
            if dist_node <= min_dist:
                node_station_dict[node] = station
                min_dist = dist_node
    #station-depot必须一对一
    stations_tmp = set(stations)
    others = []
    for depot in depots:
        station = node_station_dict[depot]
        if station in stations_tmp:
            stations_tmp.remove(station)
            station_depot_dict[station] = depot
        else:
            others.append(depot)
    for station in stations_tmp:
        depot = others.pop(0)
        node_station_dict[depot] = station
        station_depot_dict[station] = depot

    for node,station in node_station_dict.items():
        station_nodes_dict[station].append(node)
    
    tasks = list(data['tasks'].keys())
    node_task_dict, task_node_dict = defaultdict(),defaultdict()
    for task in tasks:
        node = data['tasks'][task]['delivery_to']
        node_task_dict[node] = task
        task_node_dict[task] = node
        station = node_station_dict[node]
        depot = station_depot_dict[station]
        data['tasks']['depot'] = depot
    #每个节点的物理坐标
    traffic_graph = data['traffic_graph']
    idx_coor_dict = defaultdict(tuple)
    for idx, corrs in traffic_graph._node.items():
        idx_coor_dict[idx] = (corrs['x'], corrs['y'])
    node_idx_dict, idx_node_dict = defaultdict(), defaultdict()
    for node, node_info in data['locations'].items():
        node_idx_dict[node] = node_info['node_id']
        idx_node_dict[node_info['node_id']] = node
    # 按照SWEEP扫描法划分
    global_sweep_group_res, global_sweep_kmeans_group_res = defaultdict(dict),defaultdict(dict)
    for _, station in enumerate(stations):
        nodes = station_nodes_dict[station]
        seq_node_dict = {idx:node for idx, node in enumerate(nodes)}
        task_coords = []
        demand = []
        for node in nodes:
            task_coords.append(list(idx_coor_dict[node_idx_dict[node]]))
            demand.append(data['tasks'].get(node_task_dict.get(node,""), {}).get('demand',0))
        station_corrd = idx_coor_dict[node_idx_dict[node]]
        cur_group_res = sweep_clustering_with_coords(np.array(task_coords), demand, station_corrd, vehicle_capacity=15)
        #将seq置换为task
        for group_id in cur_group_res:
            task_list = [node_task_dict[seq_node_dict[idx]] for idx in cur_group_res[group_id] if seq_node_dict[idx] in node_task_dict]
            cur_group_res[group_id] = task_list
        global_sweep_group_res[station] = cur_group_res

        cur_group_res = sweep_kmeans_clustering_with_coords(np.array(task_coords), demand, station_corrd, vehicle_capacity=20)
        #将seq置换为task
        for group_id in cur_group_res:
            task_list = [node_task_dict[seq_node_dict[idx]] for idx in cur_group_res[group_id] if seq_node_dict[idx] in node_task_dict]
            cur_group_res[group_id] = task_list
        global_sweep_kmeans_group_res[station] = cur_group_res
        
    # 绘制图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    n_groups = len(stations)
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    for idx, station in enumerate(stations):
        cor = colors[idx]
        nodes = station_nodes_dict[station]
        xs, ys = [],[]
        for node in nodes:
            x, y = idx_coor_dict[node_idx_dict[node]]
            xs.append(x)
            ys.append(y)
        ax1.scatter(xs, ys, 
                   color=cor, 
                   label=f"{station}-tasks",
                   alpha=0.7, s=30)
        ax1.scatter(idx_coor_dict[node_idx_dict[station]][0], idx_coor_dict[node_idx_dict[station]][1], 
               color=cor, marker="*", s=200, label=f"{station}")
    #绘制SWEEP+KMEANS聚类图
    n_groups = sum([len(global_sweep_group_res[station]) for station in global_sweep_group_res])
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    seq = 0
    for station in global_sweep_group_res:
        for group_id in global_sweep_group_res[station]:
            task_list = global_sweep_group_res[station][group_id]
            cor = colors[seq]
            seq += 1
            xs, ys = [],[]
            for task in task_list:
                corrd = idx_coor_dict[node_idx_dict[task_node_dict[task]]]
                xs.append(corrd[0])
                ys.append(corrd[1])
            ax2.scatter(xs, ys, 
                   color=cor, 
                   label=f"{station}-{group_id}",
                   alpha=0.7, s=30)
        ax2.scatter(idx_coor_dict[node_idx_dict[station]][0], idx_coor_dict[node_idx_dict[station]][1], 
               color=cor, marker="*", s=200, label=f"{station}")
    
    n_groups = sum([len(global_sweep_kmeans_group_res[station]) for station in global_sweep_kmeans_group_res])
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    seq = 0
    for station in global_sweep_kmeans_group_res:
        for group_id in global_sweep_kmeans_group_res[station]:
            task_list = global_sweep_kmeans_group_res[station][group_id]
            cor = colors[seq]
            seq += 1
            xs, ys = [],[]
            for task in task_list:
                corrd = idx_coor_dict[node_idx_dict[task_node_dict[task]]]
                xs.append(corrd[0])
                ys.append(corrd[1])
            ax3.scatter(xs, ys, 
                   color=cor, 
                   label=f"{station}-{group_id}",
                   alpha=0.7, s=30)
        ax3.scatter(idx_coor_dict[node_idx_dict[station]][0], idx_coor_dict[node_idx_dict[station]][1], 
               color=cor, marker="*", s=200, label=f"{station}")
    
    plt.tight_layout()
    plt.savefig("vrp_clustering_results.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #选取SWEEP+KMEANS聚类
    with open('global_sweep_kmeans_group_res.pkl','wb') as f:
        pickle.dump(global_sweep_kmeans_group_res, f)
    return global_sweep_kmeans_group_res
class ADMM_Solver:
    def __init__(self, data, grid_input_data, config, cluster_res):
        self.data = data
        self.grid_input_data = grid_input_data
        self.config = config
        self.rho = config.ADMM_RHO  # ADMM惩罚参数
        self.max_iter = config.ADMM_MAX_ITER  # 最大迭代次数
        self.eps = config.ADMM_EPS  # 收敛精度
        self.cluster_res = cluster_res #路线聚类后的结果
        
        # 初始化变量
        self.stations = list(data['stations'].keys())
        self.time_steps = range(config.TOTAL_TIME_STEPS)
        self.numtotime = {t: t * config.TIME_STEP_HOURS for t in self.time_steps}
        
        # 初始化拉格朗日乘子
        self.lambda_s = {
            (station, t): 0.0 for station in self.stations for t in self.time_steps
        }
        
        # 初始化耦合变量
        self.s_central = {
            (station, t): 0.0 for station in self.stations for t in self.time_steps
        }
        
        # 保存历史结果
        self.history = {
            'primal_residual': [],
            'dual_residual': [],
            'objective': []
        }
        #规划路网结构
        self.global_vrp_sols, self.output_dict = self.solve_sub_vrp()
        # self.global_vrp_sols = pd.read_pickle('sub_vrp_sols.pkl')
        # self.output_dict = pd.read_excel('./test.xlsx',sheet_name=None)
        self.res_tmp_dict = self.deal_vrp_res_handle() #后续与电网侧交互的信息
    
    def deal_vrp_res_handle(self):
        sub_vrp_sols = self.output_dict
        vt_df = sub_vrp_sols['vt_df']
        soct_df = sub_vrp_sols['soct_df']
        vv_df = sub_vrp_sols['vv_df']
        # 提取所有使用车辆
        vehicle_ids = vt_df[vt_df['arrive_time']>0]['vehicle_id'].unique().tolist()
        vt_df = vt_df[vt_df['vehicle_id'].isin(vehicle_ids)]
        #仅保存访问过的点
        vv_df = vv_df[vv_df['is_assigned']==1]
        all_visited_node = set(list(zip(vv_df['vehicle_id'], vv_df['task_id'])))
        vt_df['is_visited'] = vt_df.apply(lambda x: int((x['vehicle_id'],x['task_id']) in all_visited_node), axis=1)
        vt_df = vt_df[vt_df['is_visited']==1]
        #离散时间换算
        vt_df['time_idx'] = vt_df['arrive_time'].apply(lambda x: int(x // self.config.TIME_STEP_HOURS))
        #提取所有path
        vehicle_path_dict = vt_df.groupby("vehicle_id").apply(lambda x: sorted(list(zip(x['task_id'], x['arrive_time'])), key=lambda z: z[-1]) ).to_dict()
        # 每个车到换电时刻的离散间隔
        vehicle_swap_timeIdx_dict, vehicle_path_size_dict = defaultdict(),defaultdict()
        vehicle_swap_kw_dict,vehicle_swap_station_dict = defaultdict(),defaultdict()
        for vehicle_id, mod in vt_df.groupby("vehicle_id"):
            #路径长度
            cur_size = mod['time_idx'].max() - mod['time_idx'].min()
            vehicle_path_size_dict[vehicle_id] = cur_size
            #换电时刻
            swap_idx = 0
            for _,row in mod.iterrows():
                if 'Station' in row['task_id']:
                    swap_idx = row['time_idx']
            vehicle_swap_timeIdx_dict[vehicle_id] = swap_idx -   mod['time_idx'].min()
            #每个车换电量
            cur_soc_df = soct_df[soct_df['vehicle_id'] == vehicle_id]
            swap_in, swap_out,swap_station = 0, 0,""
            for _, row in cur_soc_df.iterrows():
                if 'in' in row['task_id'] and 'Station' in row['task_id']:
                    swap_in = row['soc']
                    swap_station = "_".join(row['task_id'].split('_')[:2]) 
                if 'out' in row['task_id'] and 'Station' in row['task_id']:
                    swap_out = row['soc']
            if swap_out > swap_in: 
                vehicle_swap_kw_dict[vehicle_id] = (swap_out - swap_in) / self.config.TIME_STEP_HOURS / 1000.
                vehicle_swap_station_dict[vehicle_id] = swap_station
        res_dict = {"vehiclePath":vehicle_path_dict,'vehicleSwapTimeIdx':vehicle_swap_timeIdx_dict, 
                    "vehiclePathSize":vehicle_path_size_dict, "vehicleSwapSoc":vehicle_swap_kw_dict,
                    'vehicleStation':vehicle_swap_station_dict}
        return  res_dict



    def solve_sub_vrp(self):
        #预先优化sub_vrp顺序
        data = self.data
        config = self.config
        M = 1e3
        # 构建VRP模型
        used_vehicle_set = set()
        global_vrp_sols,output_dict = {},{}
        global_v_values, global_vv_values, global_wt_values, global_soct_values, global_vt_values, global_vx_values,global_vmt_values, global_df_values = [],[],[],[],[],[],[],[]
        for station, groups in self.cluster_res.items():
            stations = [station]
            for idx,tasks in groups.items():
                print(station, " group : ",idx)
                ref_task = tasks[0]
                depot_node = self.data['tasks'][ref_task]['depot']
                model = gp.Model("VRP_Subproblem")
                # model.setRealParam('limits/time', 300)  # 子问题求解时间限制
                # model.setRealParam('limits/gap', 10)  
                # model.hideOutput()
                model.setParam("TimeLimit", 300) # 设置求解时间
                model.setParam("MipGap", 1)  #设置求解gap
                model.setParam("OutputFlag", 0)
                varDict = {}
                depot_vehicles, vehicel_depot = defaultdict(list), {}
                for vehicle_id, vehicle_info in data['vehicles'].items():
                    if vehicle_info['depot_id']  == depot_node and vehicle_id not in used_vehicle_set:
                        depot_vehicles[depot_node].append(vehicle_id)
                        vehicel_depot[vehicle_id] = depot_node
                        #每个子组，最多2辆货车
                        if len(vehicel_depot) >= 3:
                            break
                task_depot, depot_tasks = dict(), defaultdict(list)
                for task_id in tasks:
                    task_depot[task_id] = depot_node
                    depot_tasks[depot_node].append(task_id)
                def nodetoLoc(node, task_depot, data): 
                    loc = 'defaultLOC'
                    if node in task_depot:
                        loc = data['tasks'][node]['delivery_to']
                    elif node in data['stations']:
                        loc = node
                    else:
                        loc = node.split('_')[0]  + '_' + node.split('_')[1]
                    return loc
                for vehicle_id in vehicel_depot:
                    depot = vehicel_depot[vehicle_id]
                    varDict['v',vehicle_id] = model.addVar(vtype=gp.GRB.BINARY, name="v_%s" % vehicle_id) #车辆v是否启用
                    varDict['vmt',vehicle_id] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0,name="vmt_%s" % vehicle_id) #v的最大工作时长
                    # 仓库需要虚拟为两个节点，出一次，进一次
                    depots = [depot +'_out', depot +'_in']
                    for task_id in depot_tasks[depot] + depots + stations:
                        varDict['vv',(vehicle_id,task_id)] = model.addVar(vtype=gp.GRB.BINARY, name="vv_%s_%s" % (vehicle_id, task_id)) #v车是否服务任务t
                        varDict['Wt',(vehicle_id,task_id)] = model.addVar(vtype=gp.GRB.CONTINUOUS,lb=0, name="Wt_%s_%s" % (vehicle_id, task_id)) #v过任务点t时的载重量
                        if task_id in stations:
                            varDict['SOCt',(vehicle_id,task_id+'_in')] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0,name="SOCt_%s_%s" % (vehicle_id, task_id+'_in')) #v到达任务点t时的SOC
                            varDict['SOCt',(vehicle_id,task_id+'_out')] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0,name="SOCt_%s_%s" % (vehicle_id, task_id+'_out')) #v到达任务点t时的SOC
                        else:
                            varDict['SOCt',(vehicle_id,task_id)] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0,name="SOCt_%s_%s" % (vehicle_id, task_id)) #v到达任务点t时的SOC
                        varDict['vt',(vehicle_id,task_id)] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0,name="vt_%s_%s" % (vehicle_id, task_id)) #v过任务点t时的到达时间
                        for node in depot_tasks[depot] + [depot +'_in'] + stations:
                            if node == depot +'_in' and task_id == depot + '_out':
                                continue
                            if  task_id == depot + '_in':
                                continue
                            if node != task_id :
                                s_loc = nodetoLoc(task_id, task_depot,data)
                                n_loc = nodetoLoc(node, task_depot,data)
                                if n_loc in data['dist_matrix'][s_loc]:
                                    varDict['vx',(vehicle_id,task_id,node)] = model.addVar(vtype=gp.GRB.BINARY, name="vx_%s_%s_%s" % (vehicle_id, task_id, node))
                for task_id in task_depot:
                    varDict['df',task_id] = model.addVar(vtype=gp.GRB.BINARY, name="df_%s" % task_id) #任务t是否完成
                if len(tasks) > 3:
                    for station in stations:
                        model.addConstr(gp.quicksum(varDict['vv',(vehicle_id, station)] for vehicle_id in vehicel_depot) >= 1)
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
                    for task_id in depot_tasks[depot] + stations :  
                        var_list_entrys, var_list_outputs = [],[]
                        for node in depot_tasks[depot] + depots + stations:
                            if node != task_id :
                                var_list_outputs.append(varDict.get(('vx',(vehicle_id,task_id,node)),0))
                                var_list_entrys.append(varDict.get(('vx',(vehicle_id,node,task_id)),0))
                        if task_id != depot + '_out' :
                            model.addConstr(gp.quicksum(v for v in var_list_entrys) == varDict['vv',(vehicle_id,task_id)])
                        # if task_id in task_depot:
                        model.addConstr(gp.quicksum(v for v in var_list_outputs) == varDict['vv',(vehicle_id,task_id)])

                    # 仓库出度 = 车辆是否被启用
                    var_list_entrys, var_list_outputs = [],[]
                    for node in depot_tasks[depot] :
                        var_list_outputs.append(varDict.get(('vx',(vehicle_id,depot+'_out',node)),0))
                        var_list_entrys.append(varDict.get(('vx',(vehicle_id,node,depot+'_in')),0))
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
                                model.addConstr(varDict['vt',(vehicle_id,task_id)] <= varDict['vt',(vehicle_id,node)] + dur + (1-varDict.get(('vx',(vehicle_id,node,task_id)),0)) * M)
                            model.addConstr(varDict['vt',(vehicle_id,task_id)] <= varDict['vv', (vehicle_id, task_id)] * M)
                        model.addConstr(varDict['vt',(vehicle_id,node)] <= varDict['vmt',vehicle_id])
                        # if node in task_depot:
                        #     model.addConstr(varDict['vt',(vehicle_id,node)] <= data['tasks'][node]['due_time'] * varDict['df',node])
                    # 车辆如果被启用，必须满足最小服务数量&最大服务数量
                    model.addConstr(gp.quicksum(varDict['vv',(vehicle_id,task_id)] for task_id in depot_tasks[depot]) <= config.MAX_TASKS_PER_TRUCK + (1 - varDict['v',vehicle_id]) * M)
                    model.addConstr(gp.quicksum(varDict['vv',(vehicle_id,task_id)] for task_id in depot_tasks[depot]) >= min(config.MIN_TASKS_PER_TRUCK,len(depot_tasks[depot])) - (1 - varDict['v',vehicle_id]) * M)
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
                        # 强制访问一次
                        # model.addConstr(varDict['vv',(vehicle_id, node)] == 1)
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
                
                # 目标函数：原VRP目标 
                obj1 = gp.quicksum(varDict['vmt',vehicle_id] * config.MANPOWER_COST_PER_HOUR for vehicle_id in vehicel_depot) #人力成本
                obj2 = gp.quicksum(gp.quicksum(varDict['vv',(vehicle_id, node)] * config.FIXED_SWAP_COST for node in stations )for vehicle_id in vehicel_depot) #换电成本
                # obj3 = gp.quicksum((1-varDict['df',task_id]) * config.UNASSIGNED_TASK_PENALTY for task_id in task_depot)

                model.setObjective(obj1 + obj2 , sense=gp.GRB.MINIMIZE)
                # model.writeProblem('D://model.lp')
                model.optimize()
                if model.Status == gp.GRB.Status.INFEASIBLE:
                    model.computeIIS()
                    model.write("./model.ilp")
                init_sol = {key:var.X for key, var in varDict.items()}
                cur_used_vehicle = set()
                for vehicle_id in vehicel_depot:
                    if init_sol['v', vehicle_id] > 0.5:
                        used_vehicle_set.add(vehicle_id)
                        cur_used_vehicle.add(vehicle_id)
                global_vrp_sols.update(init_sol)
                vrp_sol = init_sol
                v_values, vv_values, wt_values, soct_values, vt_values, vx_values,vmt_values, df_values = [],[],[],[],[],[],[],[]
                for task_id in task_depot:
                    demand = data['tasks'][task_id]['demand']
                    df_values.append((task_id,demand,vrp_sol[('df',task_id)]))
                for vehicle_id in cur_used_vehicle:
                    depot = vehicel_depot[vehicle_id]
                    init_soc = data['vehicles'][vehicle_id]['initial_soc']
                    init_weight = config.HDT_EMPTY_WEIGHT_TON
                    v_values.append((vehicle_id,init_soc,init_weight,vrp_sol['v',vehicle_id]))
                    vmt_values.append((vehicle_id,vrp_sol[('vmt',vehicle_id)]))
                    depots = [depot +'_out', depot +'_in']
                    for task_id in depot_tasks[depot] + depots + stations:
                        vv_values.append((vehicle_id,task_id,vrp_sol[('vv',(vehicle_id,task_id))]))
                        wt_values.append((vehicle_id,task_id,vrp_sol[('Wt',(vehicle_id,task_id))]))
                        if task_id in stations:
                            soct_values.append((vehicle_id,task_id+'_in',vrp_sol[('SOCt',(vehicle_id,task_id+'_in'))]))
                            soct_values.append((vehicle_id,task_id+'_out',vrp_sol[('SOCt',(vehicle_id,task_id+'_out'))]))
                        else:
                            soct_values.append((vehicle_id,task_id,vrp_sol[('SOCt',(vehicle_id,task_id))]))
                        vt_values.append((vehicle_id,task_id,vrp_sol[('vt',(vehicle_id,task_id))]))
                        for node in depot_tasks[depot] + [depot +'_in'] + stations:
                            if node == depot +'_in' and task_id == depot + '_out':
                                continue
                            if  task_id == depot + '_in':
                                continue
                            if node != task_id :
                                vx_values.append((vehicle_id,task_id,node,vrp_sol.get(('vx',(vehicle_id,task_id,node)),0)))
                global_v_values += v_values
                global_vv_values += vv_values
                global_wt_values += wt_values
                global_soct_values += soct_values
                global_vt_values += vt_values
                global_vx_values += vx_values
                global_vmt_values += vmt_values
                global_df_values += df_values
                   
        with open('sub_vrp_sols.pkl','wb') as f:
            pickle.dump(global_vrp_sols, f)
        v_df = pd.DataFrame(global_v_values,columns=['vehicle_id', 'soc', 'weight', 'is_assigned'])
        vmt_df = pd.DataFrame(global_vmt_values,columns=['vehicle_id', 'vehicle_max_time'])
        vt_df = pd.DataFrame(global_vt_values,columns=['vehicle_id', 'task_id', 'arrive_time'])
        vv_df = pd.DataFrame(global_vv_values,columns=['vehicle_id', 'task_id', 'is_assigned'])
        wt_df = pd.DataFrame(global_wt_values,columns=['vehicle_id', 'task_id', 'weight'])
        soct_df = pd.DataFrame(global_soct_values,columns=['vehicle_id', 'task_id', 'soc'])
        vx_df = pd.DataFrame(global_vx_values,columns=['vehicle_id', 'task_id', 'node', 'is_connected']) 
        df_df = pd.DataFrame(global_df_values,columns=['task_id', 'demand', 'is_assigned'])
        output_dict = {'v_df':v_df, 'vmt_df':vmt_df, 'vt_df':vt_df, 
                'vv_df':vv_df, 'wt_df':wt_df, 'soct_df':soct_df, 
                'vx_df':vx_df, 'df_df':df_df
                }
        with pd.ExcelWriter(f'res_tmp.xlsx') as writer:
            for k in output_dict.keys():
                output_dict[k].to_excel(writer, sheet_name=k, index=False)
        return global_vrp_sols, output_dict
    
    def solve_vrp_subproblem(self, s_central, lambda_s):
        """求解路网VRP子问题"""
        data = self.data
        vehicle_swap_timeIdx_dict = self.res_tmp_dict["vehicleSwapTimeIdx"]
        vehicle_path_size_dict = self.res_tmp_dict['vehiclePathSize']
        vehicle_swap_kw_dict = self.res_tmp_dict['vehicleSwapSoc']
        vehicle_swap_station_dict = self.res_tmp_dict['vehicleStation']
        M = 1e6
        # 构建VRP模型
        model = gp.Model("VRP_Subproblem")
        # model.setRealParam('limits/time', 3600)  # 子问题求解时间限制
        # model.setRealParam('limits/gap',0.01)
        # model.hideOutput()
        model.setParam("TimeLimit", 100) # 设置求解时间
        model.setParam("MipGap", 0.1)  #设置求解gap
        model.setParam("OutputFlag", 0)
        # 变量定义（简化版，保留与原模型一致的核心变量）
        varDict = {}
        grid_price = data['electricity_prices'].to_dict()
        # 车辆相关变量
        for vehicle_id in vehicle_swap_kw_dict:
            for t in self.time_steps:
                if t < vehicle_swap_timeIdx_dict[vehicle_id] or t + vehicle_path_size_dict[vehicle_id] > max(self.time_steps): continue
                varDict['vt', (vehicle_id, t)] = model.addVar(vtype=gp.GRB.BINARY, name="vt_%s_%s"%(vehicle_id,t))
        # 定义station换电量
        for t in self.time_steps:
            for station in self.cluster_res:
                varDict['swap_g_e',(station,t)] = model.addVar(vtype=gp.GRB.CONTINUOUS,lb=0, name="swap_g_e_%s_%s"%(station,t))
        for vehicle_id in vehicle_swap_kw_dict:
            model.addConstr(gp.quicksum(varDict.get(('vt', (vehicle_id, t)),0) for t in self.time_steps) == 1)
        station_vehicles_dict = defaultdict(list)
        for vehicle, station in vehicle_swap_station_dict.items():
            station_vehicles_dict[station].append(vehicle)
        #station换电量定义式
        for t in self.time_steps:
            for station in self.cluster_res:
                var_list = []
                for vehicle_id in station_vehicles_dict[station]:
                    tmp = t - vehicle_swap_timeIdx_dict[vehicle_id]
                    if tmp >= 0:
                        var_list.append((varDict.get(('vt',(vehicle_id,tmp)),0), vehicle_swap_kw_dict[vehicle_id]))
                model.addConstr(gp.quicksum(v * c for v, c in var_list) == varDict['swap_g_e',(station,t)])
        # obj4 换电费用
        obj4 = gp.quicksum(gp.quicksum( (varDict['swap_g_e',(station,t)] )* grid_price[t] for t in self.numtotime) for station in self.cluster_res)
        # ADMM惩罚项,SCIP不接受目标是二次，因此改为强约束
        obj_admm = gp.LinExpr()
        for station in self.cluster_res:
            for t in self.numtotime:
                aux_var, aux_var_sq = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = -np.inf, ub = np.inf),model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, ub = np.inf)
                model.addConstr(aux_var == varDict['swap_g_e',(station, t)] - s_central[(station, t)])
                model.addConstr(aux_var_sq == aux_var * aux_var)
                obj_admm += lambda_s[(station, t)] * varDict['swap_g_e',(station, t)] 
                obj_admm += 0.5 * self.rho * aux_var_sq
        
        model.setObjective( obj4 + obj_admm, sense=gp.GRB.MINIMIZE)
        model.Params.NonConvex = 2
        model.optimize()
        
        # 提取结果
        result = {
            'swap_g_e': {(station, t):varDict['swap_g_e',(station, t)].X
                          for station in self.cluster_res for t in self.numtotime},
            'vrp_adj_sol':{key:var.X for key, var in varDict.items()}
        }
        
        return result

    def solve_grid_subproblem(self, s_central, lambda_s):
        """求解电网潮汐流优化子问题"""
        data = self.data
        config = self.config
        stations = self.stations
        time_steps = self.time_steps
        phases = ['a', 'b', 'c']
        phase_map = {'a': 0, 'b': 1, 'c': 2}
        alpha = [1.0+0.0j, np.exp(1j*2*np.pi/3), np.exp(1j*4*np.pi/3)]
        def get_alpha(phi_idx, psi_idx):
            return alpha[(phi_idx - psi_idx) % 3]

        def get_common_branches(i, j,node_paths):
            return list(set(node_paths[i]) & set(node_paths[j]))
        # 构建电网模型
        model = gp.Model("Grid_Subproblem")
        # model.setRealParam('limits/time', 3000)  # 子问题求解时间限制
        # model.hideOutput()
        model.setParam("TimeLimit", 100) # 设置求解时间
        model.setParam("MipGap", 0.1)  #设置求解gap
        # model.setParam("OutputFlag", 0)
        # 变量定义（简化版）
        varDict = {}
        
        # 电网侧数据
        data_input = {k: v for k, v in self.grid_input_data.items()}
        load_df = data_input['gridcat_ts_load_long']
        bus_df = data_input['gridcat_buses']
        line_infos_df =  data_input['gridcat_ldf']
        nodes = bus_df['nidou'].unique().tolist()[:GRID_SAVE_NODES]
        #提取子网
        line_infos_df = line_infos_df[(line_infos_df['from_bus'].isin(nodes)) |(line_infos_df['to_bus'].isin(nodes)) ]
        nodes = list(set(line_infos_df['from_bus'].unique().tolist() + line_infos_df['to_bus'].unique().tolist()))
        node_idx_dict = {node:idx for idx, node in enumerate(nodes)}
        #按照指定节点控制电网侧规模
        bus_df = bus_df[bus_df['nidou'].isin(nodes)]
        load_df = load_df[load_df['bus_id'].isin(nodes)]
        vmin_dict = bus_df.set_index('nidou').to_dict()['vn_min']
        vmax_dict = bus_df.set_index('nidou').to_dict()['vn_max']
        s_base = 0.816
        n_nodes = len(nodes)
        n_phases = len(phases)
        branches = {}
        for row in line_infos_df.itertuples():
            f, t = row.from_bus, row.to_bus
            r,x = row.r_pu, row.x_pu
            z_info_dict = {}
            for pf in  phase_map:
                for pt in phase_map:
                    if pf == pt:
                        z_info_dict[pf, pt] = (r,x)
                    else:
                        z_info_dict[pf,pt] = (0,0)
            branches[f,t] = z_info_dict
        G = nx.DiGraph()
        for (u, v) in branches.keys():
            G.add_edge(u, v)
        G_pseudo_undir = G.to_undirected()
        node_paths = {}
        for node in nodes:
            if node == 0:
                node_paths[node] = []
            else:
                try:
                    # 伪无向图中找0到node的路径（统一全局网络）
                    path_nodes = nx.shortest_path(G_pseudo_undir, source=0, target=node)
                    # 转换为有向支路（尽量匹配原拓扑方向）
                    path_branches = []
                    for k in range(len(path_nodes)-1):
                        u_p = path_nodes[k]
                        v_p = path_nodes[k+1]
                        if (u_p, v_p) in branches:
                            path_branches.append((u_p, v_p))
                        else:
                            path_branches.append((v_p, u_p))  # 反向补全
                    node_paths[node] = path_branches
                except nx.NetworkXNoPath:
                    node_paths[node] = []
        size = n_nodes * n_phases
        Rb = np.zeros((size, size))
        Xb = np.zeros((size, size))
        for row in range(size):
            i = row // n_phases
            i = nodes[i]
            phi_idx = row % n_phases
            phi = phases[phi_idx]
            for col in range(size):
                j = col // n_phases
                j = nodes[j]
                psi_idx = col % n_phases
                psi = phases[psi_idx]
                common_brs = get_common_branches(i, j,node_paths)
                if not common_brs:
                    continue
                alpha_phi_psi = get_alpha(phi_idx, psi_idx)
                sum_re, sum_im = 0.0, 0.0
                for (h, k) in common_brs:
                    z_key = (h, k) if (h, k) in branches else (k, h)
                    z_R, z_X = branches[z_key][(phi, psi)]
                    z_conj = (z_R - 1j * z_X)
                    term = alpha_phi_psi * z_conj
                    sum_re += term.real
                    sum_im += term.imag
                Rb[row, col] = 2 * sum_re
                Xb[row, col] = -2 * sum_im
        E0 = 1.0
        E = np.ones(n_nodes * n_phases) * E0
        diag_E_bar = np.diag(np.ones(n_nodes * n_phases) * E0)

        load_df['phase_idx'] = load_df['phase'].map(phase_map)
        load_df['bus_id'] = load_df['bus_id'].astype(int)
        load_df['idx'] = load_df.apply(lambda x: node_idx_dict[x['bus_id']] * n_phases + x['phase_idx'], axis=1)
        T = load_df['time'].unique().tolist()
        values = []
        for (time,idx), mod in load_df.groupby(['time','idx']):
            values.append((time,idx, mod['p_mw'].sum(), mod['q_mvar'].sum()))
        load_df = pd.DataFrame(values, columns=['time','idx','p_mw','q_mvar'])
        p_load = {t: np.zeros(n_nodes * n_phases) for t in T}
        q_load = {t: np.zeros(n_nodes * n_phases) for t in T}
        for t, tmod in load_df.groupby('time'):
            p_load[t][tmod['idx']] = -tmod['p_mw'] / s_base
            q_load[t][tmod['idx']] = -tmod['q_mvar'] / s_base
        stationName_busID = {stationName: data['stations'][stationName]['bus_id'] for stationName in data['stations']}
        station_bus = list(stationName_busID.values())
        stations = list(data['stations'].keys())
        M = 1e6
        varDict = {}

        for station in stations:
            for t in time_steps:
                varDict['swap_g_e',(station, t)] = model.addVar(vtype=gp.GRB.CONTINUOUS,lb=0, name=f"swap_g_e_{station}_{t}")       
        # 电网侧变量
        for t in self.numtotime:
            for idx in range(size):
                bus_id = idx // n_phases
                bus_id = nodes[bus_id]
                varDict['v',(t,idx)] = model.addVar(name=f"v_{t}_{idx}", lb=vmin_dict[bus_id], ub=vmax_dict[bus_id], obj=0.0, vtype="C")
                varDict['p',(t,idx)] = model.addVar(name=f"p_{t}_{idx}", lb=-np.inf, ub=np.inf, obj=0.0, vtype="C")
                varDict['q',(t,idx)] = model.addVar(name=f"q_{t}_{idx}", lb=-np.inf, ub=np.inf, obj=0.0, vtype="C")
                if bus_id in station_bus: continue
                if p_load[t][idx] >= 0:
                    model.addConstr(varDict['p',(t,idx)] <= p_load[t][idx])
                    model.addConstr(varDict['p',(t,idx)] >= 0)
                else:
                    model.addConstr(varDict['p',(t,idx)] >= p_load[t][idx])
                    model.addConstr(varDict['p',(t,idx)] <= 0)
                if q_load[t][idx] >= 0:
                    model.addConstr(varDict['q',(t,idx)] <= q_load[t][idx])
                    model.addConstr(varDict['q',(t,idx)] >= 0)
                else:
                    model.addConstr(varDict['q',(t,idx)] >= q_load[t][idx])
                    model.addConstr(varDict['q',(t,idx)] <= 0)

            for idx in range(size):
                rhs = gp.quicksum(Rb[idx, col] * varDict['p',(t,col)] for col in range(size))
                rhs += gp.quicksum(Xb[idx, col] * varDict['q',(t,col)] for col in range(size))
                rhs += diag_E_bar[idx, idx] * E[idx]
                model.addConstr(varDict['v',(t,idx)] == rhs, name=f"voltage_constraint_{t}_{idx}")        
        
        # 充电站买电时刻，必须满足
        station_idxs = []
        for t in self.numtotime:
            for station in stations:
                bus_id = stationName_busID[station]
                bus_id = node_idx_dict[bus_id]
                for ph in phases:
                    idx = bus_id * n_phases + phase_map[ph]
                    station_idxs.append(idx)
                    model.addConstr(varDict['p',(t,idx) ] == p_load[t][idx] - varDict['swap_g_e',(station, t)] / n_phases)
                    model.addConstr(varDict['q',(t,idx) ] == q_load[t][idx])
        
        
        # 目标函数：原电网目标 + ADMM惩罚项
        obj5 = gp.LinExpr()
        for t in self.numtotime:
            for idx in range(size):
                if p_load[t][idx] > 0:
                    obj5 += p_load[t][idx] - varDict['p',(t,idx)] 
                else:
                    obj5 += varDict['p',(t,idx)] - p_load[t][idx]
                if q_load[t][idx] > 0:
                    obj5 += q_load[t][idx] - varDict['q',(t,idx)] 
                else:
                    obj5 += varDict['q',(t,idx)] - q_load[t][idx]
        
        # ADMM惩罚项
        obj_admm = gp.LinExpr()
        for station in stations:
            for t in time_steps:
                aux_var, aux_var_sq = model.addVar(vtype=gp.GRB.CONTINUOUS, lb = -np.inf, ub = np.inf),model.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, ub = np.inf)
                model.addConstr(aux_var == s_central[(station, t)] - varDict['swap_g_e',(station, t)] )
                model.addConstr(aux_var_sq == aux_var * aux_var)
                obj_admm += lambda_s[(station, t)] * aux_var
                obj_admm += 0.5 * self.rho * aux_var_sq
        
        model.setObjective(obj5 + obj_admm, sense=gp.GRB.MINIMIZE)
        model.Params.NonConvex = 2
        model.optimize()
        
        # 提取结果
        result = {
            'swap_g_e': {(station, t): varDict['swap_g_e',(station, t)].X
                          for station in stations for t in self.numtotime},
            'grid_sol':{key:var.X for key, var in varDict.items()}
        }
        
        return result

    def update_lambda(self, vrp_result, grid_result):
        """更新拉格朗日乘子"""
        new_lambda = {}
        for station in self.stations:
            for t in self.time_steps:
                s_vrp = vrp_result['swap_g_e'][(station, t)]
                s_grid = grid_result['swap_g_e'][(station, t)]
                new_lambda[(station, t)] = self.lambda_s[(station, t)] + self.rho * (s_vrp - s_grid)
        return new_lambda


    def calculate_residuals(self, vrp_result, grid_result):
        primal_res = 0.0
        for station in self.stations:
            for t in self.time_steps:
                s_vrp = vrp_result['swap_g_e'][(station, t)]
                s_grid = grid_result['swap_g_e'][(station, t)]
                primal_res += (s_vrp - s_grid)**2
        return np.sqrt(primal_res)

    def solve(self):
        """ADMM主循环"""
        vrp_result, grid_result = {},{}
        for iter in range(self.max_iter):
            print(f"ADMM迭代次数: {iter+1}/{self.max_iter}")
            
            # 1. 求解VRP子问题
            vrp_result = self.solve_vrp_subproblem(self.s_central, self.lambda_s)
            self.s_central = vrp_result['swap_g_e'].copy()
            print('vrp swap_g_e: ',self.s_central)
            # 2. 求解电网子问题
            grid_result = self.solve_grid_subproblem(self.s_central, self.lambda_s)
            
            # 3. 计算残差
            primal_res = self.calculate_residuals(vrp_result, grid_result)
            prv_lambda =  self.lambda_s
            # 4. 更新拉格朗日乘子
            self.lambda_s = self.update_lambda(vrp_result, grid_result)
            self.s_central = grid_result['swap_g_e'].copy()
            print('grid swap_g_e: ',self.s_central)
            print('原始lambda: ', prv_lambda)
            print('更新后lambda: ', self.lambda_s)
            # 保存历史
            self.history['primal_residual'].append(primal_res)
            
            print(f"原始残差: {primal_res:.6f}")
            
            # 检查收敛
            if primal_res < self.eps :
                print(f"ADMM在第{iter+1}次迭代收敛")
                break
        
        # 返回最终结果
        output_dict = {
            'vrp_result': vrp_result,
            'grid_result': grid_result,
            'history': self.history,
            'lambda': self.lambda_s,
            's_central': self.s_central
        }
        with open('admm_solve_res.pkl','wb') as f:
            pickle.dump(output_dict,f)
        return output_dict


    def postHandle(self, vrp_res, grid_res,s_central):
        node_dfs = []
        vrp_sol = vrp_res['vrp_adj_sol']
        grid_sol = grid_res['grid_sol']
        #电网侧结果
        phases = ['a', 'b', 'c']
        phase_map = {'a': 0, 'b': 1, 'c': 2}
        data_input = {k: v for k, v in self.grid_input_data.items()}
        bus_df = data_input['gridcat_buses']
        line_infos_df =  data_input['gridcat_ldf']
        nodes = bus_df['nidou'].unique().tolist()[:GRID_SAVE_NODES]
        #提取子网
        line_infos_df = line_infos_df[(line_infos_df['from_bus'].isin(nodes)) |(line_infos_df['to_bus'].isin(nodes)) ]
        nodes = list(set(line_infos_df['from_bus'].unique().tolist() + line_infos_df['to_bus'].unique().tolist()))
        node_idx_dict = {node:idx for idx, node in enumerate(nodes)}
        n_phases = len(phases)
        for t in self.numtotime:
            node_values = []
            for i in nodes:
                for phi in phases:
                    idx = node_idx_dict[i] * n_phases + phase_map[phi]
                    v_sq = grid_sol['v',(t,idx)]
                    p_sq = grid_sol['p',(t,idx)]
                    q_sq = grid_sol['q',(t,idx)]
                    node_values.append((t, i,phi, np.sqrt(v_sq), p_sq, q_sq))
            node_df = pd.DataFrame(node_values, columns=['time','bus_id','phase','v_pu','p_mw','q_mvar'])
            node_dfs.append(node_df)
        bus_node_vol_dfs = pd.concat(node_dfs,ignore_index=True)

        #路网侧结果
        data = self.data
        stations = self.stations
        config = self.config
        vehicle_path_dict = self.res_tmp_dict["vehiclePath"]
        vehicle_swap_kw_dict = self.res_tmp_dict['vehicleSwapSoc']
        vehicle_swap_timeIdx_dict = self.res_tmp_dict["vehicleSwapTimeIdx"]
        #处理vrp_sol
        vehicle_st_dict,vehicle_stIDX_dict = defaultdict(),defaultdict()
        for h, (vehicle_id, t) in vrp_sol:
            if vrp_sol[h,(vehicle_id,t)] > 0.5:
                vehicle_st_dict[vehicle_id] = t * self.config.TIME_STEP_HOURS
                vehicle_stIDX_dict[vehicle_id] = t
        depot_nodes = set([node_info['depot'] for node_id, node_info in data['tasks'].items()])
        depot_vehicles, vehicel_depot = defaultdict(list), {}
        for vehicle_id, vehicle_info in data['vehicles'].items():
            if vehicle_info['depot_id'] in depot_nodes:
                depot_vehicles[vehicle_info['depot_id']].append(vehicle_id)
                vehicel_depot[vehicle_id] = vehicle_info['depot_id']
        
        task_depot, depot_tasks = dict(), defaultdict(list)
        for task_id, task_info in data['tasks'].items():
            task_depot[task_id] = task_info['depot']
            depot_tasks[task_info['depot']].append(task_id)

        v_values, vv_values, wt_values, soct_values, vt_values, vx_values,vmt_values, df_values = [],[],[],[],[],[],[],[]
        swap_g_e,swap_v= [],[]
        for task_id in task_depot:
            demand = data['tasks'][task_id]['demand']
            df_values.append((task_id,demand,self.global_vrp_sols[('df',task_id)]))
        df_df = pd.DataFrame(df_values,columns=['task_id', 'demand', 'is_assigned'])
        for vehicle_id in vehicel_depot:
            init_soc = data['vehicles'][vehicle_id]['initial_soc']
            init_weight = config.HDT_EMPTY_WEIGHT_TON
            v_values.append((vehicle_id,init_soc,init_weight,self.global_vrp_sols.get(('v',vehicle_id),0)))
            if vehicle_id in vehicle_path_dict:
                vmt_values.append((vehicle_id,vehicle_path_dict[vehicle_id][-1][1] - vehicle_path_dict[vehicle_id][0][1]))
            else:
                vmt_values.append((vehicle_id,0))
            for idx,(task_id,_) in enumerate(vehicle_path_dict.get(vehicle_id,[])):
                vv_values.append((vehicle_id,task_id,self.global_vrp_sols.get(('vv',(vehicle_id,task_id)),0)))
                wt_values.append((vehicle_id,task_id,self.global_vrp_sols.get(('Wt',(vehicle_id,task_id)),0)))
                if task_id in stations:
                    soct_values.append((vehicle_id,task_id+'_in',self.global_vrp_sols.get(('SOCt',(vehicle_id,task_id+'_in')),0)))
                    soct_values.append((vehicle_id,task_id+'_out',self.global_vrp_sols.get(('SOCt',(vehicle_id,task_id+'_out')),0)))
                else:
                    soct_values.append((vehicle_id,task_id,self.global_vrp_sols.get(('SOCt',(vehicle_id,task_id)),0)))
                vt_values.append((vehicle_id,task_id,self.global_vrp_sols[('vt',(vehicle_id,task_id))] + vehicle_st_dict.get(vehicle_id,0)))
                if idx >= 1:
                    vx_values.append((vehicle_id,vehicle_path_dict[vehicle_id][idx-1][0],task_id,1))
        
        for station in stations:
            for t in self.numtotime:
                for vehicle_id in vehicle_path_dict:
                    if t == vehicle_stIDX_dict.get(vehicle_id,0) + vehicle_swap_timeIdx_dict.get(vehicle_id,0):
                        swap_v.append((vehicle_id,station,t, vehicle_swap_kw_dict.get(vehicle_id,0)))
                    else:
                        swap_v.append((vehicle_id,station,t, 0))
                swap_g_e.append((station,t, s_central[station,t]))
            
    
        swap_v_df = pd.DataFrame(swap_v,columns=['vehicle_id', 'station', 'time', 'swap_bess_kwh'])
        swap_g_e_df = pd.DataFrame(swap_g_e,columns=['station', 'time', 'grid_to_bess_kw'])

        v_df = pd.DataFrame(v_values,columns=['vehicle_id', 'soc', 'weight', 'is_assigned'])
        vmt_df = pd.DataFrame(vmt_values,columns=['vehicle_id', 'vehicle_max_time'])
        vt_df = pd.DataFrame(vt_values,columns=['vehicle_id', 'task_id', 'arrive_time'])
        vv_df = pd.DataFrame(vv_values,columns=['vehicle_id', 'task_id', 'is_assigned'])
        wt_df = pd.DataFrame(wt_values,columns=['vehicle_id', 'task_id', 'weight'])
        soct_df = pd.DataFrame(soct_values,columns=['vehicle_id', 'task_id', 'soc'])
        vx_df = pd.DataFrame(vx_values,columns=['vehicle_id', 'task_id', 'node', 'is_connected']) 
        output_dict = {'v_df':v_df, 'vmt_df':vmt_df, 'vt_df':vt_df, 
                  'vv_df':vv_df, 'wt_df':wt_df, 'soct_df':soct_df, 
                   'vx_df':vx_df, 'df_df':df_df,
                   'swap_v_df':swap_v_df, 
                   'swap_g_e_df':swap_g_e_df,  
                    'bus_node_vol_dfs':bus_node_vol_dfs}
        return output_dict


def deal_vrp_res_handle(config,sub_vrp_sol='./res_tmp.xlsx'):
    sub_vrp_sols = pd.read_excel(sub_vrp_sol,sheet_name=None)
    vt_df = sub_vrp_sols['vt_df']
    soct_df = sub_vrp_sols['soct_df']
    vv_df = sub_vrp_sols['vv_df']
    # 计算每个点所在的时刻
    time_steps = range(config.TOTAL_TIME_STEPS)
    numtotime = {t: t * config.TIME_STEP_HOURS for t in time_steps}
    # 提取所有使用车辆
    vehicle_ids = vt_df[vt_df['arrive_time']>0]['vehicle_id'].unique().tolist()
    vt_df = vt_df[vt_df['vehicle_id'].isin(vehicle_ids)]
    #仅保存访问过的点
    vv_df = vv_df[vv_df['is_assigned']==1]
    all_visited_node = set(list(zip(vv_df['vehicle_id'], vv_df['task_id'])))
    vt_df['is_visited'] = vt_df.apply(lambda x: int((x['vehicle_id'],x['task_id']) in all_visited_node), axis=1)
    vt_df = vt_df[vt_df['is_visited']==1]
    #离散时间换算
    vt_df['time_idx'] = vt_df['arrive_time'].apply(lambda x: int(x // config.TIME_STEP_HOURS))
    #提取所有path
    vehicle_path_dict = vt_df.groupby("vehicle_id").apply(lambda x: sorted(list(zip(x['task_id'], x['arrive_time'])), key=lambda z: z[-1]) ).to_dict()
    # 每个车到换电时刻的离散间隔
    vehicle_swap_timeIdx_dict, vehicle_path_size_dict = defaultdict(),defaultdict()
    vehicle_swap_kw_dict = defaultdict()
    for vehicle_id, mod in vt_df.groupby("vehicle_id"):
        #路径长度
        cur_size = mod['time_idx'].max() - mod['time_idx'].min()
        vehicle_path_size_dict[vehicle_id] = cur_size
        #换电时刻
        swap_idx = 0
        for _,row in mod.iterrows():
            if 'Station' in row['task_id']:
                swap_idx = row['time_idx']
        vehicle_swap_timeIdx_dict[vehicle_id] = swap_idx -   mod['time_idx'].min()
        #每个车换电量
        cur_soc_df = soct_df[soct_df['vehicle_id'] == vehicle_id]
        swap_in, swap_out = 0, 0
        for _, row in cur_soc_df.iterrows():
            if 'in' in row['task_id'] and 'Station' in row['task_id']:
                swap_in = row['soc']
            if 'out' in row['task_id'] and 'Station' in row['task_id']:
                swap_out = row['soc']
        if swap_out > swap_in: 
            vehicle_swap_kw_dict[vehicle_id] = (swap_out - swap_in) / config.TIME_STEP_HOURS / 1000.
    res_dict = {"vehiclePath":vehicle_path_dict,'vehicleSwapTimeIdx':vehicle_swap_timeIdx_dict, 
                "vehiclePathSize":vehicle_path_size_dict, "vehicleSwapSoc":vehicle_swap_kw_dict}
    return  res_dict



if __name__ == '__main__':
    dataloarder()
    data = pd.read_pickle('data_new_0.pkl')
    global_sweep_kmeans_group_res = preProcess()
    # global_sweep_kmeans_group_res = pd.read_pickle('global_sweep_kmeans_group_res.pkl')
    raw_grid_data = pd.read_excel('grid_impedance_catalog_3ph118.xlsx',sheet_name=None) #根据数据，修正
    grid_input_data = {}
    for sheetname, df_data in raw_grid_data.items():
        grid_input_data[sheetname] = df_data

    print_station_grid_mapping(data, grid_input_data)
    admm_solver = ADMM_Solver(data,grid_input_data,config,global_sweep_kmeans_group_res)
    result = admm_solver.solve()
    # result = pd.read_pickle('admm_solve_res.pkl')

    vrp_res, grid_res,s_central = result['vrp_result'], result['grid_result'],result['s_central']
    output_dict = admm_solver.postHandle(vrp_res, grid_res,s_central)
    # # 把字典中的数据写入xlsx文件中
    with pd.ExcelWriter(f'test.xlsx') as writer:
        for k in output_dict.keys():
            output_dict[k].to_excel(writer, sheet_name=k, index=False)
    print("over")
