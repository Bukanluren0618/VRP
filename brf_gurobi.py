import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import pandas as pd
import random
from collections import defaultdict
import warnings
# import pyscipopt as gp
import pickle
import ast

# ========================
# 字符串列表 → 真正列表
# ========================
def str_to_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return []
warnings.filterwarnings('ignore')



# 数据处理
# data = pd.read_excel('./new_data_bpr.xlsx',sheet_name=None)
data = pd.read_pickle('./raw_data_bpr.pkl')

# data = {}
# data['nodes_df'] = pd.read_csv('nodes_df.csv')
# data['od_df'] =  pd.read_csv('od_df.csv')
# data['arc_df'] = pd.read_csv('arc_df.csv')
# data['path_arc_df'] = pd.read_csv('path_arc_df.csv')
# data['path_df'] =  pd.read_csv('path_df.csv')
# data['path_df']["arc_path"] = data['path_df']["arc_path"].apply(str_to_list)
# data['path_df']["node_path"] = data['path_df']["node_path"].apply(str_to_list)
# with open('new_data_bpr.pkl','rb') as f:
#     data = pickle.load(f)

# with pd.ExcelWriter(f'new_data_bpr.xlsx') as writer:
#     for k in data.keys():
#         data[k].to_excel(writer, sheet_name=k, index=False)

node_df = data['nodes_df']

#随机生成充电站节点
EVCS_NUM = 10

IESS_NUM = 5

EV_RATIO = 0.3  #30%的EV需要充电

EV_ELE_VOL = 50 #充电的EV的充电量为50KWh

EV_WIAT_TIME = 0.05 #充电时长


all_nodes = node_df['node_id'].unique().tolist()

EVCS_nodes = random.sample(all_nodes, EVCS_NUM)

IESS_nodes = random.sample(list(set(all_nodes) - set(EVCS_nodes)), IESS_NUM)

# 随机生成每个充电节点的电价,后续由电网侧影子价格控制
ele_price_dict = {n:round(random.random() * 100,2) for n in EVCS_nodes + IESS_nodes}

# 每个充电节点的容量,用于排队计算
elc_vol_dict = {n:np.ceil(random.random() * 10) for n in EVCS_nodes + IESS_nodes}

#将虚拟的充电节点写入node表中
for ev_node in EVCS_nodes + IESS_nodes:
    row = node_df[node_df['node_id']==ev_node]
    row['node_id'] = 'vir_' + str(ev_node)
    node_df = pd.concat([node_df,row],ignore_index=True)

#提取od对
od_pairs_df = data['od_df']

od_pairs_df['pairs_demand'] = od_pairs_df.apply(lambda x: (x['origin_customer_index'],x['destination_customer_index'],x['demand_veh_h']), axis=1)

od_pairs_dict = od_pairs_df.set_index('od_id').to_dict()['pairs_demand']

#arc info
# 针对EVCS和IESS节点，虚拟一条path，这个path专供EV充电
arc_df = data['arc_df']
arc_to_virArcId = defaultdict()
for ev_node in EVCS_nodes + IESS_nodes:
    row_df = arc_df[arc_df['to_node']==ev_node] 
    for _, row in row_df.iterrows():
        row_copy = row.copy()
        row_copy['to_node'] = 'vir_' + str(ev_node)
        arc_in = 'arc_' + str(row['from_node']) + '_vir_' + str(ev_node)
        row_copy['arc_id'] = arc_in
        row_copy['capacity_veh_h'] = elc_vol_dict[ev_node]
        arc_df.loc[len(arc_df)] = row_copy
        row_copy = row.copy()
        row_copy['from_node'] = 'vir_' + str(ev_node)
        arc_out = 'arc_vir_' + str(ev_node) + '_'  + str(ev_node)
        row_copy['arc_id'] = arc_out
        row_copy['capacity_veh_h'] = elc_vol_dict[ev_node]
        row_copy['t0_h'] = 0
        arc_df.loc[len(arc_df)] = row_copy
        arc_to_virArcId[row['arc_id']] = [arc_in, arc_out]
#计算精度为1e-6，将通行时间转化为分钟，并扩大十倍
arc_df['t0_h'] *= 600
arc_df['infos'] = arc_df.apply(lambda x: (x['t0_h'],x['capacity_veh_h'],x['alpha'],x['beta']),axis=1)
arc_info_dict = arc_df.set_index('arc_id').to_dict()['infos']
arc_st_dict = arc_df.set_index('arc_id').to_dict()['from_node']
arc_ed_dict = arc_df.set_index('arc_id').to_dict()['to_node']

# od->path

ev_arcs_df = arc_df[arc_df['to_node'].isin(IESS_nodes+EVCS_nodes)]
ev_all_arcs = ev_arcs_df['arc_id'].unique().tolist()

path_arc_incidence_df = data['path_arc_df']
ev_related_df = path_arc_incidence_df[path_arc_incidence_df['arc_id'].isin(ev_all_arcs)]
#记录所有受影响的OD&PATH
# ev_influence_ods = ev_related_df.values.tolist()

candidate_paths_df = data['path_df']
candidate_paths_df['is_ev'] = 0

ev_path_df = pd.DataFrame(columns=candidate_paths_df.columns)
visited_op = set()
for row in ev_related_df.itertuples():
    arc_id = row.arc_id
    ev_node = arc_ed_dict[arc_id]
    if (row.od_id, row.path_id) in visited_op: continue
    visited_op.add((row.od_id, row.path_id))
    ev_row = candidate_paths_df[(candidate_paths_df['path_id']==row.path_id) & (candidate_paths_df['od_id']==row.od_id)].iloc[0]
    # 将虚拟节点追加进去
    new_path, new_node_list = [],[]
    for arc_id in ev_row['arc_path']:
        cur_arc_id = []
        if arc_id in arc_to_virArcId:
            new_path += arc_to_virArcId[arc_id]
            cur_arc_id = arc_to_virArcId[arc_id]
        else:
            new_path.append(arc_id)
            cur_arc_id = [arc_id]
        for ca_id in cur_arc_id:
            if len(new_node_list) == 0:
                new_node_list.append(arc_st_dict[ca_id])
                new_node_list.append(arc_ed_dict[ca_id])
            else:
                new_node_list.append(arc_ed_dict[ca_id])
    ev_row['is_ev'] = 1
    ev_row['arc_path'] = new_path
    ev_row['node_path'] = new_node_list
    ev_row['path_id'] = 'ev_' + str(ev_row['path_id'])  
    ev_row['local_path_id'] =  str(ev_row['path_id'])
    candidate_paths_df.loc[len(candidate_paths_df)] = ev_row
    #把虚拟的path加入
    for ca_id in new_path:
        path_arc_row = ev_related_df.head(1).copy()
        path_arc_row['path_id'] =  str(ev_row['path_id'])  
        path_arc_row['od_id'] = row.od_id
        path_arc_row['arc_id'] = ca_id
        path_arc_incidence_df = pd.concat([path_arc_incidence_df, path_arc_row], ignore_index=True)

print('EVCS+IESS node: ', EVCS_nodes + IESS_nodes)
arc_df.drop_duplicates(subset=['arc_id'], keep='first', inplace=True)
node_df.to_csv('node_df.csv', index=False)
arc_df.to_csv('arcs_bpr.csv', index=False)
candidate_paths_df.to_csv('candidate_paths.csv', index=False)
path_arc_incidence_df.to_csv('path_arc_incidence.csv', index=False)

algo_input_dict = {'node_df': node_df, 'arc_df': arc_df, 'candicate_paths_df': candidate_paths_df, 
                   'path_arc_df': path_arc_incidence_df,'IESS_node':IESS_nodes, 'EVCS_node': EVCS_nodes,
                   'elc_price': ele_price_dict,'elc_vol':elc_vol_dict,'od_pairs': od_pairs_df}

print()

#开始建模

def build_Model(algo_input_dict):
    node_df = algo_input_dict['node_df']
    arc_df = algo_input_dict['arc_df']
    candidate_paths_df = algo_input_dict['candicate_paths_df']
    path_arc_df = algo_input_dict['path_arc_df']
    IESS_nodes = algo_input_dict['IESS_node']
    EVCS_nodes = algo_input_dict['EVCS_node']
    elc_price = algo_input_dict['elc_price']
    elc_vol = algo_input_dict['elc_vol']
    od_pairs_df = algo_input_dict['od_pairs']
    #每个od对可行paths
    od_path_dict = candidate_paths_df.groupby('od_id').apply(lambda x: x['path_id'].unique().tolist()).to_dict()
    od_ev_path_dict = candidate_paths_df[candidate_paths_df['is_ev']==1].groupby('od_id').apply(lambda x: x['path_id'].unique().tolist()).to_dict()
    od_demand_dict = od_pairs_df.set_index('od_id').to_dict()['demand_veh_h']
    all_nodes_list = node_df['node_id'].unique().tolist()
    path_arcs_dict = candidate_paths_df.set_index('path_id').to_dict()['arc_path']
    arc_path_dict = path_arc_df.groupby('arc_id').apply(lambda x: list(zip(x['od_id'],x['path_id']))).to_dict()
    
    arc_info_dict = arc_df.set_index('arc_id').to_dict()['infos']
    node_arcIn_dict = arc_df.groupby('to_node').apply(lambda x: x['arc_id'].unique().tolist()).to_dict()
    node_arcOut_dict = arc_df.groupby('from_node').apply(lambda x: x['arc_id'].unique().tolist()).to_dict()
    vars_dict = {}
    model = gp.Model("Convex_TAP_Gurobi")
    for od in od_path_dict:
        for p in od_path_dict[od]:
            vars_dict['xop',(od,p)] = model.addVar(lb=0, name=f"xop_{od}_{p}")
    for arc_id in arc_path_dict:
        vars_dict['ya',arc_id] = model.addVar(lb=0, name=f"ya_{arc_id}")
        vars_dict['ya5',arc_id] = model.addVar(lb=0, name=f"ya5_{arc_id}")
        vars_dict['bpr',arc_id] = model.addVar(lb=0, name=f"bpr_{arc_id}")
    #约束1 每个od对path流量之和等于设定值
    for od in od_path_dict:
        model.addConstr(gp.quicksum(vars_dict['xop',(od,p)] for p in od_path_dict[od]) == od_demand_dict[od])
    # 约束2 每个arc的流量等于所有经过此路径的path之和
    for arc_id in arc_path_dict:
        model.addConstr(vars_dict['ya',arc_id] == gp.quicksum(vars_dict['xop',(od,p)] for od, p in arc_path_dict[arc_id]))
    
    # #约束3 节点流量平衡
    # for node in all_nodes_list:
    #     model.addConstr(gp.quicksum(vars_dict.get(('ya',arc_id),0) for arc_id in node_arcIn_dict[node]) 
    #                     == gp.quicksum(vars_dict.get(('ya',arc_id),0) for arc_id in node_arcOut_dict[node]) )
    
    # 约束4 bpr定义式
    for arc_id in arc_path_dict:
        infos = arc_info_dict[arc_id]
        to_h, c, alpha, beta = infos
        pow_coff = 2 #若想改为5次方，修改为5即可
        # pow_coff = int(beta) #若想使用配置表里的数据，换成此行即可
        # model.addGenConstrPow(vars_dict['ya',arc_id], vars_dict['ya5',arc_id], pow_coff,"gf", "FuncPieces=1000")
        model.addConstr(vars_dict['ya5',arc_id]  == vars_dict['ya',arc_id] * vars_dict['ya',arc_id])
        coff = to_h * alpha / math.pow(c, pow_coff)
        if to_h == 0 or coff <= 1e-5:
            model.addConstr(vars_dict['bpr',arc_id] == 0)
        else:
            model.addConstr( vars_dict['bpr',arc_id] == to_h * (1 + alpha * vars_dict['ya5',arc_id] / math.pow(c, pow_coff)))
    
    #所有od对总充电的流量满足百分比
    total_demand = sum([od_demand_dict[od] for od in od_ev_path_dict])
    model.addConstr(gp.quicksum(gp.quicksum(vars_dict['xop',(od,p)] for p in od_ev_path_dict.get(od,[])) for od in od_demand_dict) == total_demand * EV_RATIO)

    # 电站等待时间  & 充电量
    for ev_node in IESS_nodes + EVCS_nodes:
        #等待时间
        vir_ev_node = 'vir_' + str(ev_node)
        vars_dict['ev_node_wt',vir_ev_node] = model.addVar(lb=0, name=f'ev_node_wt_{vir_ev_node}')
        vars_dict['ev_node_in',vir_ev_node] = model.addVar(lb=0, name=f'ev_node_in_{vir_ev_node}')
        model.addConstr(vars_dict['ev_node_in',vir_ev_node] == gp.quicksum(vars_dict.get(('ya',arc_id),0) for arc_id in node_arcIn_dict[vir_ev_node]))
        model.addConstr(vars_dict['ev_node_wt',vir_ev_node] >= EV_WIAT_TIME * (vars_dict['ev_node_in',vir_ev_node] -  elc_vol_dict[ev_node]))

    # 目标: bpr时间+ev 排队时间 + 买电花销
    obj = gp.LinExpr()
    bpr_coff, ev_wait_coff, cost_coff = 1,1,1
    for arc_id in arc_info_dict:
        obj += vars_dict.get(('bpr',arc_id),0) * bpr_coff
    for ev_node in IESS_nodes + EVCS_nodes:
        vir_ev_node = 'vir_' + str(ev_node)
        obj += vars_dict['ev_node_wt',vir_ev_node] * ev_wait_coff
        obj += vars_dict['ev_node_in',vir_ev_node] * cost_coff * elc_price[ev_node] * EV_ELE_VOL
    
    model.setObjective(obj, GRB.MINIMIZE)
    # model.setObjective(obj, sense='minimize')
    model.setParam("NonConvex", 2)
    model.setParam("OutputFlag", 1)
    # model.writeProblem('D://model.lp')
    model.optimize()
    status = model.status
    if status == GRB.INFEASIBLE:
        print("\n❌ 模型不可行，开始计算冲突约束...")
        model.computeIIS()
        model.write("infeasibility_report.ilp")
    # 保存结果
    xop_i, ev_in, ya_i, bpr_i = {},{},{},{}
    for od in od_path_dict:
        for p in od_path_dict[od]:
            xop_i[od,p] = vars_dict['xop',(od,p)].X
    for ev_node in IESS_nodes + EVCS_nodes:
        vir_ev_node = 'vir_' + str(ev_node)
        ev_in[ev_node] = vars_dict['ev_node_in',vir_ev_node].X
    for arc_id in arc_path_dict:
        ya_i[arc_id] = vars_dict['ya',arc_id].X
        bpr_i[arc_id] = vars_dict['bpr',arc_id].X
    
    res_dict = {'xop':xop_i, 'ev':ev_in, 'ya':ya_i, 'bpr':bpr_i}
    with open('traficNetWorkFlowRes.pkl','wb+') as f:
        pickle.dump(res_dict,f)
    print('traficNetWorkFlow Over')
    return res_dict


def postHandel(algo_input_dict,res_pkl):
    MAX_ITER = 50          # 最大迭代次数
    CONVERGE_THRESH = 0.01 # 收敛阈值：通行时间变化 < 0.01 分钟即稳定
    MAX_MINUTE = 24*60        # 最大时间范围
    res = pd.read_pickle(res_pkl)
    xop = res['xop']
    # ev = res['ev']
    # ya = res['ya']
    # bpr = res['bpr']
    candidate_paths_df = algo_input_dict['candicate_paths_df']
    od_path_dict = candidate_paths_df.groupby('od_id').apply(lambda x: x['path_id'].unique().tolist()).to_dict()
    arc_df = algo_input_dict['arc_df']
    arc_info_dict = arc_df.set_index('arc_id').to_dict()['infos']
    xop_i, ev_in = [],[]
    eps = 1e-3
    for od in od_path_dict:
        for p in od_path_dict[od]:
            x_val = xop.get((od,p),0)
            if x_val > eps:
                xop_i.append((od, p, x_val, 0))
    xop_df = pd.DataFrame(xop_i, columns=['od_id','path_id','qty','startTime'])

    xop_df = pd.merge(xop_df, candidate_paths_df[['od_id','path_id','arc_path']], how='left',on=['od_id','path_id'])
   

    # ===================== 3. 初始化 =====================
    # 路段字典
    arc_info = {aid: {"t0":t0, "c":c, "a":a, "b":b} for aid,(t0,c,a,b) in arc_info_dict.items()}
    arcs = list(arc_info.keys())
    od_data = xop_df[['od_id','path_id','arc_path','qty','startTime']].values.tolist()
    # 初始化：所有时刻所有路段 通行时间=自由流时间（迭代0）
    prev_tt = defaultdict(lambda: defaultdict(float))
    for a in arcs:
        for t in range(MAX_MINUTE):
            prev_tt[a][t] = arc_info[a]["t0"] / 10 #剔除十倍影响

    # ===================== 4. 迭代主循环 =====================
    for it in range(MAX_ITER):
        # print(f"\n===== 迭代 {it+1}/{MAX_ITER} =====")

        # ---------- 步骤A：加载流量：按【上一轮通行时间】计算车辆到达路段的时间 ----------
        arc_flow = defaultdict(lambda: defaultdict(float))  # arc -> t -> flow

        for o, d, path, qty, st in od_data:
            current_time = float(st)  # 车辆出发时间

            for i, arc in enumerate(path):
                # 进入当前路段的时间（已受前面路段拥堵影响）
                enter_t = round(current_time)
                if 0 <= enter_t < MAX_MINUTE:
                    arc_flow[arc][enter_t] += qty
                else:
                    print('超出一天，', enter_t, ' min' )

                # 用【上一轮迭代的通行时间】驶出该路段
                tt = prev_tt[arc][enter_t]
                current_time += tt

        # ---------- 步骤B：用BPR计算新通行时间 ----------
        new_tt = defaultdict(lambda: defaultdict(float))
        max_diff = 0.0

        for a in arcs:
            #正常path
            t0 = arc_info[a]["t0"]
            c = arc_info[a]["c"]
            alpha = arc_info[a]["a"]
            beta = arc_info[a]["b"]

            for t in range(MAX_MINUTE):
                f = arc_flow[a].get(t, 0.0)
                if c <= 0:
                    tt = t0
                elif 'vir' not in str(a) : #正常arc；BPR计算时间
                    ratio = f / c
                    tt = t0 * (1 + alpha * (ratio ** beta))
                else: #充电等待耗时
                    if t0 == 0:
                        tt = 0
                    else:
                        ev_node = int(arc_ed_dict[a][4:])
                        tt = EV_WIAT_TIME * max(f - elc_vol_dict[ev_node],0)

                new_tt[a][t] = tt
                max_diff = max(max_diff, abs(tt - prev_tt[a][t]))
                # if abs(tt - prev_tt[a][t]) > 100:
                #     print()
        

        # ---------- 步骤C：判断收敛 ----------
        # print(f"最大通行时间变化: {max_diff:.4f}")
        prev_tt = new_tt

        if max_diff < CONVERGE_THRESH:
            # print("已收敛")
            break

    

    # ===================== 5. 输出最终结果 =====================
    rows = []
    for a in arcs:
        for t in range(MAX_MINUTE):
            rows.append({
                "arc": a,
                "time": t,
                "run_time": round(prev_tt[a][t], 3),
                't0_h': arc_info_dict[a][0],
                'c': arc_info_dict[a][1],
                'alpha': arc_info_dict[a][2],
                'beta':arc_info_dict[a][-1]
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["arc", "time"]).reset_index(drop=True)
    #处理电站节点&换电量&换电时间
    df['from_node'] = df['arc'].map(arc_st_dict)
    df['to_node'] = df['arc'].map(arc_ed_dict)
    def judge_type(x):
        if 'vir' in str(x['arc']):
            if x['arc'].split('_')[-1] in IESS_nodes:
                return 'IESS'
            else:
                return 'EVCS'
        else:
            return 'Path'
    df['arc_type'] = df.apply(lambda x: judge_type(x), axis=1)
    # 提取充电节点
    elc_df = df[df['arc_type'] != 'Path']
    df = df[df['arc_type'] == 'Path']
    elc_df = elc_df[elc_df['run_time'] > 0]
    elc_df['elc_node'] = elc_df['arc'].apply(lambda x: x.split('_')[-1])
    elc_node_time_wt_dict = elc_df.groupby('elc_node').apply(lambda x: x.groupby('time').apply(lambda y: y['run_time'].sum()).to_dict()).to_dict()
    elc_node_type_dict = elc_df.set_index('elc_node').to_dict()['arc_type']
    elc_list = []
    new_wt_dict = defaultdict(lambda: defaultdict(float))
    for elc_node in elc_node_time_wt_dict:
        times = sorted(list(elc_node_time_wt_dict[elc_node].keys()))
        pre_t = 0
        for i, t in enumerate(times):
            if i == 0:
                new_wt_dict[elc_node][t] = elc_node_time_wt_dict[elc_node][t]
            else:
                rt = t - pre_t
                new_wt_dict[elc_node][t] = max(new_wt_dict[elc_node][pre_t] - rt, 0) + elc_node_time_wt_dict[elc_node][t]
            pre_t = t
    for elc_node in new_wt_dict:
        for t in new_wt_dict[elc_node]:
            run_time = new_wt_dict[elc_node][t]
            run_qty = run_time / EV_WIAT_TIME + elc_vol_dict[int(elc_node)]
            elc_list.append((elc_node, t,elc_node_type_dict[elc_node], elc_vol_dict[int(elc_node)], run_qty, run_time, EV_ELE_VOL * run_qty))
    elc_df = pd.DataFrame(elc_list, columns=['elc_node','time','type','capacity','flow_qty','wait_time','swap_qty'])
    output_dict = {'od_res': xop_df, 'arc_res':df, 'elc_res': elc_df}

    return output_dict

if __name__ == "__main__":
    build_Model(algo_input_dict)
    res_pkl = 'traficNetWorkFlowRes.pkl'
    output_dict = postHandel(algo_input_dict,res_pkl)
    with pd.ExcelWriter(f'algo_bpr_res.xlsx') as writer:
        for k in output_dict.keys():
            output_dict[k].to_excel(writer, sheet_name=k, index=False)