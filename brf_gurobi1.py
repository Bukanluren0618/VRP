# brf_scip.py — Class-Based BPR Traffic Assignment with EV Charging
# =================================================================
# Classes:
#   BrfDataHandler — 数据处理: 加载/EVCS-IESS生成/虚拟节点-弧/EV路径扩展
#   BrfSolver      — 模型求解 + 后处理 (BPR迭代收敛)
# =================================================================
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import warnings
import pyscipopt as scip
import pickle
warnings.filterwarnings('ignore')
random.seed(10)


# ============================================================
# BrfDataHandler
# ============================================================
class BrfDataHandler:
    def __init__(self, pickle_path='./raw_data_bpr.pkl',
                 evcs_num=10, iess_num=5, ev_ratio=0.6,
                ev_ele_vol=50, ev_wait_time=0.1, ev_wait_penalty=1.0,
                 bpr_cost_weight=1.0, electricity_cost_weight=1.0,
                 iess_nodes=None, evcs_nodes=None,
                 ele_price=None, elc_vol=None):
        print("=" * 70 + "\nBrfDataHandler: 加载数据\n" + "=" * 70)

        data = pd.read_pickle(pickle_path)
        self.node_df = data['nodes_df'].copy()
        self.od_df = data['od_df'].copy()
        self.arc_df = data['arc_df'].copy()
        self.path_arc_df = data['path_arc_df'].copy()
        self.path_df = data['path_df'].copy()

        # ---- 导出副本 ----
        new_data = {
            'nodes_df': data['nodes_df'], 'od_df': data['od_df'],
            'arc_df': data['arc_df'], 'path_arc_df': data['path_arc_df'],
            'path_df': data['path_df']
        }
        with pd.ExcelWriter('data_bpr.xlsx') as writer:
            for k in new_data:
                data[k].to_excel(writer, sheet_name=k, index=False)

        # ---- EVCS / IESS (接受外部注入或随机生成) ----
        self.EV_RATIO = ev_ratio
        self.EV_ELE_VOL = ev_ele_vol
        self.EV_WAIT_TIME = ev_wait_time
        self.EV_WAIT_PENALTY = ev_wait_penalty
        self.BPR_COST_WEIGHT = bpr_cost_weight
        self.ELECTRICITY_COST_WEIGHT = electricity_cost_weight

        all_nodes = self.node_df['node_id'].unique().tolist()

        if iess_nodes is not None:
            self.IESS_nodes = list(iess_nodes)
        else:
            self.IESS_NUM = iess_num
            self.IESS_nodes = random.sample(all_nodes, iess_num)

        if evcs_nodes is not None:
            self.EVCS_nodes = list(evcs_nodes)
        else:
            self.EVCS_NUM = evcs_num
            remaining = list(set(all_nodes) - set(self.IESS_nodes))
            self.EVCS_nodes = random.sample(remaining, evcs_num)

        if ele_price is not None:
            self.ele_price = dict(ele_price)
        else:
            self.ele_price = {
                n: round(random.random() * 100, 2)
                for n in self.EVCS_nodes + self.IESS_nodes
            }

        if elc_vol is not None:
            self.elc_vol = dict(elc_vol)
        else:
            self.elc_vol = {
                n: np.ceil(random.random() * 10)
                for n in self.EVCS_nodes + self.IESS_nodes
            }

        print(f"  EVCS: {len(self.EVCS_nodes)}, IESS: {len(self.IESS_nodes)}")
        print(f' ele price: {self.ele_price}' )

        # ---- 虚拟充电节点 ----
        for ev_node in self.EVCS_nodes + self.IESS_nodes:
            row = self.node_df[self.node_df['node_id'] == ev_node].copy()
            row['node_id'] = 'vir_' + str(ev_node)
            self.node_df = pd.concat([self.node_df, row], ignore_index=True)

        # ---- OD 处理 ----
        self.od_df['pairs_demand'] = self.od_df.apply(
            lambda x: (x['origin_customer_index'],
                       x['destination_customer_index'],
                       x['demand_veh_h']),
            axis=1
        )
        self.od_pairs_dict = self.od_df.set_index('od_id')['pairs_demand'].to_dict()

        # ---- 虚拟弧 ----
        arc_to_vir = {}
        for ev_node in self.EVCS_nodes + self.IESS_nodes:
            rows = self.arc_df[self.arc_df['to_node'] == ev_node]
            for _, row in rows.iterrows():
                rc_in = row.copy()
                rc_in['to_node'] = 'vir_' + str(ev_node)
                arc_in = f"arc_{row['from_node']}_vir_{ev_node}"
                rc_in['arc_id'] = arc_in
                rc_in['capacity_veh_h'] = self.elc_vol[ev_node]
                self.arc_df.loc[len(self.arc_df)] = rc_in

                rc_out = row.copy()
                rc_out['from_node'] = 'vir_' + str(ev_node)
                arc_out = f"arc_vir_{ev_node}_{ev_node}"
                rc_out['arc_id'] = arc_out
                rc_out['capacity_veh_h'] = self.elc_vol[ev_node]
                rc_out['t0_h'] = 0
                self.arc_df.loc[len(self.arc_df)] = rc_out

                arc_to_vir[row['arc_id']] = [arc_in, arc_out]

        # t0_h → 分钟
        self.arc_df['t0_h'] *= 60
        self.arc_df['infos'] = self.arc_df.apply(
            lambda x: (x['t0_h'], x['capacity_veh_h'], x['alpha'], x['beta']),
            axis=1
        )
        self.arc_info = self.arc_df.set_index('arc_id')['infos'].to_dict()
        self.arc_st = self.arc_df.set_index('arc_id')['from_node'].to_dict()
        self.arc_ed = self.arc_df.set_index('arc_id')['to_node'].to_dict()

        # ---- EV 相关路径扩展 ----
        ev_arcs = self.arc_df[
            self.arc_df['to_node'].isin(self.IESS_nodes + self.EVCS_nodes)
        ]
        ev_all_arcs = ev_arcs['arc_id'].unique().tolist()
        ev_related = self.path_arc_df[self.path_arc_df['arc_id'].isin(ev_all_arcs)]

        self.path_df['is_ev'] = 0
        visited = set()
        for row in ev_related.itertuples():
            aid = row.arc_id
            ev_node = self.arc_ed[aid]
            if (row.od_id, row.path_id) in visited:
                continue
            visited.add((row.od_id, row.path_id))
            ev_row = self.path_df[
                (self.path_df['path_id'] == row.path_id) &
                (self.path_df['od_id'] == row.od_id)
            ].iloc[0]

            new_path = []
            new_node_list = []
            for a in ev_row['arc_path']:
                if a in arc_to_vir:
                    new_path += arc_to_vir[a]
                    cur_ids = arc_to_vir[a]
                else:
                    new_path.append(a)
                    cur_ids = [a]
                for ca_id in cur_ids:
                    if len(new_node_list) == 0:
                        new_node_list += [self.arc_st[ca_id], self.arc_ed[ca_id]]
                    else:
                        new_node_list.append(self.arc_ed[ca_id])

            ev_row_copy = ev_row.to_dict()
            ev_row_copy['is_ev'] = 1
            ev_row_copy['arc_path'] = new_path
            ev_row_copy['node_path'] = new_node_list
            ev_row_copy['path_id'] = 'ev_' + str(ev_row_copy['path_id'])
            ev_row_copy['local_path_id'] = str(ev_row_copy['path_id'])
            self.path_df.loc[len(self.path_df)] = ev_row_copy

            for ca_id in new_path:
                par = ev_related.head(1).copy()
                par['path_id'] = str(ev_row_copy['path_id'])
                par['od_id'] = row.od_id
                par['arc_id'] = ca_id
                self.path_arc_df = pd.concat(
                    [self.path_arc_df, par], ignore_index=True
                )

        self.arc_df.drop_duplicates(subset=['arc_id'], keep='first', inplace=True)

        print(f"  EVCS+IESS node: {self.EVCS_nodes + self.IESS_nodes}")
        print("  Data ready.")

    def get_input_dict(self):
        return {
            'node_df': self.node_df,
            'arc_df': self.arc_df,
            'candicate_paths_df': self.path_df,
            'path_arc_df': self.path_arc_df,
            'IESS_node': self.IESS_nodes,
            'EVCS_node': self.EVCS_nodes,
            'elc_price': self.ele_price,
            'elc_vol': self.elc_vol,
            'od_pairs': self.od_df,
        }


class BrfSolver:
    def __init__(self, dh: BrfDataHandler):
        self.dh = dh
        self.algo_input = dh.get_input_dict()
        self._build_model()

    def _build_model(self):
        inp = self.algo_input
        node_df = inp['node_df']
        arc_df = inp['arc_df']
        path_df = inp['candicate_paths_df']
        path_arc_df = inp['path_arc_df']
        IESS_nodes = inp['IESS_node']
        EVCS_nodes = inp['EVCS_node']
        elc_price = inp['elc_price']
        elc_vol = inp['elc_vol']
        od_df = inp['od_pairs']

        # OD → paths
        self.od_path_dict = path_df.groupby('od_id')['path_id'].apply(
            lambda x: x.unique().tolist()
        ).to_dict()
        ev_mask = path_df['is_ev'] == 1
        self.od_ev_path_dict = path_df[ev_mask].groupby('od_id')['path_id'].apply(
            lambda x: x.unique().tolist()
        ).to_dict()
        self.od_demand_dict = od_df.set_index('od_id')['demand_veh_h'].to_dict()

        # arc → [(od, path)]
        self.arc_path_dict = path_arc_df.groupby('arc_id').apply(
            lambda x: list(zip(x['od_id'], x['path_id']))
        ).to_dict()

        self.arc_info_dict = arc_df.set_index('arc_id')['infos'].to_dict()
        self.node_arc_in = arc_df.groupby('to_node')['arc_id'].apply(
            lambda x: x.unique().tolist()
        ).to_dict()
        self.node_arc_out = arc_df.groupby('from_node')['arc_id'].apply(
            lambda x: x.unique().tolist()
        ).to_dict()

        model = scip.Model("BPR_TAP")
        var = {}

        # xop: path flow per OD
        for od in self.od_path_dict:
            for p in self.od_path_dict[od]:
                var[('xop', od, p)] = model.addVar(lb=0, name=f"xop_{od}_{p}")

        # ya, ya5, bpr per arc
        for arc_id in self.arc_path_dict:
            var[('ya', arc_id)] = model.addVar(lb=0, name=f"ya_{arc_id}")
            var[('ya5', arc_id)] = model.addVar(lb=0, name=f"ya5_{arc_id}")
            var[('bpr', arc_id)] = model.addVar(lb=0, name=f"bpr_{arc_id}")

        # ---- 约束 ----
        # C1: OD flow sum
        for od in self.od_path_dict:
            model.addCons(
                scip.quicksum(var[('xop', od, p)] for p in self.od_path_dict[od])
                == self.od_demand_dict[od]
            )

        # C2: arc flow = sum of path flows
        for arc_id in self.arc_path_dict:
            model.addCons(
                var[('ya', arc_id)]
                == scip.quicksum(
                    var[('xop', od, p)]
                    for od, p in self.arc_path_dict[arc_id]
                )
            )

        # C3: BPR definition (ya^2)
        for arc_id in self.arc_path_dict:
            t0, c, alpha, beta = self.arc_info_dict[arc_id]
            c = min(c, 300)
            model.addCons(
                var[('ya5', arc_id)]
                == var[('ya', arc_id)] * var[('ya', arc_id)]
            )
            if t0 == 0:
                model.addCons(var[('bpr', arc_id)] == 0)
            else:
                model.addCons(
                    var[('bpr', arc_id)]
                    == t0 * (1 + alpha * var[('ya5', arc_id)] / np.power(c, 2))
                )

        # C4: EV charging ratio
        total_demand = sum(
            self.od_demand_dict[od] for od in self.od_ev_path_dict
        )
        model.addCons(
            scip.quicksum(
                scip.quicksum(
                    var[('xop', od, p)]
                    for p in self.od_ev_path_dict.get(od, [])
                )
                for od in self.od_demand_dict
            ) == total_demand * self.dh.EV_RATIO
        )

        # C5: 电站等待时间
        for ev_node in IESS_nodes + EVCS_nodes:
            vir_n = 'vir_' + str(ev_node)
            var[('ev_wt', vir_n)] = model.addVar(lb=0, name=f'ev_wt_{vir_n}')
            var[('ev_in', vir_n)] = model.addVar(lb=0, name=f'ev_in_{vir_n}')
            model.addCons(
                var[('ev_in', vir_n)]
                == scip.quicksum(
                    var.get(('ya', a), 0) for a in self.node_arc_in[vir_n]
                )
            )
            model.addCons(
                var[('ev_wt', vir_n)]
                >= self.dh.EV_WAIT_TIME * (
                    var[('ev_in', vir_n)] - elc_vol[ev_node]
                )
            )

        # ---- 目标函数 ----
        model.hideOutput()
        obj = scip.Expr()
        for arc_id in self.arc_info_dict:
            obj += var.get(('bpr', arc_id), 0) * self.dh.BPR_COST_WEIGHT
        for ev_node in IESS_nodes + EVCS_nodes:
            vir_n = 'vir_' + str(ev_node)
            obj += var['ev_wt', vir_n] * self.dh.EV_WAIT_PENALTY
            obj += (var['ev_in', vir_n] * self.dh.ELECTRICITY_COST_WEIGHT
                    * elc_price[ev_node] * self.dh.EV_ELE_VOL)

        model.setObjective(obj, 'minimize')
        # model.writeProblem('D://model.lp')

        self.model = model
        self.var = var
        print(f"  Vars: {model.getNVars()}, Constraints: {model.getNConss()}")

    def solve(self):
        self.model.optimize()
        print(f"  Status: {self.model.getStatus()}")

    def extract_results(self):
        """提取模型解并保存"""
        model = self.model
        var = self.var

        xop = {}
        for od in self.od_path_dict:
            for p in self.od_path_dict[od]:
                xop[(od, p)] = model.getVal(var[('xop', od, p)])

        ev_in = {}
        for ev_node in self.dh.IESS_nodes + self.dh.EVCS_nodes:
            vir_n = 'vir_' + str(ev_node)
            ev_in[ev_node] = model.getVal(var[('ev_in', vir_n)])

        ya = {}
        bpr = {}
        for arc_id in self.arc_path_dict:
            ya[arc_id] = model.getVal(var[('ya', arc_id)])
            bpr[arc_id] = model.getVal(var[('bpr', arc_id)])

        res = {'xop': xop, 'ev': ev_in, 'ya': ya, 'bpr': bpr}
        with open('traficNetWorkFlowRes.pkl', 'wb+') as f:
            pickle.dump(res, f)
        print("  Results saved: traficNetWorkFlowRes.pkl")
        return res

    # --------------------------------------------------------
    # 后处理 (BPR 迭代收敛)
    # --------------------------------------------------------
    def post_handle(self, res, max_iter=50, converge_thresh=0.01, max_minute=24*60):
        print("\n" + "=" * 70 + "\nBrfSolver: BPR 迭代后处理\n" + "=" * 70)

        xop = res['xop']
        path_df = self.algo_input['candicate_paths_df']
        arc_df = self.algo_input['arc_df']
        arc_info_dict = arc_df.set_index('arc_id')['infos'].to_dict()
        od_path_dict = self.od_path_dict
        arc_st = self.dh.arc_st
        arc_ed = self.dh.arc_ed
        IESS_nodes = self.dh.IESS_nodes

        # ---- 构建 xop_data ----
        xop_rows = []
        eps = 1e-3
        for od in od_path_dict:
            for p in od_path_dict[od]:
                val = xop.get((od, p), 0)
                if val > eps:
                    if 'ev' in str(p):
                        rp =random.uniform(0,1)
                        if rp < 0.45:
                            xop_rows.append((od, p, val, random.choice(range(6*60, 8*60))))
                        elif rp < 0.7:
                            xop_rows.append((od, p, val, random.choice(range(0, 23*60))))
                        else:
                            xop_rows.append((od, p, val, random.choice(range(18*60, 20*60))))
                    else:
                        xop_rows.append((od, p, val, 0))
                        # rp =random.uniform(0,1)
                        # if rp < 0.45:
                        #     xop_rows.append((od, p, val, random.choice(range(6*60, 10*60))))
                        # else:
                        #     xop_rows.append((od, p, val, random.choice(range(18*60, 20*60))))
        xop_df = pd.DataFrame(
            xop_rows,
            columns=['od_id', 'path_id', 'qty', 'startTime']
        )
        xop_df = pd.merge(
            xop_df,
            path_df[['od_id', 'path_id', 'arc_path']],
            how='left', on=['od_id', 'path_id']
        )

        # ---- 路段信息 ----
        arc_info = {
            aid: {"t0": t0, "c": c, "a": a, "b": b}
            for aid, (t0, c, a, b) in arc_info_dict.items()
        }
        arcs = list(arc_info.keys())
        od_data = xop_df[
            ['od_id', 'path_id', 'arc_path', 'qty', 'startTime']
        ].values.tolist()

        # 初始化: 自由流时间 / 10
        prev_tt = defaultdict(lambda: defaultdict(float))
        for a in arcs:
            for t in range(max_minute):
                prev_tt[a][t] = arc_info[a]["t0"] / 10.0

        # ---- BPR 迭代 ----
        for it in range(max_iter):
            arc_flow = defaultdict(lambda: defaultdict(float))

            for _, _, path, qty, st in od_data:
                current_time = float(st)
                for arc in path:
                    enter_t = round(current_time)
                    if 0 <= enter_t < max_minute:
                        arc_flow[arc][enter_t] += qty
                    else:
                        print('超出一天，', enter_t, ' min')
                    tt = prev_tt[arc][enter_t]
                    current_time += tt

            new_tt = defaultdict(lambda: defaultdict(float))
            max_diff = 0.0

            for a in arcs:
                t0 = arc_info[a]["t0"]
                c = arc_info[a]["c"]
                alpha = arc_info[a]["a"]
                beta = arc_info[a]["b"]

                for t in range(max_minute):
                    f = arc_flow[a].get(t, 0.0)
                    if c <= 0:
                        tt = t0
                    elif 'vir' not in str(a):
                        ratio = f / c
                        tt = t0 * (1 + alpha * (ratio ** beta))
                    else:
                        if t0 == 0:
                            tt = 0
                        else:
                            ev_node = int(arc_ed[a][4:])
                            tt = self.dh.EV_WAIT_TIME * max(
                                f - self.dh.elc_vol[ev_node], 0
                            )
                    new_tt[a][t] = tt
                    max_diff = max(max_diff, abs(tt - prev_tt[a][t]))

            prev_tt = new_tt

            if max_diff < converge_thresh:
                print(f"  Converged at iteration {it + 1}")
                break

        # ---- 输出 arc_res ----
        rows = []
        for a in arcs:
            for t in range(max_minute):
                rows.append({
                    'arc': a,
                    'time': t,
                    'run_time': round(prev_tt[a][t], 3),
                    't0_h': arc_info_dict[a][0],
                    'c': arc_info_dict[a][1],
                    'alpha': arc_info_dict[a][2],
                    'beta': arc_info_dict[a][-1],
                })
        arc_res = pd.DataFrame(rows)
        arc_res = arc_res.sort_values(['arc', 'time']).reset_index(drop=True)
        arc_res['from_node'] = arc_res['arc'].map(arc_st)
        arc_res['to_node'] = arc_res['arc'].map(arc_ed)

        def judge_type(x):
            if 'vir' in str(x['arc']):
                if int(str(x['arc']).split('_')[-1]) in IESS_nodes:
                    return 'IESS'
                return 'EVCS'
            return 'Path'

        arc_res['arc_type'] = arc_res.apply(judge_type, axis=1)

        # ---- 充/换电站处理 ----
        elc_df = arc_res[arc_res['arc_type'] != 'Path'].copy()
        path_res = arc_res[arc_res['arc_type'] == 'Path'].copy()
        elc_df = elc_df[elc_df['run_time'] > 0].copy()
        elc_df['elc_node'] = elc_df['arc'].apply(
            lambda x: str(x).split('_')[-1]
        )

        elc_node_wt = elc_df.groupby('elc_node').apply(
            lambda x: x.groupby('time')['run_time'].sum().to_dict()
        ).to_dict()
        elc_node_type = elc_df.set_index('elc_node')['arc_type'].to_dict()

        new_wt = defaultdict(lambda: defaultdict(float))
        for elc_node in elc_node_wt:
            times = sorted(elc_node_wt[elc_node].keys())
            pre_t = 0
            for i, t in enumerate(times):
                if i == 0:
                    new_wt[elc_node][t] = elc_node_wt[elc_node][t]
                else:
                    rt = t - pre_t
                    new_wt[elc_node][t] = (
                        max(new_wt[elc_node][pre_t] - rt, 0)
                        + elc_node_wt[elc_node][t]
                    )
                pre_t = t

        elc_rows = []
        for elc_node in new_wt:
            for t in new_wt[elc_node]:
                rt_val = new_wt[elc_node][t]
                qty = rt_val / self.dh.EV_WAIT_TIME + self.dh.elc_vol[int(elc_node)]
                elc_rows.append((
                    elc_node, t,
                    elc_node_type[elc_node],
                    self.dh.elc_vol[int(elc_node)],
                    qty, rt_val,
                    self.dh.EV_ELE_VOL * rt_val,
                ))
        elc_res = pd.DataFrame(
            elc_rows,
            columns=[
                'elc_node', 'time', 'type', 'capacity',
                'flow_qty', 'wait_time', 'swap_qty'
            ]
        )

        self.output = {
            'od_res': xop_df,
            'arc_res': path_res,
            'elc_res': elc_res,
        }
        print(f"  arc_res: {len(path_res)} rows, elc_res: {len(elc_res)} rows")
        return self.output

    def export_excel(self, path='algo_bpr_res.xlsx'):
        if not hasattr(self, 'output'):
            print("  WARNING: Run post_handle() first")
            return
        with pd.ExcelWriter(path) as writer:
            for k, df in self.output.items():
                df.to_excel(writer, sheet_name=k, index=False)
        print(f"  Exported: {path}")


if __name__ == '__main__':
    dh = BrfDataHandler()
    solver = BrfSolver(dh)
    solver.solve()
    res = solver.extract_results()
    solver.post_handle(res)
    solver.export_excel('algo_bpr_res.xlsx')
    print("\nDONE!")