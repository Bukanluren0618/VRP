import numpy as np
import pandas as pd
import random
import gurobipy as gp
from gurobipy import GRB
import warnings
from collections import defaultdict
from road_graph import RoadNetwork

warnings.filterwarnings('ignore')
random.seed(10)
np.random.seed(10)


class DataHandler:
    def __init__(self, num_customers=10, depot_node=10,iess_node=[],iess_price={}, iess_cap={},
                 pickle_path='./raw_data_bpr.pkl',
                 arc_res_path='algo_bpr_res.xlsx'):
        self.num_customers = num_customers
        self.depot_node = depot_node
        print("=" * 70 + "\nDataHandler: 加载数据\n" + "=" * 70)

        raw = pd.read_pickle(pickle_path)
        self.task_df = raw['tasks_df']
        self.customer_df = raw['customers_df']
        self.node_df = raw['nodes_df']
        self.arc_df_raw = raw['arc_df']
        self.path_df = raw['path_df'].copy()

        self.customers = sorted(
            self.customer_df['customer_id'].tolist())[:num_customers]
        self.cn = self.customer_df.set_index('customer_id')['node_id'].to_dict()
        print(f"  Customers: {len(self.customers)}")

        # ---- arc_res 时间数据 ----
        arc_res_df = pd.read_excel(arc_res_path, sheet_name='arc_res')
        t0_mask = arc_res_df['time'] == 0
        self.arc_time_t0 = arc_res_df[t0_mask].set_index('arc')['run_time'].to_dict()
        self.arc_time_all = {}
        for t in sorted(arc_res_df['time'].unique()):
            mask_t = arc_res_df['time'] == t
            self.arc_time_all[t] = arc_res_df[mask_t].set_index('arc')['run_time'].to_dict()

  
        self.IESS_NODES = iess_node
        self.ele_price = iess_price
        self.iess_cap = iess_cap

        an = self.arc_df_raw[
            ['arc_id', 'from_node', 'to_node', 'length_km', 't0_h']
        ].values.tolist()
        self.arpy = RoadNetwork()
        self.arpy.build_from_data(an)
        self.arc_lookup = {}
        self.arc_len_lookup = {}
        for arc_id, f, t, l, t0 in an:
            self.arc_lookup[(f, t)] = arc_id
            self.arc_len_lookup[(f, t)] = l

        # ---- 路径数据 + 补全缺失OD对 ----
        self.path_df['cp'] = self.path_df.apply(
            lambda x: (x['origin_customer_id'], x['destination_customer_id']),
            axis=1
        )
        in_cust = self.path_df['origin_customer_id'].isin(self.customers)
        out_cust = self.path_df['destination_customer_id'].isin(self.customers)
        pfc = self.path_df[in_cust & out_cust]
        self.cp_paths = pfc.groupby('cp')['path_id'].apply(list).to_dict()
        self.pnodes = self.path_df.set_index('path_id')['node_path'].to_dict()
        self.plen_raw = self.path_df.set_index('path_id')['path_length_km'].to_dict()
        self.pth_raw = self.path_df.set_index('path_id')['free_flow_time_h'].to_dict()

        pid_c = self.path_df['path_id'].max() + 1
        for c1 in self.customers:
            for c2 in self.customers:
                if c1 == c2:
                    continue
                if (c1, c2) in self.cp_paths:
                    continue
                np_, td = self.arpy.shortest_path(
                    self.cn[c1], self.cn[c2], mode='distance'
                )
                if np_:
                    tt = 0.0
                    for i in range(len(np_) - 1):
                        for _, f, t, l, t0 in an:
                            if f == np_[i] and t == np_[i + 1]:
                                tt += t0
                                break
                    self.pnodes[pid_c] = np_
                    self.pth_raw[pid_c] = tt * 60
                    self.plen_raw[pid_c] = td
                    self.cp_paths.setdefault((c1, c2), []).append(pid_c)
                    pid_c += 1

                np2_, tt2 = self.arpy.shortest_path(
                    self.cn[c1], self.cn[c2], mode='time'
                )
                if np2_ and np2_ != np_:
                    td2 = 0.0
                    for i in range(len(np2_) - 1):
                        for _, f, t, l, _ in an:
                            if f == np2_[i] and t == np2_[i + 1]:
                                td2 += l
                                break
                    self.pnodes[pid_c] = np2_
                    self.pth_raw[pid_c] = tt2
                    self.plen_raw[pid_c] = td2
                    self.cp_paths[(c1, c2)].append(pid_c)
                    pid_c += 1

 
        self.out_nb = defaultdict(list)
        self.in_nb = defaultdict(list)
        for (c1, c2) in self.cp_paths:
            self.out_nb[c1].append(c2)
            self.in_nb[c2].append(c1)

        # ---- IESS 路径检测----
        self.iess_p = set()
        self.piess = {}
        for pid, np_ in self.pnodes.items():
            if np_:
                for n in np_:
                    if n in self.IESS_NODES:
                        self.iess_p.add(pid)
                        self.piess[pid] = n
                        break

        # ---- 虚拟路径生成  ----

        self.is_virtual = {}
        self.virtual_iess = {}
        self.iess_virtual_paths_by_node = defaultdict(list)

        new_cp_paths = defaultdict(list)
        for (c1, c2), plist in self.cp_paths.items():
            for p in plist:
                new_cp_paths[(c1, c2)].append(p)
                if p in self.iess_p:
                    vp_id = 'swap_' + str(p)
                    new_cp_paths[(c1, c2)].append(vp_id)
                    self.is_virtual[vp_id] = True
                    iess_node = self.piess[p]
                    self.virtual_iess[vp_id] = iess_node
                    self.iess_virtual_paths_by_node[iess_node].append(vp_id)
                    self.pnodes[vp_id] = self.pnodes.get(p)
        self.cp_paths = dict(new_cp_paths)

        # ---- 路径 → (c1,c2) ----
        self.path_cp = defaultdict(list)
        for (c1, c2), pl in self.cp_paths.items():
            for p in pl:
                self.path_cp[p].append((c1, c2))

        # ---- 用 arc_res time=0 替换 free_flow_time_h ----
        for pid in self.pnodes:
            self.pth_raw[pid] = self.compute_path_time(pid) / 60.0
        for vp_id in self.is_virtual:
            orig_pid = int(vp_id.replace('swap_', ''))
            self.pth_raw[vp_id] = self.compute_path_time(orig_pid) / 60.0

        # 虚拟路径 = 原路径时间 + 换电等待时间
        T = 60
        SWAP_WAIT_TIME_MIN = 15  # 换电等待时间 (分钟)
        self.cp_t = {}
        self.cp_d = {}
        for (c1, c2), pl in self.cp_paths.items():
            for p in pl:
                if str(p).startswith('swap_'):
                    orig_pid = int(str(p).replace('swap_', ''))
                    base_t = self.cp_t.get(
                        (c1, c2, orig_pid), self.compute_path_time(orig_pid)
                    )
                    self.cp_t[(c1, c2, p)] = base_t + SWAP_WAIT_TIME_MIN
                    self.cp_d[(c1, c2, p)] = self.cp_d.get(
                        (c1, c2, orig_pid), self.compute_path_distance(orig_pid)
                    )
                else:
                    self.cp_t[(c1, c2, p)] = self.pth_raw.get(p, 0) * T
                    self.cp_d[(c1, c2, p)] = self.compute_path_distance(p)

        # ---- Depot 距离 ----
        self.dep_t = {}
        self.dep_d = {}
        for c in self.customers:
            _, d = self.arpy.shortest_path(
                self.depot_node, self.cn[c], mode='distance'
            )
            _, t_h = self.arpy.shortest_path(
                self.depot_node, self.cn[c], mode='time'
            )
            self.dep_d[c] = d
            self.dep_t[c] = t_h * T
        print(f"  Depot node: {self.depot_node}")

        # ---- 客户任务数据 ----
        ts = self.task_df[self.task_df['customer_id'].isin(self.customers)]
        self.dmd = ts.set_index('customer_id')['demand'].to_dict()
        self.svc = {
            k: v * T
            for k, v in ts.set_index('customer_id')['service_time_h'].items()
        }
        self.tw = {}
        for _, r in ts.iterrows():
            self.tw[r['customer_id']] = (
                r['ready_time_h'] * T,
                r['due_time_h'] * T
            )
        self.total_dmd = sum(self.dmd.values())
        self.NV_MAX = max(2, int(np.ceil(self.total_dmd / 30.0)) + 1)
        self.NV_MAX = min(self.NV_MAX, len(self.customers))
        print(f"  Total demand: {self.total_dmd:.1f}t, "
              f"Max vehicles: {self.NV_MAX}")
        print(f"  Virtual IESS paths: {len(self.is_virtual)}")


    def get_path_arcs(self, pid):
        actual_pid = pid
        if str(pid).startswith('swap_'):
            actual_pid = int(str(pid).replace('swap_', ''))
        np_ = self.pnodes.get(actual_pid)
        if np_ is None or len(np_) < 2:
            arcs = self.path_df.loc[
                self.path_df['path_id'] == actual_pid, 'arc_path'
            ].values
            if len(arcs) > 0 and arcs[0]:
                return arcs[0]
            return []
        result = []
        for i in range(len(np_) - 1):
            aid = self.arc_lookup.get((np_[i], np_[i + 1]))
            if aid is not None:
                result.append(aid)
        return result

    def get_arc_time(self, arc_id, time_min):
        tf = int(np.floor(time_min))
        tc = int(np.ceil(time_min))
        if tf not in self.arc_time_all or tc not in self.arc_time_all:
            return self.arc_time_t0.get(arc_id, 999.0)
        v0 = self.arc_time_all[tf].get(arc_id, 999.0)
        v1 = self.arc_time_all[tc].get(arc_id, 999.0)
        if tf == tc:
            return v0
        frac = time_min - tf
        return v0 + (v1 - v0) * frac

    def compute_path_time(self, pid):
        al = self.get_path_arcs(pid)
        if not al:
            return 0.0
        return sum(self.arc_time_t0.get(a, 999.0) for a in al)

    def compute_path_distance(self, pid):
        actual_pid = pid
        if str(pid).startswith('swap_'):
            actual_pid = int(str(pid).replace('swap_', ''))
        np_ = self.pnodes.get(actual_pid)
        if np_ is None or len(np_) < 2:
            return self.plen_raw.get(actual_pid, 0.0)
        d = 0.0
        for i in range(len(np_) - 1):
            d += self.arc_len_lookup.get((np_[i], np_[i + 1]), 0.0)
        return d

    def compute_path_time_at(self, pid, arrive_min):
        al = self.get_path_arcs(pid)
        if not al:
            return 0.0
        t = arrive_min
        total = 0.0
        for a in al:
            rt = self.get_arc_time(a, t)
            total += rt
            t += rt
        return total


class VRPSolver:
    def __init__(self, dh: DataHandler, time_limit=300, gap=0.05,
                 num_veh=None):
        self.dh = dh
        self.time_limit = time_limit
        self.gap = gap
        self.DEPOT = '__D__'
        self.BATT_CAP = 282.0
        self.BATT_MIN = self.BATT_CAP * 0.2
        self.EMPTY = 10.0
        self.M = 1e6
        self.MAX_PL = 40.0
        self.BK = 0.6
        self.WK = 0.02
        self.num_veh = num_veh if num_veh else dh.NV_MAX
        self._build_model()

    def _build_model(self):
        dh = self.dh
        NV = self.num_veh
        D = self.DEPOT
        M = self.M

        model = gp.Model('VRP_VirtPath')
        var = {}

        # ---- 车辆变量 ----
        for v in range(NV):
            var[('use', v)] = model.addVar(vtype=GRB.BINARY)
            var[('vmt', v)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)

        # ---- 节点变量 ----
        for v in range(NV):
            for c in dh.customers + [D]:
                var[('vv', v, c)] = model.addVar(vtype=GRB.BINARY)
                var[('wt', v, c)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                var[('soc', v, c)] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, ub=self.BATT_CAP
                )
                var[('at', v, c)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                var[('st', v, c)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                var[('et', v, c)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                if c != D:
                    var[('cs', v, c)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                    var[('ce', v, c)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
        for c in dh.customers:
            var[('df', c)] = model.addVar(vtype=GRB.BINARY)

        # ---- 弧和路径变量 ----
        for v in range(NV):
            for c in dh.customers:
                var[('vx', v, D, c)] = model.addVar(vtype=GRB.BINARY)
                var[('vx', v, c, D)] = model.addVar(vtype=GRB.BINARY)
            for (c1, c2), pl in dh.cp_paths.items():
                var[('vx', v, c1, c2)] = model.addVar(vtype=GRB.BINARY)
                var[('vdur', v, c1, c2)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                var[('vz', v, c1, c2)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                for p in pl:
                    var[('vp', v, c1, c2, p)] = model.addVar(vtype=GRB.BINARY)

        model.update()
        print(f"  Vars: {model.NumVars}")



        # 需求覆盖
        for c in dh.customers:
            model.addConstr(
                gp.quicksum(var[('vv', v, c)] for v in range(NV))
                == var[('df', c)]
            )
            model.addConstr(var[('df', c)] == 1)

        # 车辆使用
        for v in range(NV):
            model.addConstr(
                gp.quicksum(var[('vv', v, c)] for c in dh.customers)
                <= var[('use', v)] * M
            )

        #  流量守恒
        for v in range(NV):
            d_out = gp.quicksum(var[('vx', v, D, c)] for c in dh.customers)
            d_in = gp.quicksum(var[('vx', v, c, D)] for c in dh.customers)
            model.addConstr(d_out == var[('vv', v, D)])
            model.addConstr(d_in == var[('vv', v, D)])
            model.addConstr(var[('vv', v, D)] <= var[('use', v)])

            for c in dh.customers:
                inf_ = var[('vx', v, D, c)]
                for pre in dh.in_nb.get(c, []):
                    if ('vx', v, pre, c) in var:
                        inf_ += var[('vx', v, pre, c)]
                model.addConstr(inf_ == var[('vv', v, c)])

                outf = var[('vx', v, c, D)]
                for nx in dh.out_nb.get(c, []):
                    if ('vx', v, c, nx) in var:
                        outf += var[('vx', v, c, nx)]
                model.addConstr(outf == var[('vv', v, c)])

        #  路径选择 vx = Σvp (包含虚拟路径)
        for v in range(NV):
            for (c1, c2), pl in dh.cp_paths.items():
                model.addConstr(
                    var[('vx', v, c1, c2)]
                    == gp.quicksum(var[('vp', v, c1, c2, p)] for p in pl)
                )

        #  vdur = Σvp × travel_time
        for v in range(NV):
            for (c1, c2), pl in dh.cp_paths.items():
                model.addConstr(
                    var[('vdur', v, c1, c2)]
                    == gp.quicksum(
                        var[('vp', v, c1, c2, p)]
                        * dh.cp_t.get((c1, c2, p), 0)
                        for p in pl
                    )
                )

        #  vz 线性化
        for v in range(NV):
            for (c1, c2) in dh.cp_paths:
                model.addConstr(
                    var[('vz', v, c1, c2)] <= var[('vx', v, c1, c2)] * M
                )
                model.addConstr(
                    var[('vz', v, c1, c2)] <= var[('vdur', v, c1, c2)]
                )
                model.addConstr(
                    var[('vz', v, c1, c2)]
                    >= var[('vdur', v, c1, c2)]
                    - (1 - var[('vx', v, c1, c2)]) * M
                )

        #  时间约束 (depot→c 使用 dep_t)
        for v in range(NV):
            model.addConstr(var[('at', v, D)] == 0)
            model.addConstr(var[('st', v, D)] == 0)
            model.addConstr(var[('et', v, D)] == 0)

            for c in dh.customers:
                sv = dh.svc[c]
                tw0, tw1 = dh.tw[c]

                model.addConstr(
                    var[('at', v, c)]
                    >= dh.dep_t[c]
                    - (1 - var[('vx', v, D, c)]) * M
                )
                for pre in dh.in_nb.get(c, []):
                    if ('vx', v, pre, c) in var:
                        model.addConstr(
                            var[('at', v, c)]
                            >= var[('et', v, pre)] + var[('vz', v, pre, c)]
                            - (1 - var[('vx', v, pre, c)]) * M
                        )
                model.addConstr(var[('st', v, c)] >= var[('at', v, c)])
                model.addConstr(
                    var[('et', v, c)]
                    >= var[('st', v, c)] + sv * var[('vv', v, c)]
                )
                model.addConstr(var[('at', v, c)] <= var[('vv', v, c)] * M)
                model.addConstr(var[('st', v, c)] <= var[('vv', v, c)] * M)
                model.addConstr(var[('et', v, c)] <= var[('vv', v, c)] * M)
                model.addConstr(var[('et', v, c)] <= var[('vmt', v)])
                model.addConstr(var[('cs', v, c)] >= tw0 - var[('st', v, c)])
                model.addConstr(var[('ce', v, c)] >= var[('et', v, c)] - tw1)

        # 载重约束 (knapsack + 正向传播)
        for v in range(NV):
            model.addConstr(var[('wt', v, D)] == self.EMPTY * var[('use', v)])
            # 每车累积需求 ≤ payload
            model.addConstr(
                gp.quicksum(var[('vv', v, c)] * dh.dmd[c] for c in dh.customers)
                <= self.MAX_PL
            )
            for c in dh.customers:
                dmd = dh.dmd[c]
                # 访问的客户必须承载自身货物
                model.addConstr(
                    var[('wt', v, c)]
                    >= self.EMPTY + dmd
                    - (1 - var[('vv', v, c)]) * M
                )
                # depot → c: 满载出发
                model.addConstr(
                    var[('wt', v, c)]
                    >= var[('wt', v, D)]
                    - (1 - var[('vx', v, D, c)]) * M
                )
                # wt 严格上界: 空车 + 最大载重 (防止SOC消耗公式因wt无界而强制swap)
                model.addConstr(var[('wt', v, c)] <= self.EMPTY + self.MAX_PL)
                # c → nx: nx 的载重 = c 的载重 - c 的货物 (已卸货)
                for nx in dh.out_nb.get(c, []):
                    if ('vx', v, c, nx) in var:
                        model.addConstr(
                            var[('wt', v, nx)]
                            >= var[('wt', v, c)] - dmd
                            - (1 - var[('vx', v, c, nx)]) * M
                        )
                        model.addConstr(
                            var[('wt', v, nx)]
                            <= var[('wt', v, c)] - dmd
                            + (1 - var[('vx', v, c, nx)]) * M
                        )
                model.addConstr(var[('wt', v, c)] <= var[('vv', v, c)] * M)

        # SOC 约束 (使用真实 wt 变量; 虚拟路径 → SOC满电)
        for v in range(NV):
            model.addConstr(
                var[('soc', v, D)] == self.BATT_CAP * var[('use', v)]
            )
            for c in dh.customers:
                # depot → c: SOC = 满电 - depot消耗 (基于空车 wt)
                soc_dep = dh.dep_d[c] * (self.BK + self.WK * var[('wt', v, D)])
                model.addConstr(
                    var[('soc', v, c)]
                    >= self.BATT_CAP - soc_dep
                    - (1 - var[('vx', v, D, c)]) * M
                )
                model.addConstr(
                    var[('soc', v, c)]
                    <= self.BATT_CAP + (1 - var[('vx', v, D, c)]) * M
                )
                for pre in dh.in_nb.get(c, []):
                    if ('vx', v, pre, c) not in var:
                        continue
                    for p in dh.cp_paths.get((pre, c), []):
                        dist = dh.cp_d.get((pre, c, p), 0)
                        # SOC消耗 = 距离 × (基础 + 重量系数 × 当前载重)
                        soc_use = dist * (self.BK + self.WK * var[('wt', v, pre)])
                        if str(p).startswith('swap_'):
                            model.addConstr(
                                var[('soc', v, c)]
                                >= self.BATT_CAP
                                - (1 - var[('vp', v, pre, c, p)]) * M
                            )
                            model.addConstr(
                                var[('soc', v, c)]
                                <= self.BATT_CAP
                                + (1 - var[('vp', v, pre, c, p)]) * M
                            )
                        else:
                            model.addConstr(
                                var[('soc', v, c)]
                                >= var[('soc', v, pre)] - soc_use
                                - (1 - var[('vp', v, pre, c, p)]) * M
                            )
                            model.addConstr(
                                var[('soc', v, c)]
                                <= var[('soc', v, pre)] - soc_use
                                + (1 - var[('vp', v, pre, c, p)]) * M
                            )
                model.addConstr(
                    var[('soc', v, c)] >= self.BATT_MIN * var[('vv', v, c)]
                )
                model.addConstr(
                    var[('soc', v, c)] <= self.BATT_CAP * var[('vv', v, c)]
                )

        model.update()
        print(f"  Constraints: {model.NumConstrs}")

        obj = gp.LinExpr()

        # 人力成本
        for v in range(NV):
            obj += var[('vmt', v)] * 80

        # 虚拟路径换电成本 (换电花费 = 50 × 电价)
        SWAP_BASE_COST = 500
        for v in range(NV):
            for (c1, c2), pl in dh.cp_paths.items():
                for p in pl:
                    if str(p).startswith('swap_'):
                        iess_node = dh.virtual_iess.get(p)
                        if iess_node is not None:
                            price = dh.ele_price.get(iess_node, 1.0)
                            obj += var[('vp', v, c1, c2, p)] * SWAP_BASE_COST * price

        # 未交付惩罚
        for c in dh.customers:
            obj += (1 - var[('df', c)]) * 100000

        # 车辆固定成本
        for v in range(NV):
            obj += var[('use', v)] * 50000

        # 时间窗惩罚
        for v in range(NV):
            for c in dh.customers:
                obj += var[('cs', v, c)] * 0.01
                obj += var[('ce', v, c)] * 0.02

        model.setObjective(obj, GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)
        self.model = model
        self.var = var

    # ---------- 求解 ----------
    def solve(self):
        print("\n" + "=" * 70 + "\nVRPSolver: 求解\n" + "=" * 70)

        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('MIPGap', self.gap)
        self.model.optimize()

        status_code = self.model.Status
        status_map = {
            GRB.OPTIMAL: 'optimal',
            GRB.TIME_LIMIT: 'timelimit',
            GRB.INFEASIBLE: 'infeasible',
            GRB.UNBOUNDED: 'unbounded',
            GRB.INF_OR_UNBD: 'inf_or_unbd',
            GRB.INTERRUPTED: 'interrupted',
            GRB.NUMERIC: 'numeric',
            GRB.SUBOPTIMAL: 'suboptimal',
        }
        st = status_map.get(status_code, str(status_code))

        if self.model.SolCount > 0:
            print(f"  Status: {st}, Obj: {self.model.ObjVal:.1f}")
        else:
            print(f"  Status: {st}")

        if status_code == GRB.INFEASIBLE:
            print("  Model infeasible. Writing IIS to path_schedule_infeasible.ilp")
            self.model.computeIIS()
            self.model.write("path_schedule_infeasible.ilp")

        self.sol = {}
        if self.model.SolCount > 0:
            for k, vv in self.var.items():
                try:
                    self.sol[k] = vv.X
                except Exception:
                    self.sol[k] = 0.0
        else:
            for k in self.var:
                self.sol[k] = 0.0

        # 从 vx 解中提取路线
        dh = self.dh
        sol = self.sol
        self.routes = {}
        for vid in range(self.num_veh):
            if sol.get(('use', vid), 0) < 0.5:
                continue
            succ = {}
            for (c1, c2) in dh.cp_paths:
                if sol.get(('vx', vid, c1, c2), 0) > 0.5:
                    succ[c1] = c2
            start = None
            for c in dh.customers:
                if sol.get(('vx', vid, self.DEPOT, c), 0) > 0.5:
                    start = c
                    break
            if start is None:
                continue
            rt = [start]
            cur = start
            for _ in range(50):
                nx = succ.get(cur)
                if nx is None:
                    break
                rt.append(nx)
                cur = nx
                if sol.get(('vx', vid, cur, self.DEPOT), 0) > 0.5:
                    break
            self.routes[vid] = rt
            print(f"    Veh{vid}: DEPOT → "
                  f"{' → '.join(str(c) for c in rt)} → DEPOT")


class PostHandler:
    def __init__(self, dh: DataHandler, solver: VRPSolver):
        self.dh = dh
        self.solver = solver
        self.DEPOT = '__D__'

        print("\n" + "=" * 70 + "\nPostHandler: 后处理 (虚拟路径换电)\n" + "=" * 70)
        self._reconstruct_routes()
        self._derive_timeline()
        self._build_tables()

    def _reconstruct_routes(self):
        dh = self.dh
        sol = self.solver.sol
        self.routes = self.solver.routes
        self.path_sel = {}
        self.swap_vp = {} 

        for vid, rt in self.routes.items():
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                for p in dh.cp_paths.get((c1, c2), []):
                    if sol.get(('vp', vid, c1, c2, p), 0) > 0.5:
                        self.path_sel[(vid, c1, c2)] = p
                        if str(p).startswith('swap_'):
                            self.swap_vp[(vid, c1, c2)] = True
                        break
            print(f"  Veh{vid}: DEPOT → "
                  f"{' → '.join(str(c) for c in rt)} → DEPOT")

    def _derive_timeline(self):
        dh = self.dh
        self.derived_wt = {}
        self.derived_at = {}
        self.derived_et = {}
        self.derived_soc = {}
        self.swap_events = []

        for vid, rt in self.routes.items():
            # 载重 (逆序累加)
            w = 10.0
            for c in reversed(rt):
                w += dh.dmd.get(c, 0)
                self.derived_wt[(vid, c)] = w

            # 时间 & SOC 正向传播
            cs0 = 282.0
            t = dh.dep_t.get(rt[0], 0)
            for i, c in enumerate(rt):
                self.derived_at[(vid, c)] = t
                if i == 0:
                    soc_use_dep = dh.dep_d[c] * (
                        0.6 + 0.02 * self.derived_wt.get((vid, c), 10.0)
                    )
                    self.derived_soc[(vid, c)] = max(
                        cs0 - soc_use_dep, 56.4
                    )
                    cs0 = self.derived_soc[(vid, c)]
                else:
                    pre = rt[i - 1]
                    pid = self.path_sel.get((vid, pre, c))
                    if pid:
                        # 使用原始路径距离 (虚拟路径距离相同)
                        actual_pid = pid
                        if str(pid).startswith('swap_'):
                            actual_pid = int(str(pid).replace('swap_', ''))
                        dk = dh.cp_d.get((pre, c, pid), dh.compute_path_distance(actual_pid))
                        wt_v = self.derived_wt.get((vid, pre), 10.0)
                        soc_use = dk * (0.6 + 0.02 * wt_v)

                        if str(pid).startswith('swap_'):
                            # 虚拟路径 → 进入换电站
                            iess_node = dh.virtual_iess.get(pid)
                            kw = 282.0 - (cs0 - soc_use)
                            self.swap_events.append((
                                vid, iess_node, t,
                                max(cs0 - soc_use, 56.4), 282.0, kw
                            ))
                            self.derived_soc[(vid, c)] = 282.0
                            cs0 = 282.0
                        else:
                            self.derived_soc[(vid, c)] = max(
                                cs0 - soc_use, 56.4
                            )
                            cs0 = self.derived_soc[(vid, c)]
                    else:
                        self.derived_soc[(vid, c)] = cs0

                t += dh.svc.get(c, 0)
                self.derived_et[(vid, c)] = t

                # 下一段行驶
                if i < len(rt) - 1:
                    nc = rt[i + 1]
                    pid = self.path_sel.get((vid, c, nc))
                    if pid:
                        travel = dh.compute_path_time_at(pid, t)
                        t += travel

        print(f"  Swap events (via virtual paths): {len(self.swap_events)}")
        for ev in self.swap_events[:10]:
            print(f"    Veh{ev[0]} IESS{ev[1]} at {ev[2]:.1f}min "
                  f"swap {ev[5]:.1f}kWh")

    def _build_tables(self):
        dh = self.dh

        # ---- 车辆路径顺序表 ----
        route_rows = []
        for vid in sorted(self.routes.keys()):
            rt = self.routes[vid]
            for i, c in enumerate(rt):
                pre = 'DEPOT' if i == 0 else rt[i - 1]
                pid = self.path_sel.get((vid, pre, c)) if i > 0 else 'VIR'
                is_swap = (str(pid).startswith('swap_')) if pid != 'VIR' else False

                travel_t = 0.0
                travel_d = 0.0
                if i > 0 and pid != 'VIR':
                    n1 = dh.cn.get(str(pre), 0)
                    n2 = dh.cn.get(str(c), 0)
                    _, travel_d = dh.arpy.shortest_path(
                        n1, n2, mode='distance'
                    )
                    travel_t = (
                        self.derived_at.get((vid, c), 0)
                        - self.derived_et.get((vid, pre), 0)
                    )

                wt_ = self.derived_wt.get((vid, c), 0)
                at_ = self.derived_at.get((vid, c), 0)
                et_ = self.derived_et.get((vid, c), 0)
                soc_v = self.derived_soc.get((vid, c), 0)
                soc_out = soc_v
                sw_kwh = 0
                ie_n = ''

                if is_swap:
                    ie_n = str(dh.virtual_iess.get(pid, ''))
                    for ev in self.swap_events:
                        if (ev[0] == vid
                                and ev[1] == dh.virtual_iess.get(pid)
                                and abs(ev[2] - at_) < 1):
                            sw_kwh = ev[5]
                            soc_out = 282.0
                            break

                early = max(0, dh.tw[c][0] - et_) if c in dh.tw else 0
                delay = max(0, et_ - dh.tw[c][1]) if c in dh.tw else 0

                route_rows.append((
                    vid, i + 1, c, pid,
                    round(at_, 1), round(et_, 1),
                    round(travel_t, 1), round(travel_d, 2),
                    dh.dmd.get(c, 0), round(wt_, 3),
                    round(soc_v, 1), round(soc_out, 1),
                    '1' if sw_kwh > 0 else '0',
                    round(sw_kwh, 1), ie_n,
                    round(early, 1), round(delay, 1),
                ))

        self.route_df = pd.DataFrame(route_rows, columns=[
            '车辆ID', '序号', '客户', '路径ID',
            '到达min', '离开min', '行驶min', '行驶km',
            '交付t', '载重t', '入站SOC', '出站SOC',
            '是否换电', '换电量kWh', '换电站',
            '早到min', '延迟min',
        ])

        # ---- 换电站信息表 ----
        swap_rows = []
        for ev in self.swap_events:
            vid, iess_n, at_, si, so, kw = ev
            swap_rows.append((
                vid, iess_n, round(at_, 1),
                round(si, 1), round(so, 1), round(kw, 1),
            ))
        self.swap_df = pd.DataFrame(
            swap_rows,
            columns=[
                '车辆ID', 'IESS节点', '换电时间min',
                '进站SOC', '出站SOC', '换电量kWh',
            ]
        ) if swap_rows else pd.DataFrame()

        # ---- 车辆汇总表 ----
        v_summary = []
        for vid in sorted(self.routes.keys()):
            rt = self.routes[vid]
            last_et = self.derived_et.get((vid, rt[-1]), 0)
            total_d = 0.0
            for i in range(1, len(rt)):
                n1 = dh.cn.get(rt[i - 1], 0)
                n2 = dh.cn.get(rt[i], 0)
                _, d_seg = dh.arpy.shortest_path(n1, n2, mode='distance')
                total_d += d_seg
            total_sw = sum(
                ev[5] for ev in self.swap_events if ev[0] == vid
            )
            v_summary.append((
                vid, len(rt), round(last_et, 1),
                round(total_d, 2), round(total_sw, 1),
                ' → '.join(str(c) for c in rt),
            ))
        self.v_df = pd.DataFrame(v_summary, columns=[
            '车辆ID', '客户数', '总耗时min', '总里程km',
            '总换电kWh', '路线概要',
        ])

        print(f"  Route: {len(self.route_df)} rows, "
              f"Swap: {len(self.swap_df)} rows, "
              f"Summary: {len(self.v_df)} rows")

    def export_excel(self, path='path_schedule_scip_result.xlsx'):
        out = {}
        if hasattr(self, 'v_df') and not self.v_df.empty:
            out['1_车辆汇总'] = self.v_df
        if hasattr(self, 'route_df') and not self.route_df.empty:
            out['2_车辆路径顺序'] = self.route_df
        if hasattr(self, 'swap_df') and not self.swap_df.empty:
            out['3_换电站信息'] = self.swap_df

        if not out:
            print("  WARNING: No data")
            return

        with pd.ExcelWriter(path, engine='openpyxl') as w:
            for sn, sd in out.items():
                sd.to_excel(w, sheet_name=sn, index=False)
        print(f"  Exported: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    dh = DataHandler(num_customers=5, depot_node=10)
    solver = VRPSolver(dh, time_limit=120, gap=0.01, num_veh=5)
    solver.solve()
    ph = PostHandler(dh, solver)
    ph.export_excel('path_schedule_scip_result.xlsx')
    print("\nDONE!")