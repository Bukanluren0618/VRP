import numpy as np
import pandas as pd
import random
# import pyscipopt as gp
import gurobipy as gp
from gurobipy import GRB
import warnings
from collections import defaultdict
from road_graph import RoadNetwork
# from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2

warnings.filterwarnings('ignore')
random.seed(10)
np.random.seed(10)


class DataHandler:
    def __init__(self, num_customers=10, depot_node=10, iess_node=[],
                 iess_price={}, iess_cap={},
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

        # ---- 用 arc_res time=0 替换 free_flow_time_h ----
        for pid in self.pnodes:
            self.pth_raw[pid] = self.compute_path_time(pid) / 60.0
        for vp_id in self.is_virtual:
            orig_pid = int(vp_id.replace('swap_', ''))
            self.pth_raw[vp_id] = self.compute_path_time(orig_pid) / 60.0

        T = 60
        SWAP_WAIT_TIME_MIN = 15
        self.cp_t = {}
        self.cp_d = {}
        for (c1, c2), pl in self.cp_paths.items():
            for p in pl:
                if str(p).startswith('swap_'):
                    orig_pid = int(str(p).replace('swap_', ''))
                    base_t = self.cp_t.get(
                        (c1, c2, orig_pid), self.compute_path_time(orig_pid))
                    self.cp_t[(c1, c2, p)] = base_t + SWAP_WAIT_TIME_MIN
                    self.cp_d[(c1, c2, p)] = self.cp_d.get(
                        (c1, c2, orig_pid), self.compute_path_distance(orig_pid))
                else:
                    self.cp_t[(c1, c2, p)] = self.pth_raw.get(p, 0) * T
                    self.cp_d[(c1, c2, p)] = self.compute_path_distance(p)

        # ---- Depot 距离 ----
        self.dep_t = {}
        self.dep_d = {}
        for c in self.customers:
            _, d = self.arpy.shortest_path(
                self.depot_node, self.cn[c], mode='distance')
            _, t_h = self.arpy.shortest_path(
                self.depot_node, self.cn[c], mode='time')
            self.dep_d[c] = d
            self.dep_t[c] = t_h * T
        print(f"  Depot node: {self.depot_node}")

        # ---- 客户任务数据 ----
        ts = self.task_df[self.task_df['customer_id'].isin(self.customers)]
        self.dmd = ts.set_index('customer_id')['demand'].to_dict()
        self.svc = {
            k: v * T
            for k, v in ts.set_index('customer_id')['service_time_h'].items()}
        self.tw = {}
        for _, r in ts.iterrows():
            self.tw[r['customer_id']] = (
                r['ready_time_h'] * T,
                r['due_time_h'] * T)
        self.total_dmd = sum(self.dmd.values())
        self.NV_MAX = max(2, int(np.ceil(self.total_dmd / 20.0)) + 1)
        self.NV_MAX = min(self.NV_MAX, len(self.customers))
        print(f"  Total demand: {self.total_dmd:.1f}t, "
              f"Max vehicles: {self.NV_MAX}")
        print(f"  Virtual IESS paths: {len(self.is_virtual)}")

        # ---- 客户坐标 (用于分组) ----
        self._build_customer_coords()

    def _build_customer_coords(self):
        """构建客户坐标映射，用于空间分组"""
        self.cust_coords = {}
        self.cust_node = {}
        for c in self.customers:
            node_id = self.cn.get(c)
            self.cust_node[c] = node_id if node_id else 0

    def group_customers(self, group_size=8):
        """将客户按空间坐标K-Means分组，每组约group_size个任务点"""
        n_customers = len(self.customers)
        if n_customers <= group_size:
            print(f"  [Grouping] Only {n_customers} customers, no grouping needed")
            return [list(self.customers)]

        coords = []
        for c in self.customers:
            node_id = self.cust_node.get(c, 0)
            dep_d = self.dep_d.get(c, 0)
            coords.append([node_id % 73, dep_d])
        coords = np.array(coords)

        n_groups = max(1, int(np.ceil(n_customers / group_size)))
        n_groups = min(n_groups, n_customers)
        if n_groups == 1:
            return [list(self.customers)]

        # kmeans = KMeans(n_clusters=n_groups, random_state=10, n_init='auto')
        centroids, labels = kmeans2(coords, k=n_groups, minit='points')

        groups = defaultdict(list)
        for i, c in enumerate(self.customers):
            groups[int(labels[i])].append(c)
        group_list = list(groups.values())

        balanced = []
        for g in group_list:
            if len(g) > 20:
                sub_n = max(1, int(np.ceil(len(g) / group_size)))
                sub_coords = np.array(
                    [coords[self.customers.index(c)] for c in g])
                _,sub_labels = kmeans2(sub_coords, k=min(sub_n, len(g)), minit='points')
                sub_groups = defaultdict(list)
                for j, c in enumerate(g):
                    sub_groups[int(sub_labels[j])].append(c)
                balanced.extend(list(sub_groups.values()))
            else:
                balanced.append(g)

        print(f"  [Grouping] {n_customers} customers -> {len(balanced)} groups: "
              f"{[len(g) for g in balanced]}")
        return balanced

    def get_path_arcs(self, pid):
        actual_pid = pid
        if str(pid).startswith('swap_'):
            actual_pid = int(str(pid).replace('swap_', ''))
        np_ = self.pnodes.get(actual_pid)
        if np_ is None or len(np_) < 2:
            arcs = self.path_df.loc[
                self.path_df['path_id'] == actual_pid, 'arc_path'].values
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
    """
    VRP 求解器。
    可通过 customer_subset 指定只求解一部分客户（分组求解时用）。
    当 customer_subset 为 None 时求解 dh.customers 全集。
    """
    def __init__(self, dh: DataHandler, time_limit=300, gap=0.05,
                 num_veh=None, customer_subset=None):
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

        if customer_subset is not None:
            self._active_customers = list(customer_subset)
        else:
            self._active_customers = list(dh.customers)

        total_dmd = sum(dh.dmd.get(c, 0) for c in self._active_customers)
        nv_auto = max(2, int(np.ceil(total_dmd / 30.0)) + 1)
        nv_auto = min(nv_auto, len(self._active_customers))
        self.num_veh = num_veh if num_veh else nv_auto

        # 预过滤：只保留组内OD对
        active_set = set(self._active_customers)
        self._active_cp_paths = {
            (c1, c2): pl
            for (c1, c2), pl in dh.cp_paths.items()
            if c1 in active_set and c2 in active_set
        }
        self._active_out_nb = defaultdict(list)
        self._active_in_nb = defaultdict(list)
        for (c1, c2) in self._active_cp_paths:
            self._active_out_nb[c1].append(c2)
            self._active_in_nb[c2].append(c1)

        self._build_model()

    @property
    def customers(self):
        return self._active_customers

    @property
    def cp_paths(self):
        return self._active_cp_paths

    @property
    def out_nb(self):
        return self._active_out_nb

    @property
    def in_nb(self):
        return self._active_in_nb

    def _build_model(self):
        dh = self.dh
        custs = self.customers
        cp_paths = self.cp_paths
        out_nb = self.out_nb
        in_nb = self.in_nb
        NV = self.num_veh
        D = self.DEPOT
        M = self.M

        model = gp.Model('VRP_VirtPath')
        var = {}

        # ---- 车辆变量 ----
        for v in range(NV):
            var[('use', v)] = model.addVar(vtype='B')
            var[('vmt', v)] = model.addVar(vtype='C', lb=0)

        # ---- 节点变量 ----
        for v in range(NV):
            for c in custs + [D]:
                var[('vv', v, c)] = model.addVar(vtype='B')
                var[('wt', v, c)] = model.addVar(vtype='C', lb=0)
                var[('soc', v, c)] = model.addVar(
                    vtype='C', lb=0, ub=self.BATT_CAP)
                var[('at', v, c)] = model.addVar(vtype='C', lb=0)
                var[('st', v, c)] = model.addVar(vtype='C', lb=0)
                var[('et', v, c)] = model.addVar(vtype='C', lb=0)
                if c != D:
                    var[('cs', v, c)] = model.addVar(vtype='C', lb=0)
                    var[('ce', v, c)] = model.addVar(vtype='C', lb=0)
        for c in custs:
            var[('df', c)] = model.addVar(vtype='B')

        # ---- 弧和路径变量 ----
        for v in range(NV):
            for c in custs:
                var[('vx', v, D, c)] = model.addVar(vtype='B')
                var[('vx', v, c, D)] = model.addVar(vtype='B')
            for (c1, c2), pl in cp_paths.items():
                var[('vx', v, c1, c2)] = model.addVar(vtype='B')
                var[('vdur', v, c1, c2)] = model.addVar(vtype='C', lb=0)
                var[('vz', v, c1, c2)] = model.addVar(vtype='C', lb=0)
                for p in pl:
                    var[('vp', v, c1, c2, p)] = model.addVar(vtype='B')

        # print(f"  Vars: {model.getNVars()}")

        # 需求覆盖
        for c in custs:
            model.addConstr(
                gp.quicksum(var[('vv', v, c)] for v in range(NV))
                == var[('df', c)])
            model.addConstr(var[('df', c)] == 1)

        # 车辆使用
        for v in range(NV):
            model.addConstr(
                gp.quicksum(var[('vv', v, c)] for c in custs)
                <= var[('use', v)] * M)

        #  流量守恒
        for v in range(NV):
            d_out = gp.quicksum(var[('vx', v, D, c)] for c in custs)
            d_in = gp.quicksum(var[('vx', v, c, D)] for c in custs)
            model.addConstr(d_out == var[('vv', v, D)])
            model.addConstr(d_in == var[('vv', v, D)])
            model.addConstr(var[('vv', v, D)] <= var[('use', v)])

            for c in custs:
                inf_ = var[('vx', v, D, c)]
                for pre in in_nb.get(c, []):
                    if ('vx', v, pre, c) in var:
                        inf_ += var[('vx', v, pre, c)]
                model.addConstr(inf_ == var[('vv', v, c)])

                outf = var[('vx', v, c, D)]
                for nx in out_nb.get(c, []):
                    if ('vx', v, c, nx) in var:
                        outf += var[('vx', v, c, nx)]
                model.addConstr(outf == var[('vv', v, c)])

        #  路径选择 vx = Sum(vp)
        for v in range(NV):
            for (c1, c2), pl in cp_paths.items():
                model.addConstr(
                    var[('vx', v, c1, c2)]
                    == gp.quicksum(var[('vp', v, c1, c2, p)] for p in pl))

        #  vdur = Sum(vp * travel_time)
        for v in range(NV):
            for (c1, c2), pl in cp_paths.items():
                model.addConstr(
                    var[('vdur', v, c1, c2)]
                    == gp.quicksum(
                        var[('vp', v, c1, c2, p)] *
                        dh.cp_t.get((c1, c2, p), 0) for p in pl))

        #  vz 线性化
        for v in range(NV):
            for (c1, c2) in cp_paths:
                model.addConstr(var[('vz', v, c1, c2)] <=
                              var[('vx', v, c1, c2)] * M)
                model.addConstr(var[('vz', v, c1, c2)] <=
                              var[('vdur', v, c1, c2)])
                model.addConstr(var[('vz', v, c1, c2)] >=
                              var[('vdur', v, c1, c2)] -
                              (1 - var[('vx', v, c1, c2)]) * M)

        #  时间约束
        for v in range(NV):
            model.addConstr(var[('at', v, D)] == 0)
            model.addConstr(var[('st', v, D)] == 0)
            model.addConstr(var[('et', v, D)] == 0)

            for c in custs:
                sv = dh.svc[c]
                tw0, tw1 = dh.tw[c]

                model.addConstr(
                    var[('at', v, c)] >= dh.dep_t[c] -
                    (1 - var[('vx', v, D, c)]) * M)
                for pre in in_nb.get(c, []):
                    if ('vx', v, pre, c) in var:
                        model.addConstr(
                            var[('at', v, c)] >=
                            var[('et', v, pre)] + var[('vz', v, pre, c)] -
                            (1 - var[('vx', v, pre, c)]) * M)
                model.addConstr(var[('st', v, c)] >= var[('at', v, c)])
                model.addConstr(var[('et', v, c)] >=
                              var[('st', v, c)] + sv * var[('vv', v, c)])
                model.addConstr(var[('at', v, c)] <= var[('vv', v, c)] * M)
                model.addConstr(var[('st', v, c)] <= var[('vv', v, c)] * M)
                model.addConstr(var[('et', v, c)] <= var[('vv', v, c)] * M)
                model.addConstr(var[('et', v, c)] <= var[('vmt', v)])
                model.addConstr(var[('cs', v, c)] >= tw0 - var[('st', v, c)])
                model.addConstr(var[('ce', v, c)] >= var[('et', v, c)] - tw1)

        # 载重约束
        for v in range(NV):
            model.addConstr(var[('wt', v, D)] == self.EMPTY * var[('use', v)])
            model.addConstr(
                gp.quicksum(var[('vv', v, c)] * dh.dmd[c] for c in custs)
                <= self.MAX_PL)
            for c in custs:
                dmd = dh.dmd[c]
                model.addConstr(var[('wt', v, c)] >= self.EMPTY + dmd -
                              (1 - var[('vv', v, c)]) * M)
                model.addConstr(var[('wt', v, c)] >= var[('wt', v, D)] -
                              (1 - var[('vx', v, D, c)]) * M)
                model.addConstr(var[('wt', v, c)] <= self.EMPTY + self.MAX_PL)
                for nx in out_nb.get(c, []):
                    if ('vx', v, c, nx) in var:
                        model.addConstr(
                            var[('wt', v, nx)] >= var[('wt', v, c)] - dmd -
                            (1 - var[('vx', v, c, nx)]) * M)
                        model.addConstr(
                            var[('wt', v, nx)] <= var[('wt', v, c)] - dmd +
                            (1 - var[('vx', v, c, nx)]) * M)
                model.addConstr(var[('wt', v, c)] <= var[('vv', v, c)] * M)

        # SOC 约束
        for v in range(NV):
            model.addConstr(var[('soc', v, D)] ==
                          self.BATT_CAP * var[('use', v)])
            for c in custs:
                soc_dep = dh.dep_d[c] * (self.BK + self.WK * var[('wt', v, D)])
                model.addConstr(var[('soc', v, c)] >= self.BATT_CAP - soc_dep -
                              (1 - var[('vx', v, D, c)]) * M)
                model.addConstr(var[('soc', v, c)] <= self.BATT_CAP +
                              (1 - var[('vx', v, D, c)]) * M)
                for pre in in_nb.get(c, []):
                    if ('vx', v, pre, c) not in var:
                        continue
                    for p in cp_paths.get((pre, c), []):
                        dist = dh.cp_d.get((pre, c, p), 0)
                        soc_use = dist * (self.BK + self.WK *
                                          var[('wt', v, pre)])
                        if str(p).startswith('swap_'):
                            model.addConstr(
                                var[('soc', v, c)] >= self.BATT_CAP -
                                (1 - var[('vp', v, pre, c, p)]) * M)
                            model.addConstr(
                                var[('soc', v, c)] <= self.BATT_CAP +
                                (1 - var[('vp', v, pre, c, p)]) * M)
                        else:
                            model.addConstr(
                                var[('soc', v, c)] >=
                                var[('soc', v, pre)] - soc_use -
                                (1 - var[('vp', v, pre, c, p)]) * M)
                            model.addConstr(
                                var[('soc', v, c)] <=
                                var[('soc', v, pre)] - soc_use +
                                (1 - var[('vp', v, pre, c, p)]) * M)
                model.addConstr(var[('soc', v, c)] >=
                              self.BATT_MIN * var[('vv', v, c)])
                model.addConstr(var[('soc', v, c)] <=
                              self.BATT_CAP * var[('vv', v, c)])

        # print(f"  Constraints: {model.getNConss()}")

        obj = gp.LinExpr()

        # 人力成本
        for v in range(NV):
            obj += var[('vmt', v)] * 200

        # 虚拟路径换电成本
        SWAP_BASE_COST = 50
        for v in range(NV):
            for (c1, c2), pl in cp_paths.items():
                for p in pl:
                    if str(p).startswith('swap_'):
                        iess_node = dh.virtual_iess.get(p)
                        if iess_node is not None:
                            price = dh.ele_price.get(iess_node, 1.0)
                            obj += var[('vp', v, c1, c2, p)] * SWAP_BASE_COST * price

        # 未交付惩罚
        for c in custs:
            obj += (1 - var[('df', c)]) * 100000

        # 车辆固定成本
        for v in range(NV):
            obj += var[('use', v)] * 50000

        # 时间窗惩罚
        for v in range(NV):
            for c in custs:
                obj += var[('cs', v, c)] * 0.01
                obj += var[('ce', v, c)] * 0.02

        model.setObjective(obj,  GRB.MINIMIZE)
        # model.hideOutput()
        model.Params.OutputFlag = 0
        self.model = model
        self.var = var

    def solve(self):
        print("\n" + "=" * 70 +
              f"\nVRPSolver: ({len(self.customers)} customers, {self.num_veh} veh)"
              + "\n" + "=" * 70)
        self.model.Params.TimeLimit = self.time_limit
        self.model.Params.MIPGap = self.gap
        self.model.optimize()

        status_map = {
                    GRB.OPTIMAL: 'optimal',
                    GRB.INFEASIBLE: 'infeasible',
                    GRB.INF_OR_UNBD: 'inf_or_unbd',
                    GRB.UNBOUNDED: 'unbounded',
                    GRB.TIME_LIMIT: 'time_limit',
                    GRB.NODE_LIMIT: 'node_limit',
                    GRB.ITERATION_LIMIT: 'iteration_limit',
                    GRB.SOLUTION_LIMIT: 'solution_limit',
                    GRB.INTERRUPTED: 'interrupted',
                    GRB.SUBOPTIMAL: 'suboptimal',
                    GRB.NUMERIC: 'numeric',
                }
        st = status_map.get(self.model.Status, f'status_{self.model.Status}')

        if self.model.SolCount > 0:
            print(f"  Status: {st}, Obj: {self.model.ObjVal:.1f}")
        else:
            print(f"  Status: {st}, no feasible solution")

        self.sol = {}
        if self.model.SolCount > 0:
            for k, vv in self.var.items():
                try:
                    self.sol[k] = vv.X
                except (AttributeError, gp.GurobiError):
                    self.sol[k] = 0.0
        else:
            self.sol = {k: 0.0 for k in self.var}
        

        dh = self.dh
        custs = self.customers
        cp_paths = self.cp_paths
        sol = self.sol
        self.routes = {}
        for vid in range(self.num_veh):
            if sol.get(('use', vid), 0) < 0.5:
                continue
            succ = {}
            for (c1, c2) in cp_paths:
                if sol.get(('vx', vid, c1, c2), 0) > 0.5:
                    succ[c1] = c2
            start = None
            for c in custs:
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
            print(f"    Veh{vid}: DEPOT -> "
                  f"{' -> '.join(str(c) for c in rt)} -> DEPOT")


class PostHandler:
    def __init__(self, dh: DataHandler, solver: VRPSolver):
        self.dh = dh
        self.solver = solver
        self.DEPOT = '__D__'

        print("\n" + "=" * 70 +
              "\nPostHandler: 后处理 (虚拟路径换电)\n" + "=" * 70)
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
            print(f"  Veh{vid}: DEPOT -> "
                  f"{' -> '.join(str(c) for c in rt)} -> DEPOT")

    def _derive_timeline(self):
        dh = self.dh
        self.derived_wt = {}
        self.derived_at = {}
        self.derived_et = {}
        self.derived_soc = {}
        self.swap_events = []

        for vid, rt in self.routes.items():
            w = 10.0
            for c in reversed(rt):
                w += dh.dmd.get(c, 0)
                self.derived_wt[(vid, c)] = w

            cs0 = 282.0
            t = dh.dep_t.get(rt[0], 0)
            for i, c in enumerate(rt):
                self.derived_at[(vid, c)] = t
                if i == 0:
                    soc_use_dep = dh.dep_d[c] * (
                        0.6 + 0.02 * self.derived_wt.get((vid, c), 10.0))
                    self.derived_soc[(vid, c)] = max(
                        cs0 - soc_use_dep, 56.4)
                    cs0 = self.derived_soc[(vid, c)]
                else:
                    pre = rt[i - 1]
                    pid = self.path_sel.get((vid, pre, c))
                    if pid:
                        actual_pid = pid
                        if str(pid).startswith('swap_'):
                            actual_pid = int(str(pid).replace('swap_', ''))
                        dk = dh.cp_d.get((pre, c, pid),
                                         dh.compute_path_distance(actual_pid))
                        wt_v = self.derived_wt.get((vid, pre), 10.0)
                        soc_use = dk * (0.6 + 0.02 * wt_v)

                        if str(pid).startswith('swap_'):
                            iess_node = dh.virtual_iess.get(pid)
                            kw = 282.0 - (cs0 - soc_use)
                            self.swap_events.append((
                                vid, iess_node, t,
                                max(cs0 - soc_use, 56.4), 282.0, kw))
                            self.derived_soc[(vid, c)] = 282.0
                            cs0 = 282.0
                        else:
                            self.derived_soc[(vid, c)] = max(
                                cs0 - soc_use, 56.4)
                            cs0 = self.derived_soc[(vid, c)]
                    else:
                        self.derived_soc[(vid, c)] = cs0

                t += dh.svc.get(c, 0)
                self.derived_et[(vid, c)] = t

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

        route_rows = []
        for vid in sorted(self.routes.keys()):
            rt = self.routes[vid]
            for i, c in enumerate(rt):
                pre = 'DEPOT' if i == 0 else rt[i - 1]
                pid = self.path_sel.get((vid, pre, c)) if i > 0 else 'VIR'
                is_swap = (str(pid).startswith('swap_')
                           if pid != 'VIR' else False)

                travel_t = 0.0
                travel_d = 0.0
                if i > 0 and pid != 'VIR':
                    n1 = dh.cn.get(str(pre), 0)
                    n2 = dh.cn.get(str(c), 0)
                    _, travel_d = dh.arpy.shortest_path(
                        n1, n2, mode='distance')
                    travel_t = (self.derived_at.get((vid, c), 0) -
                                self.derived_et.get((vid, pre), 0))

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
                        if (ev[0] == vid and
                                ev[1] == dh.virtual_iess.get(pid) and
                                abs(ev[2] - at_) < 1):
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
                    round(soc_out - sw_kwh, 1), round(soc_out, 1),
                    '1' if sw_kwh > 0 else '0',
                    round(sw_kwh, 1), ie_n,
                    round(early, 1), round(delay, 1)))

        self.route_df = pd.DataFrame(route_rows, columns=[
            '车辆ID', '序号', '客户', '路径ID',
            '到达min', '离开min', '行驶min', '行驶km',
            '交付t', '载重t', '入站SOC', '出站SOC',
            '是否换电', '换电量kWh', '换电站',
            '早到min', '延迟min'])

        swap_rows = []
        for ev in self.swap_events:
            vid, iess_n, at_, si, so, kw = ev
            swap_rows.append((vid, iess_n, round(at_, 1),
                              round(si, 1), round(so, 1), round(kw, 1)))
        self.swap_df = pd.DataFrame(swap_rows, columns=[
            '车辆ID', 'IESS节点', '换电时间min',
            '进站SOC', '出站SOC', '换电量kWh']) if swap_rows else pd.DataFrame()

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
            total_sw = sum(ev[5] for ev in self.swap_events if ev[0] == vid)
            v_summary.append((
                vid, len(rt), round(last_et, 1),
                round(total_d, 2), round(total_sw, 1),
                ' -> '.join(str(c) for c in rt)))
        self.v_df = pd.DataFrame(v_summary, columns=[
            '车辆ID', '客户数', '总耗时min', '总里程km',
            '总换电kWh', '路线概要'])

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


class GroupedSolver:
    def __init__(self, dh: DataHandler, group_size=8, time_limit=300,
                 gap=0.05):
        self.dh = dh
        self.group_size = group_size
        self.time_limit = time_limit
        self.gap = gap

    def solve(self):
        dh = self.dh
        groups = dh.group_customers(group_size=self.group_size)

        if len(groups) == 1:
            print("\n[GroupedSolver] 无需分组, 直接全量求解")
            solver = VRPSolver(dh, time_limit=self.time_limit, gap=self.gap)
            solver.solve()
            ph = PostHandler(dh, solver)
            self._merge_posthandler_results([ph])
            return

        print(f"\n{'=' * 70}")
        print(f"[GroupedSolver] 分组求解: {len(groups)} 组, "
              f"每组 {self.time_limit}s")
        print(f"{'=' * 70}")

        all_phs = []
        for gid, group in enumerate(groups):
            print(f"\n{'*' * 40}")
            print(f"  Group {gid + 1}/{len(groups)}: {len(group)} customers "
                  f"({group[0] if group else ''} ...)")
            print(f"{'*' * 40}")

            try:
                solver = VRPSolver(dh,
                                   customer_subset=list(group),
                                   time_limit=self.time_limit,
                                   gap=self.gap)
                solver.solve()
                ph = PostHandler(dh, solver)
                all_phs.append(ph)
            except Exception as e:
                print(f"  [WARNING] Group {gid + 1} failed: {e}")

        self._merge_posthandler_results(all_phs)

    def _merge_posthandler_results(self, phs):
        all_route_dfs = []
        all_swap_dfs = []
        all_v_dfs = []
        global_vid = 0

        for ph in phs:
            if not hasattr(ph, 'route_df') or ph.route_df.empty:
                continue
            vid_map = {}
            for old_vid in ph.route_df['车辆ID'].unique():
                vid_map[old_vid] = global_vid
                global_vid += 1

            ph.route_df['车辆ID'] = ph.route_df['车辆ID'].map(vid_map)
            if hasattr(ph, 'swap_df') and not ph.swap_df.empty:
                ph.swap_df['车辆ID'] = ph.swap_df['车辆ID'].map(vid_map)
            if hasattr(ph, 'v_df') and not ph.v_df.empty:
                ph.v_df['车辆ID'] = ph.v_df['车辆ID'].map(vid_map)

            all_route_dfs.append(ph.route_df)
            if hasattr(ph, 'swap_df') and not ph.swap_df.empty:
                all_swap_dfs.append(ph.swap_df)
            if hasattr(ph, 'v_df') and not ph.v_df.empty:
                all_v_dfs.append(ph.v_df)

        self.route_df = (pd.concat(all_route_dfs, ignore_index=True)
                         if all_route_dfs else pd.DataFrame())
        self.swap_df = (pd.concat(all_swap_dfs, ignore_index=True)
                        if all_swap_dfs else pd.DataFrame())
        self.v_df = (pd.concat(all_v_dfs, ignore_index=True)
                     if all_v_dfs else pd.DataFrame())

        print(f"\n[GroupedSolver] Total: {len(self.v_df)} vehicles, "
              f"{len(self.route_df)} route rows, "
              f"{len(self.swap_df)} swap events")

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


if __name__ == '__main__':
    IESS_NODES = [4, 3, 10, 9, 18]
    iess_cap = {iess: 2 for iess in IESS_NODES}
    iess_price = {4: 0.6, 3: 0.6, 10: 0.6, 9: 6.6, 18: 0.6,
                  42: 1.67, 36: 1.31, 71: 0.4, 23: 0.4, 62: 0.67,
                  73: 1.42, 20: 0.4, 11: 1.14, 46: 10.05, 28: 6.63}
    dh = DataHandler(num_customers=80, depot_node=10,
                     iess_node=IESS_NODES, iess_cap=iess_cap,
                     iess_price=iess_price)
    grouped_solver = GroupedSolver(dh, group_size=8, time_limit=300, gap=0.05)
    grouped_solver.solve()
    grouped_solver.export_excel('path_schedule_scip_result.xlsx')
    print("\nDONE!")