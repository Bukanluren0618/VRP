import numpy as np
import pandas as pd
import random
import gurobipy as gp
from gurobipy import GRB
import warnings
from collections import defaultdict
from road_graph import RoadNetwork

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)


class DataHandler:
    def __init__(self, num_customers=10, depot_node=10,
                 pickle_path='./raw_data_bpr.pkl',
                 arc_res_path='algo_bpr_res.xlsx',
                 iess_node=None, iess_price=None, iess_cap=None):
        self.num_customers = num_customers
        self.depot_node = depot_node
        print("=" * 70 + "\nMilpDataHandler: 加载数据\n" + "=" * 70)

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

        # ---- arc_res 时间 ----
        arc_res_df = pd.read_excel(arc_res_path, sheet_name='arc_res')
        t0_mask = arc_res_df['time'] == 0
        self.arc_time_t0 = arc_res_df[t0_mask].set_index('arc')['run_time'].to_dict()
        self.arc_time_all = {}
        for t in sorted(arc_res_df['time'].unique()):
            mask_t = arc_res_df['time'] == t
            self.arc_time_all[t] = arc_res_df[mask_t].set_index('arc')['run_time'].to_dict()

        all_nodes = self.node_df['node_id'].tolist()
        if iess_node is not None:
            self.IESS_NODES = list(iess_node)
        else:
            self.IESS_NODES = random.sample(all_nodes, 5)

        if iess_price is not None:
            self.ele_price = dict(iess_price)
        else:
            self.ele_price = {
                n: round(random.random() * 100, 2) for n in self.IESS_NODES
            }

        if iess_cap is not None:
            self.iess_cap = dict(iess_cap)
        else:
            self.iess_cap = {
                n: np.ceil(random.random() * 10) for n in self.IESS_NODES
            }

        # ---- 路网 ----
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

        # ---- 路径数据 + 补全 ----
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
                                tt += t0 * 60
                                break
                    self.pnodes[pid_c] = np_
                    self.pth_raw[pid_c] = tt
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
        print(f"  OD pairs: {len(self.cp_paths)}")
        # ---- IESS 检测 ----
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
        iess_original = {p for p in self.iess_p}
        new_cp = defaultdict(list)
        for (c1, c2), pl in self.cp_paths.items():
            for p in pl:
                new_cp[(c1, c2)].append(p)
                if p in iess_original:
                    vp_id = 'swap_' + str(p)
                    new_cp[(c1, c2)].append(vp_id)
                    self.is_virtual[vp_id] = True
                    self.virtual_iess[vp_id] = self.piess[p]
                    # 复用原路径的node_path
                    self.pnodes[vp_id] = self.pnodes.get(p)
        self.cp_paths = dict(new_cp)
        self.iess_p = {'swap_' + str(p) for p in self.iess_p}
        self.piess = {'swap_' + str(p):n for p, n in self.piess.items()}

        self.out_nb = defaultdict(list)
        self.in_nb = defaultdict(list)
        for (c1, c2) in self.cp_paths:
            self.out_nb[c1].append(c2)
            self.in_nb[c2].append(c1)
        

        
        
        

        # ---- 用 arc_res time=0 替换 (含虚拟路径) ----
        T = 60
        for pid in self.pnodes:
            self.pth_raw[pid] = self.compute_path_time(pid) / 60
        for vp_id in self.is_virtual:
            orig = int(vp_id.replace('swap_', ''))
            self.pth_raw[vp_id] = self.compute_path_time(orig) / 60

        # 虚拟路径 = 原路径时间 + 换电等待15min ----
        
        SWAP_WAIT = 15
        self.cp_t = {}
        self.cp_d = {}
        for (c1, c2), pl in self.cp_paths.items():
            for p in pl:
                if str(p).startswith('swap_'):
                    orig = int(str(p).replace('swap_', ''))
                    self.cp_d[(c1, c2, p)] = self.cp_d.get(
                        (c1, c2, orig), self.compute_path_distance(orig))
                    base_t = self.cp_t.get((c1, c2, orig),
                                           self.compute_path_time(orig) )
                    self.cp_t[(c1, c2, p)] = base_t + SWAP_WAIT
                else:
                    self.cp_t[(c1, c2, p)] = self.pth_raw.get(p, 0) * T
                    self.cp_d[(c1, c2, p)] = self.compute_path_distance(p)

        # ---- OD平均属性 (Stage1用) ----
        self.cp_avg_t = {}
        self.cp_avg_d = {}
        self.cp_has_iess = {}
        for (c1, c2), pl in self.cp_paths.items():
            ts = [self.cp_t.get((c1, c2, p), 0) for p in pl]
            ds = [self.cp_d.get((c1, c2, p), 0) for p in pl]
            self.cp_avg_t[(c1, c2)] = sum(ts) / len(ts) if ts else 0
            self.cp_avg_d[(c1, c2)] = sum(ds) / len(ds) if ds else 0
            self.cp_has_iess[(c1, c2)] = any(p in self.iess_p for p in pl)

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

        # ---- 客户任务 ----
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
        self.NV_MAX = max(2, int(np.ceil(self.total_dmd / 30)) + 1)
        self.NV_MAX =  min(self.NV_MAX, len(self.customers))
        print(f"  Total demand: {self.total_dmd:.1f}t, "
              f"Max vehicles: {self.NV_MAX}")


    def get_path_arcs(self, pid):
        np_ = self.pnodes.get(pid)
        if np_ is None or len(np_) < 2:
            arcs = self.path_df.loc[
                self.path_df['path_id'] == pid, 'arc_path'
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
        return sum(self.arc_time_t0.get(a, 999.0)  for a in al) 

    def compute_path_distance(self, pid):
        np_ = self.pnodes.get(pid)
        if np_ is None or len(np_) < 2:
            return self.plen_raw.get(pid, 0.0)
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


class TwoStageSolver:
    @staticmethod
    def _status_name(model):
        status_map = {
            GRB.OPTIMAL: 'optimal',
            GRB.INFEASIBLE: 'infeasible',
            GRB.UNBOUNDED: 'unbounded',
            GRB.INF_OR_UNBD: 'inf_or_unbd',
            GRB.TIME_LIMIT: 'timelimit',
            GRB.SOLUTION_LIMIT: 'bestsollimit',
            GRB.INTERRUPTED: 'interrupted',
            GRB.NUMERIC: 'numeric',
            GRB.SUBOPTIMAL: 'suboptimal',
        }
        return status_map.get(model.Status, str(model.Status))

    def __init__(self, dh: DataHandler, time_limit=120, gap=0.05,
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
        self.WK = 0.08
        self.SOC_EST = self.EMPTY + self.MAX_PL * 0.8
        self.num_veh = num_veh if num_veh else dh.NV_MAX

        print("\n" + "=" * 70 +
              "\nTwoStageSolver: Stage1 Route + Stage2 Path\n" + "=" * 70)

    def _stage1_route_optimization(self):
        dh = self.dh
        NV = self.num_veh
        m1 = gp.Model('Stage1_Route')
        v1 = {}

        for vid in range(NV):
            v1[('use', vid)] = m1.addVar(vtype=GRB.BINARY)
        for vid in range(NV):
            for c in dh.customers + [self.DEPOT]:
                v1[('vv', vid, c)] = m1.addVar(vtype=GRB.BINARY)
                v1[('soc', vid, c)] = m1.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, ub=self.BATT_CAP)

        for vid in range(NV):
            for c in dh.customers:
                v1[('vx', vid, self.DEPOT, c)] = m1.addVar(vtype=GRB.BINARY)
                v1[('vx', vid, c, self.DEPOT)] = m1.addVar(vtype=GRB.BINARY)
            for (c1, c2) in dh.cp_paths:
                v1[('vx', vid, c1, c2)] = m1.addVar(vtype=GRB.BINARY)

        for c in dh.customers:
            m1.addConstr(gp.quicksum(v1[('vv', vid, c)]
                        for vid in range(NV)) == 1)

        for vid in range(NV):
            m1.addConstr(gp.quicksum(v1[('vv', vid, c)]
                        for c in dh.customers) <= v1[('use', vid)] * self.M)

        for vid in range(NV):
            d_out = gp.quicksum(v1[('vx', vid, self.DEPOT, c)]
                                  for c in dh.customers)
            d_in = gp.quicksum(v1[('vx', vid, c, self.DEPOT)]
                                 for c in dh.customers)
            m1.addConstr(d_out == v1[('vv', vid, self.DEPOT)])
            m1.addConstr(d_in == v1[('vv', vid, self.DEPOT)])
            m1.addConstr(v1[('vv', vid, self.DEPOT)] <= v1[('use', vid)])

            for c in dh.customers:
                inf_ = v1[('vx', vid, self.DEPOT, c)]
                for pre in dh.in_nb.get(c, []):
                    if ('vx', vid, pre, c) in v1:
                        inf_ += v1[('vx', vid, pre, c)]
                m1.addConstr(inf_ == v1[('vv', vid, c)])

                outf = v1[('vx', vid, c, self.DEPOT)]
                for nx in dh.out_nb.get(c, []):
                    if ('vx', vid, c, nx) in v1:
                        outf += v1[('vx', vid, c, nx)]
                m1.addConstr(outf == v1[('vv', vid, c)])

        for vid in range(NV):
            m1.addConstr(gp.quicksum(v1[('vv', vid, c)] * dh.dmd[c]
                        for c in dh.customers) <= self.MAX_PL)

        for vid in range(NV):
            m1.addConstr(v1[('soc', vid, self.DEPOT)]
                       == self.BATT_CAP * v1[('use', vid)])
            for c in dh.customers:
                sd = dh.dep_d[c] * (self.BK + self.WK * self.SOC_EST)
                m1.addConstr(v1[('soc', vid, c)] >= self.BATT_CAP - sd -
                           (1 - v1[('vx', vid, self.DEPOT, c)]) * self.M)
                m1.addConstr(v1[('soc', vid, c)] <= self.BATT_CAP +
                           (1 - v1[('vx', vid, self.DEPOT, c)]) * self.M)
                for pre in dh.in_nb.get(c, []):
                    if ('vx', vid, pre, c) not in v1:
                        continue
                    su = dh.cp_avg_d.get((pre, c), 0) * \
                        (self.BK + self.WK * self.SOC_EST)
                    if dh.cp_has_iess.get((pre, c), False):
                        m1.addConstr(v1[('soc', vid, c)] >= self.BATT_CAP -
                                   (1 - v1[('vx', vid, pre, c)]) * self.M)
                        m1.addConstr(v1[('soc', vid, c)] <= self.BATT_CAP +
                                   (1 - v1[('vx', vid, pre, c)]) * self.M)
                    else:
                        m1.addConstr(v1[('soc', vid, c)] >= v1[
                                   ('soc', vid, pre)] - su - (1 - v1[('vx', vid, pre, c)]) * self.M)
                        m1.addConstr(v1[('soc', vid, c)] <= v1[
                                   ('soc', vid, pre)] - su + (1 - v1[('vx', vid, pre, c)]) * self.M)
                m1.addConstr(v1[('soc', vid, c)]
                           >= self.BATT_MIN * v1[('vv', vid, c)])

        obj1 = gp.LinExpr()
        for vid in range(NV):
            obj1 += v1[('use', vid)] * 5000
        for vid in range(NV):
            for c in dh.customers:
                obj1 += v1[('vx', vid, self.DEPOT, c)] *  dh.dep_d[c] * 10
                obj1 += v1[('vx', vid, c, self.DEPOT)] * dh.dep_d[c] * 10
            for (c1, c2) in dh.cp_paths:
                obj1 += v1[('vx', vid, c1, c2)] * dh.cp_avg_d.get((c1, c2), 0) * 10

        m1.setObjective(obj1, GRB.MINIMIZE)
        m1.setParam('TimeLimit', 60)
        m1.setParam('MIPGap', 0.05)
        m1.setParam('OutputFlag', 0)
        # m1.write('D://model.lp')
        m1.optimize()
        return m1, v1

    def _extract_routes(self, m1, v1):
        dh = self.dh
        s1 = self._status_name(m1)
        sol1 = {k: vv.X for k, vv in v1.items()
                if s1 in ('optimal', 'bestsollimit', 'gaplimit') and m1.SolCount > 0}
        routes = {}
        for vid in range(self.num_veh):
            if sol1.get(('use', vid), 0) < 0.5:
                continue
            succ = {}
            for (c1, c2) in dh.cp_paths:
                if sol1.get(('vx', vid, c1, c2), 0) > 0.5:
                    succ[c1] = c2
            start = None
            for c in dh.customers:
                if sol1.get(('vx', vid, self.DEPOT, c), 0) > 0.5:
                    start = c
                    break
            if start is None:
                continue
            rt = [start]
            cur = start
            for _ in range(len(dh.customers)):
                nx = succ.get(cur)
                if nx is None:
                    break
                rt.append(nx)
                cur = nx
                if sol1.get(('vx', vid, cur, self.DEPOT), 0) > 0.5:
                    break
            if rt:
                routes[vid] = rt

        covered = set()
        for rt in routes.values():
            covered.update(rt)
        missing = [c for c in dh.customers if c not in covered]
        if missing:
            rl = {vid: sum(dh.dmd.get(c, 0)
                           for c in rt) for vid, rt in routes.items()}
            mx = max(routes.keys()) if routes else -1
            for c in missing:
                dem = dh.dmd.get(c, 0)
                placed = False
                for vid in sorted(routes.keys(), key=lambda v: self.MAX_PL - rl.get(v, 0), reverse=True):
                    if rl.get(vid, 0) + dem <= self.MAX_PL and len(routes.get(vid, [])) < 10:
                        routes[vid].append(c)
                        rl[vid] += dem
                        placed = True
                        break
                if not placed:
                    mx += 1
                    routes[mx] = [c]
                    rl[mx] = dem
        return routes

    def _heuristic_routes(self):
        dh = self.dh
        rem = list(dh.customers)
        routes = {}
        vid = 0

        while rem:
            # 如果还有未分配客户但已达车数上限，强制开新车
            if vid >= self.num_veh:
                self.num_veh += 1  # 自动扩容

            cs = self.BATT_CAP
            cw = self.MAX_PL
            ct = 0.0
            rt = []
            pr = None

            # 选起始客户 (时间窗最近截止，且可达客户最多的)
            cu_scores = []
            for c in rem:
                nb_count = len(set(rem) & set(dh.out_nb.get(c, [])))
                cu_scores.append((c, dh.tw[c][1], -nb_count))
            cu = min(cu_scores, key=lambda x: (x[1], x[2]))[0]
            rem.remove(cu)

            rt.append(cu)
            cw -= dh.dmd[cu]
            ct = max(dh.tw[cu][0], ct) + dh.svc[cu]
            pr = cu

            # 贪心扩展：可达 + 载重OK + SOC OK
            while rem:
                cans = [(nc, dh.cp_paths.get((pr, nc), []))
                        for nc in rem if nc in dh.out_nb.get(pr, [])]
                if not cans:
                    # 尝试从rem中找可达pr的（可能不在out_nb中但可通过in_nb反向）
                    extended = [(nc, dh.cp_paths.get((pr, nc), []))
                                for nc in rem
                                if (pr, nc) in dh.cp_paths]
                    cans = extended if extended else cans

                best = None
                for nc, pl in cans:
                    for p in pl:
                        tm = dh.cp_t.get((pr, nc, p), 0)
                        dk = dh.cp_d.get((pr, nc, p), 0)
                        hi = p in dh.iess_p
                        sc = dk * (self.BK + self.WK * (cw + self.EMPTY))
                        # 放宽约束: 仅检查载重, SOC可通过换电解决
                        if cw < dh.dmd.get(nc, 0) and hi:
                            continue  # 换电不解决载重问题
                        if cw < dh.dmd.get(nc, 0):
                            continue
                        s = -tm + (50 if hi else 0)  # 偏好短时间 + IESS
                        if ct + tm > dh.tw[nc][0] + 120:
                            s -= 1000
                        if best is None or s > best[2]:
                            best = (nc, p, s, tm, dk, hi, dh.piess.get(p, -1))
                if best is None:
                    break  # 当前路线无法再扩展
                nc, p, _, tm, dk, hi, in_ = best
                rem.remove(nc)
                sc = dk * (self.BK + self.WK * (cw + self.EMPTY))
                cs -= sc
                if hi:
                    cs = self.BATT_CAP
                ct += tm
                cw -= dh.dmd[nc]
                rt.append(nc)
                pr = nc

            routes[vid] = rt
            vid += 1

        # 如果某车只有1个客户且载重OK，尝试将单独客户并入其他车
        covered = set()
        for rt in routes.values():
            covered.update(rt)
        missing = [c for c in dh.customers if c not in covered]
        if missing:
            # 强制每个缺失客户单独一车
            for c in missing:
                routes[vid] = [c]
                vid += 1

        return routes, vid

    def _stage2_path_selection(self, routes):
        dh = self.dh

        arcs_used = set()
        for vid, rt in routes.items():
            arcs_used.add((self.DEPOT, rt[0]))
            arcs_used.add((rt[-1], self.DEPOT))
            for i in range(len(rt) - 1):
                arcs_used.add((rt[i], rt[i + 1]))

        m2 = gp.Model('Stage2_Path')
        v2 = {}

        for vid, rt in routes.items():
            if not rt:
                continue
            for c in rt + [self.DEPOT]:
                v2[('wt', vid, c)] = m2.addVar(vtype=GRB.CONTINUOUS, lb=0)
                v2[('soc', vid, c)] = m2.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, ub=self.BATT_CAP)
                v2[('at', vid, c)] = m2.addVar(vtype=GRB.CONTINUOUS, lb=0)
                v2[('et', vid, c)] = m2.addVar(vtype=GRB.CONTINUOUS, lb=0)
            for c in rt:
                v2[('cs', vid, c)] = m2.addVar(vtype=GRB.CONTINUOUS, lb=0)
                # v2[('csa', vid, c)] = m2.addVar(vtype=GRB.BINARY)
                v2[('ce', vid, c)] = m2.addVar(vtype=GRB.CONTINUOUS, lb=0)
                # v2[('cea', vid, c)] = m2.addVar(vtype=GRB.BINARY)
            for (c1, c2), pl in dh.cp_paths.items():
                if (c1, c2) in arcs_used:
                    for p in pl:
                        v2[('vp', vid, c1, c2, p)] = m2.addVar(vtype=GRB.BINARY)
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                v2[('sw_flag', vid, c1, c2)] = m2.addVar(vtype=GRB.BINARY)

        # Path selection
        for vid, rt in routes.items():
            if not rt:
                continue
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                pl = dh.cp_paths.get((c1, c2), [])
                if pl:
                    m2.addConstr(gp.quicksum(
                        v2[('vp', vid, c1, c2, p)] for p in pl) == 1)
                hi = [p for p in pl if p in dh.iess_p]
                if hi:
                    m2.addConstr(gp.quicksum(
                        v2[('vp', vid, c1, c2, p)] for p in hi) == v2[('sw_flag', vid, c1, c2)])

        # Weight
        for vid, rt in routes.items():
            if not rt:
                continue
            m2.addConstr(v2[('wt', vid, self.DEPOT)] == self.EMPTY)
            w = self.EMPTY
            for c in reversed(rt):
                w += dh.dmd[c]
                m2.addConstr(v2[('wt', vid, c)] == w)

        # Time
        for vid, rt in routes.items():
            if not rt:
                continue
            m2.addConstr(v2[('at', vid, rt[0])] == dh.dep_t[rt[0]])
            m2.addConstr(v2[('et', vid, self.DEPOT)] == 0)
            for i, c in enumerate(rt):
                tw0, tw1 = dh.tw[c]
                m2.addConstr(v2[('et', vid, c)]
                           >= v2[('at', vid, c)] + dh.svc[c])
                if i < len(rt) - 1:
                    nc = rt[i + 1]
                    dur = gp.quicksum(
                        v2[('vp', vid, c, nc, p)] * dh.cp_t.get((c, nc, p), 0)
                        for p in dh.cp_paths.get((c, nc), []))
                    m2.addConstr(v2[('at', vid, nc)] >= v2[(
                        'et', vid, c)] + dur + v2[('sw_flag', vid, c, nc)] * 15)
                m2.addConstr(v2[('cs', vid, c)] >= tw0 - v2[('at', vid, c)])
                # m2.addConstr(v2[('cs', vid, c)] <= v2[('csa', vid, c)] * self.M)
                # m2.addConstr(v2[('cs', vid, c)] <= tw0 - v2[(
                #     'at', vid, c)] + (1 - v2[('csa', vid, c)]) * self.M)
                m2.addConstr(v2[('ce', vid, c)]
                           >= v2[('et', vid, c)] - tw1)
                # m2.addConstr(v2[('ce', vid, c)] <= v2[('cea', vid, c)] * self.M)
                # m2.addConstr(v2[('ce', vid, c)] <= v2[(
                #     'et', vid, c)] - tw1 + (1 - v2[('cea', vid, c)]) * self.M)

        # SOC
        for vid, rt in routes.items():
            if not rt:
                continue
            m2.addConstr(v2[('soc', vid, self.DEPOT)] == self.BATT_CAP)
            c0 = rt[0]
            sf = max(self.BATT_CAP - dh.dep_d[c0] *
                     (self.BK + self.WK * (self.EMPTY + sum(dh.dmd[cc] for cc in rt))), self.BATT_MIN)
            m2.addConstr(v2[('soc', vid, c0)] == sf)
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                for p in dh.cp_paths.get((c1, c2), []):
                    dk = dh.cp_d.get((c1, c2, p), 0)
                    su = dk * (self.BK + self.WK * v2[('wt', vid, c1)])
                    if str(p).startswith('swap_'):
                        m2.addConstr(v2[('soc', vid, c2)] >= self.BATT_CAP -
                                   (1 - v2[('vp', vid, c1, c2, p)]) * self.M)
                        m2.addConstr(v2[('soc', vid, c2)] <= self.BATT_CAP +
                                   (1 - v2[('vp', vid, c1, c2, p)]) * self.M)
                    else:
                        m2.addConstr(v2[('soc', vid, c2)] >= v2[(
                            'soc', vid, c1)] - su - (1 - v2[('vp', vid, c1, c2, p)]) * self.M)
                        m2.addConstr(v2[('soc', vid, c2)] <= v2[(
                            'soc', vid, c1)] - su + (1 - v2[('vp', vid, c1, c2, p)]) * self.M)
            # if rt:
            #     m2.addConstr(v2[('soc', vid, rt[-1])] >= self.BATT_MIN)

        # Objective
        obj2 = gp.LinExpr()
        for vid, rt in routes.items():
            if not rt:
                continue
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                pl = dh.cp_paths.get((c1, c2), [])
                obj2 += gp.quicksum(
                    v2[('vp', vid, c1, c2, p)] * dh.cp_t.get((c1, c2, p), 0) *  60 for p in pl)
        for vid, rt in routes.items():
            for c in rt:
                obj2 += v2[('cs', vid, c)] * 0.1 + v2[('ce', vid, c)] * 0.2
        # Swap cost 
        for vid, rt in routes.items():
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                for p in dh.cp_paths.get((c1, c2), []):
                    is_vp = str(p).startswith('swap_')
                    if is_vp and dh.virtual_iess.get(p) in dh.IESS_NODES:
                        iess_node = dh.virtual_iess[p]
                        price = dh.ele_price.get(iess_node, 1.0)
                        obj2 += v2[('vp', vid, c1, c2, p)] * price
      
        m2.setObjective(obj2, GRB.MINIMIZE)
        m2.setParam('TimeLimit', 60)
        m2.setParam('MIPGap', 0.01)
        m2.write('model.lp')
        m2.setParam('OutputFlag', 0)
        return m2, v2

    def solve(self):
        dh = self.dh

        # Stage 1
        print("\n>>> Stage1: Route Order Optimization")
        m1, v1 = self._stage1_route_optimization()
        m1.optimize()
        s1 = self._status_name(m1)

        routes = {}
        if s1 in ('optimal', 'bestsollimit', 'gaplimit') and m1.SolCount > 0:
            print(f"  S1: {s1} obj={m1.ObjVal:.1f}")
            routes = self._extract_routes(m1, v1)
            self.NV = len(routes)
        else:
            print(f"  S1: {s1} → heuristic fallback")
            routes, self.NV = self._heuristic_routes()

        print(f"  Routes: {self.NV}")
        for vid, rt in routes.items():
            print(f"    Veh{vid}: DEPOT → "
                  f"{' → '.join(str(c) for c in rt)} → DEPOT")

        # Stage 2
        print("\n>>> Stage2: Path Selection")
        m2, v2 = self._stage2_path_selection(routes)
        m2.optimize()
        s2 = self._status_name(m2)
        if s2 in ('optimal', 'bestsollimit', 'gaplimit', 'timelimit') and m2.SolCount > 0:
            print(f"  S2: {s2} obj={m2.ObjVal:.1f}")
        else:
            print(f"  S2: {s2}")

        self._sol = {k: vv.X for k, vv in v2.items()}
        self._routes = routes
        self._m2 = m2
        self._v2 = v2
        return self._sol

    def get_swap_events(self):
        dh = self.dh
        sol = self._sol
        swap_events = []
        path_sel = {}
        for vid, rt in self._routes.items():
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                for p in dh.cp_paths.get((c1, c2), []):
                    if sol.get(('vp', vid, c1, c2, p), 0) > 0.5:
                        path_sel[(vid, c1, c2)] = p
                        break
        for vid, rt in self._routes.items():
            cs0 = self.BATT_CAP
            w = self.EMPTY + sum(dh.dmd.get(cc, 0) for cc in rt)
            t = dh.dep_t.get(rt[0], 0)
            for i, c in enumerate(rt):
                if i == 0:
                    dk = dh.dep_d.get(c, 10.0)
                else:
                    pre = rt[i - 1]
                    pid = path_sel.get((vid, pre, c))
                    if pid:
                        dk = dh.cp_d.get((pre, c, pid), dh.compute_path_distance(
                            pid if not str(pid).startswith('swap_') else int(str(pid).replace('swap_', ''))))
                    else:
                        dk = 10.0
                soc_use = dk * (self.BK + self.WK * w)
                if i > 0:
                    pre = rt[i - 1]
                    pid = path_sel.get((vid, pre, c))
                    if pid and pid in dh.iess_p and dh.piess.get(pid) in dh.IESS_NODES:
                        kw = self.BATT_CAP - (cs0 - soc_use)
                        swap_events.append(
                            (dh.piess[pid], kw, t))
                        cs0 = self.BATT_CAP
                    else:
                        cs0 = max(cs0 - soc_use, self.BATT_MIN)
                else:
                    cs0 = max(cs0 - soc_use, self.BATT_MIN)
                w -= dh.dmd.get(c, 0)
                t += dh.svc.get(c, 0)
                if i < len(rt) - 1:
                    nc = rt[i + 1]
                    pid = path_sel.get((vid, c, nc))
                    if pid:
                        travel = dh.compute_path_time_at(pid, t)
                        t += travel
        return swap_events

    def get_route_summary(self):
        dh = self.dh
        sol = self._sol
        rows = []
        path_sel = {}
        for vid, rt in self._routes.items():
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                for p in dh.cp_paths.get((c1, c2), []):
                    if sol.get(('vp', vid, c1, c2, p), 0) > 0.5:
                        path_sel[(vid, c1, c2)] = p
                        break

        for vid, rt in self._routes.items():
            for i, c in enumerate(rt):
                pre = self.DEPOT if i == 0 else rt[i - 1]
                pid = path_sel.get((vid, pre, c)) if i > 0 else 'VIR'
                is_swap = False
                ie_n = ''
                if i > 0 and pid and pid != 'VIR' and pid in dh.iess_p and dh.piess.get(pid) in dh.IESS_NODES:
                    is_swap = True
                    ie_n = str(dh.piess[pid])
                rows.append((vid, i + 1, c, pid, is_swap, ie_n))

        return pd.DataFrame(rows, columns=['车辆ID', '序号', '客户', '路径ID', '是否换电', '换电站'])

    def post_handle(self):
        dh = self.dh
        sol = self._sol
        swap_events = self.get_swap_events()
        path_sel = {}
        sw_sel = {}
        for vid, rt in self._routes.items():
            for i in range(len(rt) - 1):
                c1, c2 = rt[i], rt[i + 1]
                for p in dh.cp_paths.get((c1, c2), []):
                    if sol.get(('vp', vid, c1, c2, p), 0) > 0.5:
                        path_sel[(vid, c1, c2)] = p
                        break
                sw_sel[(vid, c1, c2)] = sol.get(
                    ('sw_flag', vid, c1, c2), 0)
        derived_wt = {}
        derived_at = {}
        derived_et = {}
        derived_soc = {}
        swap_list = []

        for vid, rt in self._routes.items():
            if not rt:
                continue
            w = self.EMPTY
            for c in reversed(rt):
                w += dh.dmd.get(c, 0)
                derived_wt[(vid, c)] = w

            cs0 = self.BATT_CAP
            t = dh.dep_t.get(rt[0], 0)
            for i, c in enumerate(rt):
                derived_at[(vid, c)] = t
                if i == 0:
                    derived_soc[(vid, c)] = max(
                        cs0 - dh.dep_d[c] * (self.BK + self.WK * derived_wt.get((vid, c), 0)), self.BATT_MIN)
                    cs0 = derived_soc[(vid, c)]
                else:
                    pre = rt[i - 1]
                    pid = path_sel.get((vid, pre, c))
                    if pid:
                        dk = dh.cp_d.get((pre, c, pid), 0)
                        wt_v = derived_wt.get((vid, pre), 0)
                        soc_use = dk * (self.BK + self.WK * wt_v)
                        if pid in dh.iess_p and dh.piess.get(pid) in dh.IESS_NODES:
                            kw = self.BATT_CAP - (cs0 - soc_use)
                            swap_list.append(
                                (vid, dh.piess[pid], t, max(cs0 - soc_use, self.BATT_MIN), self.BATT_CAP, kw))
                            derived_soc[(vid, c)] = self.BATT_CAP
                            cs0 = self.BATT_CAP
                        else:
                            derived_soc[(vid, c)] = max(
                                cs0 - soc_use, self.BATT_MIN)
                            cs0 = derived_soc[(vid, c)]
                    else:
                        derived_soc[(vid, c)] = cs0
                t += dh.svc.get(c, 0)
                derived_et[(vid, c)] = t
                if i < len(rt) - 1:
                    nc = rt[i + 1]
                    pid = path_sel.get((vid, c, nc))
                    if pid:
                        travel = dh.compute_path_time_at(pid, t)
                        t += travel

        route_rows = []
        swap_rows = []
        for vid, rt in self._routes.items():
            if not rt:
                continue
            for i, c in enumerate(rt):
                pre = self.DEPOT if i == 0 else rt[i - 1]
                pid = path_sel.get((vid, pre, c)) if i > 0 else 'VIR'

                travel_t = 0.0
                travel_d = 0.0
                if i > 0 and pid != 'VIR':
                    n1 = dh.cn.get(str(pre), 0)
                    n2 = dh.cn.get(str(c), 0)
                    _, travel_d = dh.arpy.shortest_path(
                        n1, n2, mode='distance')
                    travel_t = derived_at.get(
                        (vid, c), 0) - derived_et.get((vid, pre), 0)

                wt_ = derived_wt.get((vid, c), 0)
                at_ = derived_at.get((vid, c), 0)
                et_ = derived_et.get((vid, c), 0)
                soc_v = derived_soc.get((vid, c), 0)
                soc_out = soc_v
                sw_kwh = 0
                ie_n = ''
                if pid and pid != 'VIR' and pid in dh.iess_p and dh.piess.get(pid) in dh.IESS_NODES:
                    ie_n = str(dh.piess[pid])
                    for ev in swap_list:
                        if ev[0] == vid and ev[1] == dh.piess[pid] and abs(ev[2] - at_) < 1:
                            sw_kwh = ev[5]
                            soc_out = self.BATT_CAP
                            break
                    swap_rows.append((vid, ie_n, round(at_, 1),
                                      round(soc_out - sw_kwh, 1), round(soc_out, 1), round(sw_kwh, 1)))

                route_rows.append((vid, i + 1, c, pid, round(at_, 1), round(et_, 1),
                                   round(travel_t, 1), round(travel_d, 2), dh.dmd.get(c, 0),
                                   round(wt_, 3), round(soc_out - sw_kwh, 1), round(soc_out, 1),
                                   '1' if sw_kwh > 0 else '0', round(sw_kwh, 1), ie_n))

        self.swap_df = pd.DataFrame(swap_rows, columns=[
            '车辆ID', 'IESS节点', '换电时间min', '进站SOC', '出站SOC', '换电量kWh']) if swap_rows else pd.DataFrame()
        self.route_df = pd.DataFrame(route_rows, columns=[
            '车辆ID', '序号', '客户', '路径ID', '到达min', '离开min', '行驶min', '行驶km',
            '交付t', '载重t', '入站SOC', '出站SOC', '是否换电', '换电量kWh', '换电站'])
        self.v_df = pd.DataFrame([
            (vid, len(rt), round(derived_et.get((vid, rt[-1]), 0), 1),
             round(sum(dh.arpy.shortest_path(dh.cn.get(rt[i - 1], 0), dh.cn.get(rt[i], 0), mode='distance')[1] for i in range(1, len(rt))), 2),
             round(sum(ev[5] for ev in swap_list if ev[0] == vid), 1),
             ' → '.join(str(c) for c in rt))
            for vid, rt in self._routes.items()],
            columns=['车辆ID', '客户数', '总耗时min', '总里程km', '总换电kWh', '路线概要']
        )

        print(f"  Route: {len(self.route_df)} rows, "
              f"Swap: {len(self.swap_df)} rows, "
              f"Summary: {len(self.v_df)} rows")
        self.swap_list = swap_list
        return self.swap_list

    def export_excel(self, path='path_schedule_milp_result.xlsx'):
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
    iess_price = {4: 0.6, 3: 0.6, 10: 0.6, 9: 6.6, 18: 0.6, 42: 1.67, 36: 1.31, 71: 0.4, 23: 0.4, 62: 0.67, 73: 1.42, 20: 0.4, 11: 1.14, 46: 10.05, 28: 6.63}
    dh = DataHandler(num_customers=80, depot_node=10,iess_node=IESS_NODES, iess_cap=iess_cap,iess_price=iess_price)
    solver = TwoStageSolver(dh, time_limit=120, gap=0.05)
    solver.solve()
    solver.post_handle()
    solver.export_excel('path_schedule_milp_result.xlsx')
    print("\nDONE!")