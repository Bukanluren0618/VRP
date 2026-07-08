import numpy as np
import pandas as pd
import random
import pyscipopt as scip
import networkx as nx
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

class GridModel:
    def __init__(self, grid_xlsx = 'grid_impedance_catalog_3ph118.xlsx',
                 s_base = 0.816, n_time = 6, n_nodes_pre = 20):
        raw  =  pd.read_excel(grid_xlsx, sheet_name = None)
        self.bus_df  =  raw['gridcat_buses']
        self.line_df  =  raw['gridcat_ldf']
        self.load_df  =  raw['gridcat_ts_load_long']
        self.vlimits_df  =  raw['gridcat_vlimits_pu']
        self.S_BASE  =  s_base
        self.T_STEPS  =  n_time
        self.N_NODES_PRE  =  n_nodes_pre
        self.PHASES = ['a','b','c']
        self.N_PH = len(self.PHASES)
        self.PH_MAP = {'a':0,'b':1,'c':2}
        self.ALPHA = [1.0 + 0.0j,np.exp(1j*2*np.pi/3),np.exp(1j*4*np.pi/3)]

        all_nodes = sorted(self.bus_df['nidou'].tolist())[:self.N_NODES_PRE]
        self.line_df = self.line_df[(self.line_df['from_bus'].isin(all_nodes))|
                                   (self.line_df['to_bus'].isin(all_nodes))]
        nodes = sorted(set(self.line_df['from_bus'].tolist() + 
                         self.line_df['to_bus'].tolist()))
        self.nodes = nodes
        self.node_idx = {n:i for i,n in enumerate(self.nodes)}
        self.N_NODES = len(self.nodes)
        self.vmin = dict(zip(self.vlimits_df['bus'],self.vlimits_df['vmin_pu']))
        self.vmax = dict(zip(self.vlimits_df['bus'],self.vlimits_df['vmax_pu']))
        self.branches = {}
        for row in self.line_df.itertuples():
            f,t,r,x = row.from_bus,row.to_bus,row.r_pu,row.x_pu
            zd = {}
            for pf in self.PH_MAP:
                for pt in self.PH_MAP:
                    zd[pf,pt] = (r,x) if pf == pt else (0,0)
            self.branches[f,t] = zd
        self._build_RbXb()
        self._process_loads()

    def _build_RbXb(self):
        G = nx.DiGraph()
        for(u,v) in self.branches:G.add_edge(u,v)
        Gu = G.to_undirected()
        node_paths = {}
        for node in self.nodes:
            if node == 0:
                node_paths[node] = []
            else:
                try:
                    pn = nx.shortest_path(Gu,source = 0,target = node)
                    pb = []
                    for k in range(len(pn)-1):
                        up,vp = pn[k],pn[k + 1]
                        pb.append((up,vp) if (up,vp) in self.branches else (vp,up))
                    node_paths[node] = pb
                except: node_paths[node] = []
        self.SIZE = self.N_NODES*self.N_PH
        self.Rb = np.zeros((self.SIZE,self.SIZE))
        self.Xb = np.zeros((self.SIZE,self.SIZE))
        for r in range(self.SIZE):
            i = r//self.N_PH
            i_node = self.nodes[i]
            phi_i = self.PH_MAP[self.PHASES[r%self.N_PH]]
            for c in range(self.SIZE):
                j = c//self.N_PH
                j_node = self.nodes[j]
                psi_i = self.PH_MAP[self.PHASES[c%self.N_PH]]
                cb = list(set(node_paths[i_node])&set(node_paths[j_node]))
                if not cb: continue
                a = self.ALPHA[(phi_i-psi_i)%3]
                sr,si = 0.0,0.0
                for(h,k) in cb:
                    zk = (h,k) if (h,k) in self.branches else (k,h)
                    zR,zX = self.branches[zk][(self.PHASES[r%self.N_PH],self.PHASES[c%self.N_PH])]
                    term = a*(zR-1j*zX)
                    sr  += term.real
                    si  += term.imag
                self.Rb[r,c] = 2*sr
                self.Xb[r,c] = -2*si
        self.E0 = 1.0
        self.E = np.ones(self.SIZE)*self.E0
        self.diagE = np.diag(np.ones(self.SIZE)*self.E0)

    def _process_loads(self):
        ld = self.load_df[self.load_df['bus_id'].isin(self.nodes)].copy()
        ld['pi'] = ld['phase'].map(self.PH_MAP)
        ld['idx'] = ld.apply(lambda x:self.node_idx[x['bus_id']]*self.N_PH + x['pi'],axis = 1)
        ld = ld[ld['time']<self.T_STEPS]
        self.T = sorted(ld['time'].unique().tolist())
        self.T  =  [int(t) for t in self.T]
        vals = []
        for(t,idx),mod in ld.groupby(['time','idx']):
            vals.append((t,idx,mod['p_mw'].sum(),mod['q_mvar'].sum()))
        agg = pd.DataFrame(vals,columns = ['time','idx','p_mw','q_mvar'])
        self.p_load = {t:np.zeros(self.SIZE) for t in self.T}
        self.q_load = {t:np.zeros(self.SIZE) for t in self.T}
        for t,mod in agg.groupby('time'):
            for _,row in mod.iterrows():
                self.p_load[t][int(row['idx'])] = -row['p_mw']/self.S_BASE
                self.q_load[t][int(row['idx'])] = -row['q_mvar']/self.S_BASE

    def build_and_solve(self, total_swap_dict,
                         iess_to_bus, evcs_to_bus):
        SWAP_PENALTY  =  1e6
        m  =  scip.Model("Grid_Coupled")
        vv = {}

        for t, n in total_swap_dict:
            if n in iess_to_bus:
                g_n  =  iess_to_bus[n]
            else:
                g_n  =  evcs_to_bus[n]
            for ph_i,ph in enumerate(self.PHASES):
                idx = self.node_idx[g_n]*self.N_PH + ph_i
                vv[('sl',t,idx)] = m.addVar(vtype = 'C',lb = 0) #未满足量


        # p,q,v
        for t in self.T:
            for idx in range(self.SIZE):
                b = self.nodes[idx//self.N_PH]
                vv[('v',t,idx)] = m.addVar(vtype = 'C',lb = self.vmin.get(b,0.9),ub = self.vmax.get(b,1.1))
                vv[('p',t,idx)] = m.addVar(vtype = 'C',lb = -np.inf)
                vv[('q',t,idx)] = m.addVar(vtype = 'C',lb = -np.inf)

        all_buses = set(list(iess_to_bus.values()) + list(evcs_to_bus.values()))

        for t in self.T:
            for idx in range(self.SIZE):
                rhs = 0.0
                for c in range(self.SIZE):
                    rhs  += self.Rb[idx,c]*vv[('p',t,c)]  +  self.Xb[idx,c]*vv[('q',t,c)]
                rhs  += self.diagE[idx,idx]*self.E[idx]
                m.addCons(vv[('v',t,idx)] == rhs)

        for t in self.T:
            for idx in range(self.SIZE):
                b = self.nodes[idx//self.N_PH]
                if b in all_buses: continue
                pl = self.p_load[t][idx]
                ql = self.q_load[t][idx]
                if pl >= 0:
                    m.addCons(vv[('p',t,idx)] <= pl)
                    m.addCons(vv[('p',t,idx)] >= 0)
                else:
                    m.addCons(vv[('p',t,idx)] >= pl)
                    m.addCons(vv[('p',t,idx)] <= 0)
                if ql >= 0:
                    m.addCons(vv[('q',t,idx)] <= ql)
                    m.addCons(vv[('q',t,idx)] >= 0)
                else:
                    m.addCons(vv[('q',t,idx)] >= ql)
                    m.addCons(vv[('q',t,idx)] <= 0)

        # IESS & EVCS buses 
        for t in self.T:
            for iess,bid in iess_to_bus.items():
                for ph_i,ph in enumerate(self.PHASES):
                    idx = self.node_idx[bid]*self.N_PH + ph_i
                    pl = self.p_load[t][idx]
                    ql = self.q_load[t][idx]
                    if (t, iess) in total_swap_dict:
                        m.addCons(vv[('p',t,idx)] == pl - total_swap_dict[t,iess]/1000./self.N_PH / 0.25 +  vv.get(('sl',t,idx),0))
                        m.addCons(vv[('q',t,idx)] == ql)
                    else:
                        m.addCons(vv[('p',t,idx)] == pl)
                        m.addCons(vv[('q',t,idx)] == ql)

            for evcs,bid in evcs_to_bus.items():
                for ph_i,ph in enumerate(self.PHASES):
                    idx = self.node_idx[bid] * self.N_PH  +  ph_i
                    pl = self.p_load[t][idx]
                    ql = self.q_load[t][idx]
                    if (t, evcs) in total_swap_dict:
                        m.addCons(vv[('p',t,idx)] == pl - total_swap_dict[t,evcs]/1000./self.N_PH / 0.25  +  vv.get(('sl',t,idx),0))
                        m.addCons(vv[('q',t,idx)] == ql)
                    else:
                        m.addCons(vv[('p',t,idx)] == pl)
                        m.addCons(vv[('q',t,idx)] == ql)

        obj = scip.Expr()
        for t in self.T:
            for idx in range(self.SIZE):
                if self.p_load[t][idx] > 0:
                    obj  +=  self.p_load[t][idx] - vv[('p',t,idx)]
                else:
                    obj  +=  vv[('p',t,idx)] - self.p_load[t][idx]
                if self.q_load[t][idx] > 0:
                    obj  +=  self.q_load[t][idx] - vv[('q',t,idx)]
                else:
                    obj  +=  vv[('q',t,idx)] - self.q_load[t][idx]
                
        for k in list(vv.keys()):
            if k[0].startswith('sl'): 
                obj  +=  vv[k]*SWAP_PENALTY

        m.setObjective(obj,'minimize')
        m.setRealParam('limits/time',300)
        # m.writeProblem('D://model.lp')
        m.optimize()
        st = m.getStatus()

        self._vv  =  vv
        self._model  =  m

        slack_vals = {}
        for k in list(vv.keys()):
            if k[0].startswith('sl'):
                try:
                    rd = self.nodes [int(k[-1] / 3)]
                    slack_vals[rd] = round(m.getVal(vv[k]),6)
                except:
                    pass
        return {'status':st,'slack_vals':slack_vals}

    def post_handle(self):
        if not hasattr(self, '_vv'):
            print("  WARNING: Run build_and_solve() first")
            return None

        vv  =  self._vv
        node_dfs  =  []
        for t in self.T:
            node_values  =  []
            for i, bus_id in enumerate(self.nodes):
                for ph_idx, ph in enumerate(self.PHASES):
                    idx = i * self.N_PH  +  ph_idx
                    try:
                        v_sq = self._model.getVal(vv[('v', t, idx)])
                        p = self._model.getVal(vv[('p', t, idx)])
                        q = self._model.getVal(vv[('q', t, idx)])
                        node_values.append((
                            t, int(bus_id), ph,
                            round(np.sqrt(max(v_sq, 0)), 4),
                            round(p * self.S_BASE * 1000, 3),   # pu → kW
                            round(q * self.S_BASE * 1000, 3),
                        ))
                    except Exception:
                        node_values.append((t, int(bus_id), ph, 0.0, 0.0, 0.0))
            df  =  pd.DataFrame(node_values,
                columns = ['time', 'bus_id', 'phase', 'v_pu', 'p_kw', 'q_kvar'])
            node_dfs.append(df)

        self.bus_node_vol_dfs  =  pd.concat(node_dfs, ignore_index = True)
        self.bus_node_vol_dfs['vmin_pu']  =  self.bus_node_vol_dfs['bus_id'].map(
            lambda b: self.vmin.get(b, 0.9))
        self.bus_node_vol_dfs['vmax_pu']  =  self.bus_node_vol_dfs['bus_id'].map(
            lambda b: self.vmax.get(b, 1.1))
        self.bus_node_vol_dfs['voltage_violation']  =  (
            (self.bus_node_vol_dfs['v_pu'] < self.bus_node_vol_dfs['vmin_pu']) |
            (self.bus_node_vol_dfs['v_pu'] > self.bus_node_vol_dfs['vmax_pu'])
        ).astype(int)

        print(f"  bus_node_vol_dfs: {len(self.bus_node_vol_dfs)} rows, "
              f"violations: {self.bus_node_vol_dfs['voltage_violation'].sum()}")
        return self.bus_node_vol_dfs
