import numpy as np
import pandas as pd
import random
import warnings
import os
# 限制底层线性代数库线程，避免多线程库冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from collections import defaultdict
from grid_model import GridModel
warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

class GlobalScheduler:
    def __init__(self, num_customers=5, depot_node=10, max_iter=30,
                 grid_xlsx='grid_impedance_catalog_3ph118.xlsx',
                 vrp_solver='gurobi'):  # 'gurobi' or 'milp'
        self.num_cust = num_customers
        self.depot_node = depot_node
        self.max_iter = max_iter
        self.vrp_solver = vrp_solver
        print(f"  VRP Solver: {vrp_solver}")

        gm = GridModel(grid_xlsx=grid_xlsx)
        self.grid_nodes = gm.nodes

        # all_nodes = list(range(80))
        # self.IESS_NODES = random.sample(all_nodes, 5)
        # remaining = list(set(all_nodes)-set(self.IESS_NODES))
        # self.EVCS_NODES = random.sample(remaining, 10)

        # self.iess_to_bus = {}
        # iess_bus = random.sample(self.grid_nodes, len(self.IESS_NODES))
        # for i, iess in enumerate(self.IESS_NODES):
        #     self.iess_to_bus[iess] = iess_bus[i]
        # self.evcs_to_bus = {}
        # remaining = list(set(self.grid_nodes)-set(iess_bus))
        # evcs_bus =random.sample(remaining, len(self.EVCS_NODES))
        # for i, evcs in enumerate(self.EVCS_NODES):
        #     self.evcs_to_bus[evcs] = evcs_bus[i]

        self.IESS_NODES = [4, 3, 10, 9, 18]
        self.EVCS_NODES = [42, 36, 71, 23, 62, 73, 20, 11, 46, 28]

        # 35 kV grid-bus mapping from the same preview table.
        self.iess_to_bus = {
            4: 0,
            3: 1,
            10: 2,
            9: 3,
            18: 4,
        }
        self.evcs_to_bus = {
            42: 5,
            36: 6,
            71: 7,
            23: 8,
            62: 9,
            73: 10,
            20: 11,
            11: 12,
            46: 13,
            28: 14,
        }

            
        self.bus_to_station = {v:k for k, v in self.iess_to_bus.items()}
        self.bus_to_station.update({v:k for k, v in self.evcs_to_bus.items()})

        # self.iess_cap = {iess: np.ceil(random.random()*10) for iess in self.IESS_NODES}
        # self.evcs_cap = {evcs: np.ceil(random.random()*10) for evcs in self.EVCS_NODES}
        self.iess_cap = {iess: 2 for iess in self.IESS_NODES}
        self.evcs_cap = {evcs: 10 for evcs in self.EVCS_NODES}
        
        # self.ele_prices = {**{iess: round(random.random()*100,2) for iess in self.IESS_NODES},
        #                    **{evcs: round(random.random()*100,2) for evcs in self.EVCS_NODES}}

        self.iess_price = 0.60
        self.evcs_price = 0.40
        self.electricity_cost_weight = 1.0
        self.slack_price_step = 0.10
        self.bpr_cost_weight = 1.0
        self.wait_penalty = 5.0

        self.electricity_cost_weight = 1.0
        self.slack_price_step = 0.10
        self.ele_prices = {**{iess: self.iess_price for iess in self.IESS_NODES},
                           **{evcs: self.evcs_price for evcs in self.EVCS_NODES}}

        print(f"  IESS: {self.IESS_NODES}")
        print(f"  EVCS: {self.EVCS_NODES}")
        print(f"  Prices: {dict(list(self.ele_prices.items()))}")

    def run(self):
        for it in range(self.max_iter):
            print(f"\n{'='*50}\nITER {it+1}/{self.max_iter}\n{'='*50}")

            # Step 1: BPR
            print(">>> BPR")
            from brf_gurobi import BrfDataHandler, BrfSolver
            dh_brf = BrfDataHandler(iess_nodes=self.IESS_NODES, 
                                    evcs_nodes=self.EVCS_NODES,
                                    ele_price=self.ele_prices,
                                    elc_vol={**self.iess_cap,**self.evcs_cap},
                                    ev_wait_penalty=self.wait_penalty,
                                    bpr_cost_weight=self.bpr_cost_weight,
                                    electricity_cost_weight=self.electricity_cost_weight)
            solver_brf = BrfSolver(dh_brf)
            solver_brf.solve()
            solver_brf.post_handle(solver_brf.extract_results())
            solver_brf.export_excel('algo_bpr_res.xlsx')
            print("  Done")

            # Step 2: VRP
            print(">>> VRP")
            iess_prices = {k:v for k,v in self.ele_prices.items() if k in self.IESS_NODES}
            iess_caps = {k:v for k,v in self.iess_cap.items() if k in self.IESS_NODES}

            if self.vrp_solver == 'milp':
                # ---- Two-Stage MILP (分组求解) ----
                from path_schedule_milp import (
                    DataHandler as MilpDataHandler,
                    GroupedSolver as MilpGroupedSolver,
                )
                dh_vrp = MilpDataHandler(
                    num_customers=self.num_cust,
                    depot_node=self.depot_node,
                    iess_node=self.IESS_NODES,
                    iess_price=iess_prices,
                    iess_cap=iess_caps,
                )
                solver = MilpGroupedSolver(
                    dh_vrp,
                    group_size=15,
                    time_limit=120,
                    gap=0.05,
                )
            elif self.vrp_solver == 'gurobi':
                # ---- Gurobi VRP (分组求解) ----
                from path_schedule_gurobi import (
                    DataHandler as GurobiDataHandler,
                    GroupedSolver as GurobiGroupedSolver,
                )
                dh_vrp = GurobiDataHandler(
                    num_customers=self.num_cust,
                    depot_node=self.depot_node,
                    iess_node=self.IESS_NODES,
                    iess_price=iess_prices,
                    iess_cap=iess_caps,
                )
                solver = GurobiGroupedSolver(
                    dh_vrp,
                    group_size=15,
                    time_limit=120,
                    gap=0.05,
                )
            else:
                raise ValueError(
                    "vrp_solver 只能设置为 'gurobi' 或 'milp'"
                )

            solver.solve()
            solver.export_excel('path_schedule_result.xlsx')
            print(f"  Done. Swaps:{len(solver.swap_df)}")
            swap_df = solver.swap_df.copy()



            # Step 3: Grid
            print(">>> Grid")
            # grid是每15分钟，算一个点，需要进行转换
            try:
                swap_df = pd.read_excel('path_schedule_result.xlsx',sheet_name='3_换电站信息')
                swap_df['grid_time'] = swap_df['换电时间min'].apply(lambda x: np.floor(float(x) / 15))
                swap_df['换电量kWh'] = swap_df['换电量kWh'].astype(float)
                swap_info_dict = swap_df.groupby(['grid_time','IESS节点']).apply(lambda x: x['换电量kWh'].sum()).to_dict()
            except:
                print("  ERROR: path_schedule_result.xlsx not found or invalid format")
                swap_info_dict = {}
            
            try:
                elc_df=pd.read_excel('algo_bpr_res.xlsx','elc_res')
                elc_df['grid_time'] = elc_df['time'].apply(lambda x: np.floor(float(x) / 15))
                elc_df['swap_qty'] = elc_df['swap_qty'].astype(float)
                elc_info_dict = elc_df.groupby(['grid_time','elc_node']).apply(lambda x: x['swap_qty'].sum()).to_dict()
            except:
                print("  ERROR: algo_bpr_res.xlsx not found or invalid format")
                elc_info_dict = {}
            
            #在电网节点汇总总电量需求
            total_swap_dict ={}
            for t, n in swap_info_dict:
                total_swap_dict[int(t), n] = total_swap_dict.get((int(t), n), 0) + swap_info_dict[t,n]
            for t, n in elc_info_dict:
                total_swap_dict[int(t), n] = total_swap_dict.get((int(t), n), 0) + elc_info_dict[t,n]

            gm=GridModel(n_time=96,n_nodes_pre=100)
            res=gm.build_and_solve(total_swap_dict,
                                   self.iess_to_bus,self.evcs_to_bus)
            # Post-handle: export bus voltage data
            df_vol = gm.post_handle()
            if df_vol is not None:
                with pd.ExcelWriter('grid_bus_voltage.xlsx', engine='openpyxl') as w:
                    df_vol.to_excel(w, sheet_name='bus_voltage', index=False)
            st=res['status']
            sval=res.get('slack_vals',{})
            max_slack = max([v for v in sval.values() if v is not None], default=0)
            print(f"  Grid:{st} max_slack={max_slack:.6f}")

            if max_slack < 1e-4 and st != 'infeasible':
                print("  CONVERGED: Grid meets all demands ")
                break
            updated = 0
            node_max_price = {}
            for k, v in sval.items():
                if v is None or v < 1e-4: continue
                node_id = self.bus_to_station[k[-1]] #转换回路网
                node_max_price[node_id] = max(node_max_price.get(node_id,0), v)
            for node_id,v in node_max_price.items():
                price_old = self.ele_prices.get(node_id, 1.0)
                price_new =  price_old + v * self.slack_price_step  # slack  × factor
                self.ele_prices[node_id] = round(price_new, 2)
                updated += 1
                print(f"    Node {node_id} slack={v:.4f} price: {price_old}→{price_new:.2f}")

            print(f"  Updated {updated} prices, iter continue...")

        print(f"\n{'='*50}\nFINAL\n{'='*50}")
        print(f"  Iters:{it+1} Prices:{dict(list(self.ele_prices.items()))}")
        print("DONE")

if __name__=='__main__':
    gs=GlobalScheduler(num_customers=80, depot_node=10, max_iter=10,
                       vrp_solver='gurobi')
    gs.run()