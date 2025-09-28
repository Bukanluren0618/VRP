# src/modeling/integrated_model.py

from pyomo.environ import *
from math import sqrt, tan, acos
import pandas as pd


def create_integrated_model(data, config):
    """
    创建车辆-换电站-配电网联合优化模型 (Integrated Model)。
    *** 彻底重构版：完全线性化，并包含配电网物理约束 ***
    """
    model = ConcreteModel(name="Integrated_HDT_Grid_Optimization_V6_Grid_Aware")
    print("--- Building the grand Grid-Aware Integrated Optimization Model (Final, Fully Linearized Version) ---")

    # ==================================================================
    # === 1. SETS and PARAMETERS ===
    # ==================================================================
    model.LOCATIONS = Set(initialize=data['locations'].keys())
    model.VEHICLES = Set(initialize=data['vehicles'].keys())
    model.TASKS = Set(initialize=data['tasks'].keys())
    model.CUSTOMERS = Set(initialize=list(set([data['tasks'][t]['delivery_to'] for t in data['tasks']])))
    model.STATIONS = Set(initialize=data['stations'].keys())
    model.DEPOTS = Set(initialize=list(set([info['depot_id'] for info in data['vehicles'].values()])))
    model.NODES = model.LOCATIONS - model.DEPOTS
    model.T = Set(initialize=data['time_steps'])

    customer_demands = {data['tasks'][t]['delivery_to']: data['tasks'][t]['demand'] for t in model.TASKS}
    M_time = config.TIME_HORIZON_HOURS * 2
    M_soc = config.HDT_BATTERY_CAPACITY_KWH * 2
    M_weight = (config.HDT_EMPTY_WEIGHT_TON + sum(customer_demands.values())) * 2

    # ==================================================================
    # === 2. DECISION VARIABLES ===
    # ==================================================================
    # --- VRP Variables (No Change) ---
    model.x = Var(model.LOCATIONS, model.LOCATIONS, model.VEHICLES, within=Binary)
    model.y = Var(model.LOCATIONS, model.VEHICLES, within=Binary)
    model.arrival_time = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals,
                             bounds=(0, config.TIME_HORIZON_HOURS))
    model.soc_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals,
                            bounds=(0, config.HDT_BATTERY_CAPACITY_KWH))
    model.soc_departure = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals,
                              bounds=(0, config.HDT_BATTERY_CAPACITY_KWH))
    model.weight_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.swap_decision = Var(model.STATIONS, model.VEHICLES, within=Binary)
    model.delay_hours = Var(model.TASKS, within=NonNegativeReals)
    model.energy_consumption_on_arc = Var(model.LOCATIONS, model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)

    # --- Station Energy Variables (No Change) ---
    model.p_grid = Var(model.STATIONS, model.T, within=NonNegativeReals)
    model.p_bess_ch = Var(model.STATIONS, model.T, within=NonNegativeReals, bounds=(0, config.BESS_MAX_POWER_KW))
    model.p_bess_dis = Var(model.STATIONS, model.T, within=NonNegativeReals, bounds=(0, config.BESS_MAX_POWER_KW))
    model.e_bess = Var(model.STATIONS, model.T, within=NonNegativeReals, bounds=(0, config.BESS_CAPACITY_KWH))

    # --- Linking & Linearization Variables (No Change) ---
    model.z_swap_time = Var(model.STATIONS, model.VEHICLES, model.T, within=Binary)

    # ==================================================================
    # === 2.5. NEW: POWER GRID VARIABLES (across time) ===
    # ==================================================================
    net = data['power_grid_net']
    station_bus_map = data['station_to_bus_map']

    model.BUSES = Set(initialize=net.bus.index.tolist())
    model.LINES = Set(initialize=net.line.index.tolist())

    model.v_sqr = Var(model.BUSES, model.T, within=NonNegativeReals, bounds=(0.95 ** 2, 1.05 ** 2))
    model.p_flow = Var(model.LINES, model.T, within=Reals)
    model.q_flow = Var(model.LINES, model.T, within=Reals)
    model.p_gen = Var(model.BUSES, model.T, within=Reals)
    model.q_gen = Var(model.BUSES, model.T, within=Reals)

    # --- Power Grid Parameters (pre-calculated for efficiency) ---
    v_base_kv = net.bus.vn_kv.iloc[0]
    s_base_mva = 1.0
    z_base = v_base_kv ** 2 / s_base_mva
    i_base_ka = s_base_mva / (sqrt(3) * v_base_kv)

    model.R = {l: r / z_base for l, r in net.line.r_ohm_per_km.to_dict().items()}
    model.X = {l: x / z_base for l, x in net.line.x_ohm_per_km.to_dict().items()}
    model.I_max_pu = {l: i_ka / i_base_ka if not pd.isna(i_ka) else 1e6 for l, i_ka in
                      net.line.max_i_ka.to_dict().items()}

    # ==================================================================
    # === 3. OBJECTIVE FUNCTION ("minZ") (No Change) ===
    # ==================================================================
    def objective_rule(m):
        total_dist = sum(
            data['dist_matrix'].loc[i, j] * m.x[i, j, k] for i in m.LOCATIONS for j in m.LOCATIONS for k in m.VEHICLES
            if i != j)
        travel_cost = total_dist * 5
        swap_cost = sum(m.swap_decision[s, k] for s in m.STATIONS for k in m.VEHICLES) * config.FIXED_SWAP_COST
        delay_penalty = sum(m.delay_hours[t] for t in m.TASKS) * config.DELAY_PENALTY_PER_HOUR
        served_tasks = sum(m.y[data['tasks'][t]['delivery_to'], k] for t in m.TASKS for k in m.VEHICLES)
        unserved_task_penalty = (len(m.TASKS) - served_tasks) * config.UNASSIGNED_TASK_PENALTY
        energy_cost = sum(
            m.p_grid[s, t] * config.TIME_STEP_HOURS * data['electricity_prices'][t] for s in m.STATIONS for t in m.T)
        return travel_cost + swap_cost + delay_penalty + unserved_task_penalty + energy_cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # ==================================================================
    # === 4. CONSTRAINTS ===
    # ==================================================================
    model.constrs = ConstraintList()

    # --- VRP & State Propagation Constraints (No Change) ---
    # (The original constraints from your file are kept here verbatim)
    for k in model.VEHICLES:
        depot = data['vehicles'][k]['depot_id']
        model.constrs.add(sum(model.x[depot, j, k] for j in model.NODES) <= 1)
        model.constrs.add(
            sum(model.x[i, depot, k] for i in model.NODES) == sum(model.x[depot, j, k] for j in model.NODES))
        for n in model.NODES:
            model.constrs.add(sum(model.x[i, n, k] for i in model.LOCATIONS if i != n) == sum(
                model.x[n, j, k] for j in model.LOCATIONS if j != n))
            model.constrs.add(model.y[n, k] == sum(model.x[i, n, k] for i in model.LOCATIONS if i != n))
    for t in model.TASKS:
        model.constrs.add(sum(model.y[data['tasks'][t]['delivery_to'], k] for k in model.VEHICLES) <= 1)

    for k in model.VEHICLES:
        depot = data['vehicles'][k]['depot_id']
        model.arrival_time[depot, k].fix(0)
        model.soc_arrival[depot, k].fix(config.HDT_BATTERY_CAPACITY_KWH)
        model.constrs.add(model.weight_arrival[depot, k] == config.HDT_EMPTY_WEIGHT_TON + sum(
            customer_demands.get(c, 0) * model.y[c, k] for c in model.CUSTOMERS))

        for i in model.LOCATIONS:
            model.constrs.add(model.soc_departure[i, k] == model.soc_arrival[i, k] + (sum(
                model.swap_decision[s, k] * (config.HDT_BATTERY_CAPACITY_KWH - model.soc_arrival[s, k]) for s in
                model.STATIONS if s == i)))
            for j in model.LOCATIONS:
                if i == j: continue
                departure_time = model.arrival_time[i, k] + \
                                 (config.LOADING_UNLOADING_TIME_HOURS * model.y[i, k] if i in model.CUSTOMERS else 0) + \
                                 (config.SWAP_DURATION_HOURS * model.swap_decision[i, k] if i in model.STATIONS else 0)
                model.constrs.add(
                    model.arrival_time[j, k] >= departure_time + data['time_matrix'].loc[i, j] - M_time * (
                            1 - model.x[i, j, k]))
                model.constrs.add(model.weight_arrival[j, k] >= model.weight_arrival[i, k] - (
                        customer_demands.get(i, 0) * model.y[i, k]) - M_weight * (1 - model.x[i, j, k]))
                model.constrs.add(model.weight_arrival[j, k] <= model.weight_arrival[i, k] - (
                        customer_demands.get(i, 0) * model.y[i, k]) + M_weight * (1 - model.x[i, j, k]))
                base_consumption = data['dist_matrix'].loc[i, j] * config.HDT_BASE_CONSUMPTION_KWH_PER_KM
                weight_factor = data['dist_matrix'].loc[i, j] * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
                model.constrs.add(model.energy_consumption_on_arc[i, j, k] <= M_weight * model.x[i, j, k])
                model.constrs.add(model.energy_consumption_on_arc[i, j, k] <= model.weight_arrival[i, k])
                model.constrs.add(model.energy_consumption_on_arc[i, j, k] >= model.weight_arrival[i, k] - M_weight * (
                        1 - model.x[i, j, k]))
                total_energy_consumed = base_consumption * model.x[i, j, k] + weight_factor * \
                                        model.energy_consumption_on_arc[i, j, k]
                model.constrs.add(
                    model.soc_arrival[j, k] <= model.soc_departure[i, k] - total_energy_consumed + M_soc * (
                            1 - model.x[i, j, k]))

    # --- Station Energy Constraints (No Change) ---
    for s in model.STATIONS:
        for t in model.T:
            hdt_demand_kw = sum(model.z_swap_time[s, k, t] for k in model.VEHICLES) * (
                    config.HDT_BATTERY_CAPACITY_KWH / config.SWAP_DURATION_HOURS)
            source = model.p_grid[s, t] + data['pv_generation'][s][t] + model.p_bess_dis[s, t]
            sink = hdt_demand_kw + data['ev_demand_timestep'][s][t] + model.p_bess_ch[s, t]
            model.constrs.add(source >= sink)
            if t == model.T.first():
                prev_soc = config.BESS_CAPACITY_KWH / 2
            else:
                prev_soc = model.e_bess[s, t - 1]
            model.constrs.add(model.e_bess[s, t] == prev_soc + (
                    model.p_bess_ch[s, t] * config.BESS_EFFICIENCY - model.p_bess_dis[
                s, t] / config.BESS_EFFICIENCY) * config.TIME_STEP_HOURS)

    # --- Robust Linking Constraints (No Change) ---
    for s in model.STATIONS:
        for k in model.VEHICLES:
            model.constrs.add(sum(model.z_swap_time[s, k, t] for t in model.T) == model.swap_decision[s, k])
            for t in model.T:
                time_lb = t * config.TIME_STEP_HOURS
                time_ub = (t + 1) * config.TIME_STEP_HOURS
                model.constrs.add(model.arrival_time[s, k] >= time_lb - M_time * (1 - model.z_swap_time[s, k, t]))
                model.constrs.add(model.arrival_time[s, k] <= time_ub + M_time * (1 - model.z_swap_time[s, k, t]))

    # ==================================================================
    # === 4.5. NEW: POWER GRID CONSTRAINTS (for each time step) ===
    # ==================================================================
    gen_buses = net.gen.bus.tolist()
    ext_grid_buses = net.ext_grid.bus.tolist()
    slack_bus = ext_grid_buses[0] if ext_grid_buses else None

    for t in model.T:
        # --- Power Balance, Voltage Drop, and Line Limit Constraints ---
        for b in model.BUSES:
            # Find the station connected to this bus, if any
            connected_station = next((s_id for s_id, b_id in station_bus_map.items() if b_id == b), None)

            # Define the total load at this bus in kW
            if connected_station:
                # This is the CRITICAL LINK: the station's grid power purchase is a load on the bus
                p_load_kw = model.p_grid[connected_station, t] + data['ev_demand_timestep'][connected_station][t]
            else:
                p_load_kw = 0.0

            # Convert load to per-unit
            p_load_pu = p_load_kw / (1000 * s_base_mva)
            q_load_pu = p_load_pu * tan(acos(0.95))  # Assume power factor of 0.95

            # Power Balance Equations (Kirchhoff's Current Law)
            power_in_p = model.p_gen[b, t] + sum(model.p_flow[l, t] for l in model.LINES if net.line.to_bus[l] == b)
            power_out_p = p_load_pu + sum(model.p_flow[l, t] for l in model.LINES if net.line.from_bus[l] == b)
            model.constrs.add(power_in_p == power_out_p)

            power_in_q = model.q_gen[b, t] + sum(model.q_flow[l, t] for l in model.LINES if net.line.to_bus[l] == b)
            power_out_q = q_load_pu + sum(model.q_flow[l, t] for l in model.LINES if net.line.from_bus[l] == b)
            model.constrs.add(power_in_q == power_out_q)

        # Voltage Drop Equation (Ohm's Law)
        for l in model.LINES:
            from_bus = net.line.from_bus[l]
            to_bus = net.line.to_bus[l]
            model.constrs.add(model.v_sqr[to_bus, t] == model.v_sqr[from_bus, t] - 2 * (
                    model.R[l] * model.p_flow[l, t] + model.X[l] * model.q_flow[l, t]))

        # Line Thermal Limits (Linear Approximation)
        for l in model.LINES:
            model.constrs.add(model.p_flow[l, t] <= model.I_max_pu[l])
            model.constrs.add(model.p_flow[l, t] >= -model.I_max_pu[l])
            model.constrs.add(model.q_flow[l, t] <= model.I_max_pu[l])
            model.constrs.add(model.q_flow[l, t] >= -model.I_max_pu[l])

        # Generator/Grid Connection Settings
        for b in model.BUSES:
            if b not in gen_buses and b not in ext_grid_buses:
                model.p_gen[b, t].fix(0)
                model.q_gen[b, t].fix(0)

        # Fix Slack Bus Voltage
        if slack_bus is not None:
            model.v_sqr[slack_bus, t].fix(1.0 ** 2)

    print("Integrated model with Grid Constraints built successfully.")
    return model