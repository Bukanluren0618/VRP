# HDT_Swapping_Optimization/src/modeling/model_final.py

from pyomo.environ import *
from src.common import config_final as config
import numpy as np


def create_final_model(data):
    """
    【最终修复版】: 修复了所有已知的物理约束BUG，逻辑清晰且稳健。
    """
    print("开始构建最终修复版集成优化模型...")
    model = ConcreteModel(name="Robust_Coupled_Model")

    # --- 1. 集合 ---
    model.LOCATIONS = Set(initialize=data['locations'].keys())
    model.VEHICLES = Set(initialize=data['vehicles'].keys())
    model.TASKS = Set(initialize=data['tasks'].keys())
    model.CUSTOMERS = Set(initialize=[t_info['delivery_to'] for t_info in data['tasks'].values()])
    model.STATIONS = Set(initialize=data['stations'].keys())
    model.TIME = Set(initialize=data['time_steps'])
    model.DEPOT = Set(initialize=['Depot'])
    model.NODES = model.LOCATIONS - model.DEPOT
    model.BUSES = Set(initialize=data['power_buses'])
    model.LINES = Set(initialize=data['power_lines'])

    # --- 2. 决策变量 ---
    model.x = Var(model.LOCATIONS, model.LOCATIONS, model.VEHICLES, within=Binary)
    model.y = Var(model.LOCATIONS, model.VEHICLES, within=Binary)
    model.is_task_unassigned = Var(model.TASKS, within=Binary)
    model.arrival_time = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.soc_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.weight_on_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.swap_decision = Var(model.STATIONS, model.VEHICLES, within=Binary)
    model.task_delay = Var(model.TASKS, model.VEHICLES, within=NonNegativeReals)
    model.P_grid = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.P_charge_batt = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.ev_unserved = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.N_full_batt = Var(model.STATIONS, model.TIME, within=NonNegativeIntegers)
    model.N_empty_batt = Var(model.STATIONS, model.TIME, within=NonNegativeIntegers)
    model.hdt_swap_in_t = Var(model.STATIONS, model.VEHICLES, model.TIME, within=Binary)
    model.P_flow = Var(model.LINES, model.TIME, within=Reals)
    model.Q_flow = Var(model.LINES, model.TIME, within=Reals)
    model.V_squared = Var(model.BUSES, model.TIME, within=NonNegativeReals)

    # --- 3. 目标函数 ---
    def objective_rule(m):
        travel_cost = config.MANPOWER_COST_PER_HOUR * sum(m.arrival_time[d, k] for d in m.DEPOT for k in m.VEHICLES)
        swap_cost = config.FIXED_SWAP_COST * sum(m.swap_decision[s, k] for s, k in m.swap_decision)
        grid_cost = sum(m.P_grid[s, t] * config.TIME_STEP_HOURS * data['electricity_prices'][t] for s, t in m.P_grid)
        delay_penalty = config.DELAY_PENALTY_PER_HOUR * sum(m.task_delay.values())
        ev_penalty = config.EV_UNSERVED_PENALTY_PER_KWH * sum(m.ev_unserved.values())
        unassigned_penalty = config.UNASSIGNED_TASK_PENALTY * sum(m.is_task_unassigned.values())
        return travel_cost + swap_cost + grid_cost + delay_penalty + ev_penalty + unassigned_penalty

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- 4. 约束 ---
    model.constrs = ConstraintList()

    # -- 延迟变量定义 --
    for t, t_info in data['tasks'].items():
        customer = t_info['delivery_to']
        for k in model.VEHICLES:
            model.constrs.add(model.task_delay[t, k] >= model.arrival_time[customer, k] - t_info['due_time'])

    # -- 任务与客户服务约束 --
    for t, t_info in data['tasks'].items():
        customer = t_info['delivery_to']
        model.constrs.add(sum(model.y[customer, k] for k in model.VEHICLES) + model.is_task_unassigned[t] == 1)
    for c in model.CUSTOMERS:
        model.constrs.add(sum(model.y[c, k] for k in model.VEHICLES) <= 1)

    # -- 路径与流平衡约束 --
    for k in model.VEHICLES:
        depot = next(iter(model.DEPOT))
        model.constrs.add(sum(model.x[depot, j, k] for j in model.NODES) <= 1)
        model.constrs.add(
            sum(model.x[i, depot, k] for i in model.NODES) == sum(model.x[depot, j, k] for j in model.NODES))
        for n in model.LOCATIONS:
            if n == depot:
                model.constrs.add(model.y[n, k] == sum(model.x[n, j, k] for j in model.NODES))
            else:
                model.constrs.add(model.y[n, k] == sum(model.x[i, n, k] for i in model.LOCATIONS if i != n))
                model.constrs.add(sum(model.x[i, n, k] for i in model.LOCATIONS if i != n) == sum(
                    model.x[n, j, k] for j in model.LOCATIONS if j != n))

    # -- 物理状态传播（最终修复版） --
    for k in model.VEHICLES:
        depot = next(iter(model.DEPOT))
        # 初始状态
        model.constrs.add(model.arrival_time[depot, k] == 0)
        model.constrs.add(model.soc_arrival[depot, k] == data['vehicles'][k]['initial_soc'])
        model.constrs.add(model.weight_on_arrival[depot, k] == config.HDT_EMPTY_WEIGHT_TON)

        # 离开仓库时的重量
        weight_leaving_depot = config.HDT_EMPTY_WEIGHT_TON + sum(
            t_info['demand'] * model.y[t_info['delivery_to'], k] for t, t_info in data['tasks'].items())

        for i in model.LOCATIONS:
            # 离开i时的重量
            demand_at_i = sum(t_info['demand'] for t, t_info in data['tasks'].items() if t_info['delivery_to'] == i)
            weight_depart_i = model.weight_on_arrival[i, k] - demand_at_i * model.y[i, k]
            if i == depot:
                weight_depart_i = weight_leaving_depot

            for j in model.LOCATIONS:
                if i != j:
                    # 载重传播
                    model.constrs.add(
                        model.weight_on_arrival[j, k] >= weight_depart_i - config.BIG_M * (1 - model.x[i, j, k]))
                    model.constrs.add(
                        model.weight_on_arrival[j, k] <= weight_depart_i + config.BIG_M * (1 - model.x[i, j, k]))

                    # 时间传播
                    service_time = config.LOADING_UNLOADING_TIME_HOURS if i in model.CUSTOMERS else 0
                    if i in model.STATIONS: service_time += model.swap_decision[i, k] * config.SWAP_DURATION_HOURS
                    departure_time_i = model.arrival_time[i, k] + service_time
                    model.constrs.add(
                        model.arrival_time[j, k] >= departure_time_i + data['time_matrix'].loc[i, j] - config.BIG_M * (
                                    1 - model.x[i, j, k]))

                    # 电量传播
                    soc_departure_i = model.soc_arrival[i, k] + (model.swap_decision[i, k] * (
                                config.HDT_BATTERY_CAPACITY_KWH - model.soc_arrival[
                            i, k]) if i in model.STATIONS else 0)
                    energy_consumed = data['dist_matrix'].loc[i, j] * (
                                config.HDT_BASE_CONSUMPTION_KWH_PER_KM + weight_depart_i * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
                    model.constrs.add(model.soc_arrival[j, k] <= soc_departure_i - energy_consumed + config.BIG_M * (
                                1 - model.x[i, j, k]))

        # 时间窗与最低电量约束
        for t, t_info in data['tasks'].items():
            customer = t_info['delivery_to']
            model.constrs.add(model.arrival_time[customer, k] <= t_info['due_time'] + model.task_delay[t, k])
        for n in model.NODES:
            model.constrs.add(model.soc_arrival[n, k] >= config.HDT_MIN_SOC_KWH - config.BIG_M * (1 - model.y[n, k]))

    # -- LinDistFlow潮流约束 --
    substation = data['substation_bus']
    base_mva = config.BASE_POWER_MVA
    for t in model.TIME:
        model.constrs.add(model.V_squared[substation, t] == 1.0 ** 2)
        for b in model.BUSES:
            if b != substation:
                power_in_p = sum(model.P_flow[i, j, t] for i, j in model.LINES if j == b)
                power_out_p = sum(model.P_flow[i, j, t] for i, j in model.LINES if i == b)
                load_p_pu = 0
                for station, bus_name in data['station_to_bus_map'].items():
                    if bus_name == b:
                        station_load_kw = model.P_grid[station, t]
                        load_p_pu += station_load_kw / (base_mva * 1000)
                model.constrs.add(power_in_p - power_out_p == load_p_pu)

                power_in_q = sum(model.Q_flow[i, j, t] for i, j in model.LINES if j == b)
                power_out_q = sum(model.Q_flow[i, j, t] for i, j in model.LINES if i == b)
                load_q_pu = load_p_pu * np.tan(np.arccos(0.95))
                model.constrs.add(power_in_q - power_out_q == load_q_pu)
        for i, j in model.LINES:
            r_pu, x_pu = data['line_params'][(i, j)]['r_pu'], data['line_params'][(i, j)]['x_pu']
            model.constrs.add(model.V_squared[j, t] == model.V_squared[i, t] - 2 * (
                        r_pu * model.P_flow[i, j, t] + x_pu * model.Q_flow[i, j, t]))

    # -- 电网安全运行约束 --
    v_min_sq, v_max_sq = config.MIN_VOLTAGE_PU ** 2, config.MAX_VOLTAGE_PU ** 2
    s_max_pu = config.LINE_THERMAL_LIMIT_MVA / base_mva
    for t in model.TIME:
        for b in model.BUSES:
            model.constrs.add(model.V_squared[b, t] >= v_min_sq)
            model.constrs.add(model.V_squared[b, t] <= v_max_sq)
        for i, j in model.LINES:
            model.constrs.add(model.P_flow[i, j, t] ** 2 + model.Q_flow[i, j, t] ** 2 <= s_max_pu ** 2)

    # -- 能源站平衡约束 --
    for s, k, t in model.STATIONS * model.VEHICLES * model.TIME:
        model.constrs.add(
            model.arrival_time[s, k] >= t * config.TIME_STEP_HOURS - config.BIG_M * (1 - model.hdt_swap_in_t[s, k, t]))
        model.constrs.add(model.arrival_time[s, k] <= (t + 1) * config.TIME_STEP_HOURS + config.BIG_M * (
                    1 - model.hdt_swap_in_t[s, k, t]))
    for s, k in model.STATIONS * model.VEHICLES:
        model.constrs.add(model.swap_decision[s, k] == sum(model.hdt_swap_in_t[s, k, t] for t in model.TIME))
    for s in model.STATIONS:
        for t in model.TIME:
            ev_demand_t = data['ev_demand_timestep'][s][t]
            model.constrs.add(model.P_grid[s, t] + data['pv_generation'][s][t] >= model.P_charge_batt[s, t] + (
                        ev_demand_t - model.ev_unserved[s, t]) / config.TIME_STEP_HOURS)
            newly_charged = model.P_charge_batt[
                                s, t] * config.TIME_STEP_HOURS * config.BATTERY_CHARGE_EFFICIENCY / config.HDT_BATTERY_CAPACITY_KWH
            hdt_swap_demand = sum(model.hdt_swap_in_t[s, k, t] for k in model.VEHICLES)
            if t > 0:
                model.constrs.add(
                    model.N_full_batt[s, t] == model.N_full_batt[s, t - 1] - hdt_swap_demand + newly_charged)
                model.constrs.add(
                    model.N_empty_batt[s, t] == model.N_empty_batt[s, t - 1] + hdt_swap_demand - newly_charged)
            else:
                model.constrs.add(model.N_full_batt[s, t] == data['stations'][s]['initial_full'] - hdt_swap_demand)
                model.constrs.add(model.N_empty_batt[s, t] == data['stations'][s]['initial_empty'] + hdt_swap_demand)
            model.constrs.add(hdt_swap_demand <= model.N_full_batt[s, t - 1 if t > 0 else 0])

    print("最终修复版模型构建完成。")
    return model