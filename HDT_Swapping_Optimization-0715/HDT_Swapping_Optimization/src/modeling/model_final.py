# HDT_Swapping_Optimization/src/modeling/model_final.py

from pyomo.environ import *
from src.common import config_final as config
import numpy as np


def create_operational_model(data, vehicle_ids, task_ids, deactivated_constraints=None):
    """
    【第二阶段：运营执行模型 - V2修复版】
    为指定的车辆和任务子集构建精细化的路径和能源调度模型。
    """
    if deactivated_constraints is None:
        deactivated_constraints = []

    print(f"开始构建【第二阶段：运营执行】模型...")
    print(f"  -> 规划车辆: {vehicle_ids}")
    print(f"  -> 规划任务: {task_ids}")
    if deactivated_constraints:
        print(f"  -> 警告：以下约束组将被禁用 -> {deactivated_constraints}")

    model = ConcreteModel(name="Operational_VRP_Model")

    # --- 1. 集合 ---
    model.LOCATIONS = Set(initialize=data['locations'].keys())
    model.VEHICLES = Set(initialize=vehicle_ids)
    model.TASKS = Set(initialize=task_ids)
    model.CUSTOMERS = Set(initialize=[data['tasks'][t]['delivery_to'] for t in model.TASKS])
    model.STATIONS = Set(initialize=data['stations'].keys())
    model.TIME = Set(initialize=data['time_steps'])
    depot_ids = list({v['depot_id'] for v in data['vehicles'].values() if v['depot_id'] in data['locations']})
    model.DEPOT = Set(initialize=depot_ids)
    model.NODES = model.LOCATIONS - model.DEPOT
    model.BUSES = Set(initialize=data['power_buses'])
    model.LINES = Set(initialize=data['power_lines'])

    # --- 2. 决策变量 ---
    model.x = Var(model.LOCATIONS, model.LOCATIONS, model.VEHICLES, within=Reals, bounds=(0, 1))
    model.y = Var(model.LOCATIONS, model.VEHICLES, within=Reals, bounds=(0, 1))
    model.arrival_time = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.soc_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.weight_on_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.swap_decision = Var(model.STATIONS, model.VEHICLES, within=Reals, bounds=(0, 1))
    model.task_delay = Var(model.TASKS, model.VEHICLES, within=NonNegativeReals)
    model.P_grid = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.P_charge_batt = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.ev_unserved = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.N_full_batt = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.N_empty_batt = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.hdt_swap_in_t = Var(model.STATIONS, model.VEHICLES, model.TIME, within=Reals, bounds=(0, 1))
    model.P_flow = Var(model.LINES, model.TIME, within=Reals)
    model.Q_flow = Var(model.LINES, model.TIME, within=Reals)
    model.V_squared = Var(model.BUSES, model.TIME, within=NonNegativeReals,
                          bounds=(config.MIN_VOLTAGE_PU ** 2, config.MAX_VOLTAGE_PU ** 2))

    # --- 3. 目标函数 ---
    def objective_rule(m):
        travel_cost = config.MANPOWER_COST_PER_HOUR * sum(m.arrival_time[d, k] for d in m.DEPOT for k in m.VEHICLES)
        swap_cost = config.FIXED_SWAP_COST * sum(m.swap_decision[s, k] for s in m.STATIONS for k in m.VEHICLES)
        grid_cost = sum(
            m.P_grid[s, t] * config.TIME_STEP_HOURS * data['electricity_prices'][t] for s in m.STATIONS for t in m.TIME)
        delay_penalty = config.DELAY_PENALTY_PER_HOUR * sum(m.task_delay[t, k] for t in m.TASKS for k in m.VEHICLES)
        ev_penalty = config.EV_UNSERVED_PENALTY_PER_KWH * sum(
            m.ev_unserved[s, t] * config.TIME_STEP_HOURS for s in m.STATIONS for t in m.TIME)

        # 【关键修复】: 添加对二进制松弛变量的惩罚项，引导它们取 0 或 1，避免0成本解
        binary_penalty = 1e-2 * sum(
            m.x[i, j, k] * (1 - m.x[i, j, k]) for i in m.LOCATIONS for j in m.LOCATIONS for k in m.VEHICLES if i != j)

        return travel_cost + swap_cost + grid_cost + delay_penalty + ev_penalty + binary_penalty

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- 4. 约束 ---
    model.constrs = ConstraintList()

    if 'hdt_routing' not in deactivated_constraints:
        for t in model.TASKS:
            customer = data['tasks'][t]['delivery_to']
            for k in model.VEHICLES:
                model.constrs.add(
                    model.task_delay[t, k] >= model.arrival_time[customer, k] - data['tasks'][t]['due_time'])
        for t in model.TASKS:
            customer = data['tasks'][t]['delivery_to']
            model.constrs.add(sum(model.y[customer, k] for k in model.VEHICLES) == 1)
        for c in model.CUSTOMERS:
            model.constrs.add(sum(model.y[c, k] for k in model.VEHICLES) <= 1)
        for k in model.VEHICLES:
            depot_id = data['vehicles'][k]['depot_id']
            model.constrs.add(sum(model.x[depot_id, j, k] for j in model.LOCATIONS if j != depot_id) <= 1)
            model.constrs.add(sum(model.x[i, depot_id, k] for i in model.LOCATIONS if i != depot_id) == sum(
                model.x[depot_id, j, k] for j in model.LOCATIONS if j != depot_id))
            for n in model.LOCATIONS:
                if n == depot_id:
                    model.constrs.add(model.y[n, k] == sum(model.x[n, j, k] for j in model.LOCATIONS if j != n))
                else:
                    model.constrs.add(model.y[n, k] == sum(model.x[i, n, k] for i in model.LOCATIONS if i != n))
                    model.constrs.add(sum(model.x[i, n, k] for i in model.LOCATIONS if i != n) == sum(
                        model.x[n, j, k] for j in model.LOCATIONS if j != n))

    for k in model.VEHICLES:
        depot_id = data['vehicles'][k]['depot_id']
        model.constrs.add(model.arrival_time[depot_id, k] == 0)
        model.constrs.add(model.soc_arrival[depot_id, k] == data['vehicles'][k]['initial_soc'])
        model.constrs.add(model.weight_on_arrival[depot_id, k] == config.HDT_EMPTY_WEIGHT_TON)
        weight_leaving_depot = config.HDT_EMPTY_WEIGHT_TON + sum(
            data['tasks'][t]['demand'] * model.y[data['tasks'][t]['delivery_to'], k] for t in model.TASKS)
        for i in model.LOCATIONS:
            demand_at_i = sum(data['tasks'][t]['demand'] for t in model.TASKS if data['tasks'][t]['delivery_to'] == i)
            weight_depart_i = model.weight_on_arrival[i, k] - demand_at_i * model.y[i, k]
            if i == depot_id:
                weight_depart_i = weight_leaving_depot
            for j in model.LOCATIONS:
                if i != j:
                    if 'hdt_weight' not in deactivated_constraints:
                        model.constrs.add(
                            model.weight_on_arrival[j, k] >= weight_depart_i - config.BIG_M * (1 - model.x[i, j, k]))
                        model.constrs.add(
                            model.weight_on_arrival[j, k] <= weight_depart_i + config.BIG_M * (1 - model.x[i, j, k]))
                    if 'hdt_time' not in deactivated_constraints:
                        service_time = config.LOADING_UNLOADING_TIME_HOURS if i in model.CUSTOMERS else 0
                        if i in model.STATIONS: service_time += model.swap_decision[i, k] * config.SWAP_DURATION_HOURS
                        departure_time_i = model.arrival_time[i, k] + service_time
                        model.constrs.add(model.arrival_time[j, k] >= departure_time_i + data['time_matrix'].loc[
                            i, j] - config.BIG_M * (1 - model.x[i, j, k]))
                    if 'hdt_soc' not in deactivated_constraints:
                        soc_after_swap = model.soc_arrival[i, k] + (model.swap_decision[i, k] * (
                                    config.HDT_BATTERY_CAPACITY_KWH - model.soc_arrival[
                                i, k])) if i in model.STATIONS else model.soc_arrival[i, k]
                        energy_consumed = data['dist_matrix'].loc[i, j] * (
                                    config.HDT_BASE_CONSUMPTION_KWH_PER_KM + weight_depart_i * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
                        model.constrs.add(model.soc_arrival[j, k] <= soc_after_swap - energy_consumed + config.BIG_M * (
                                    1 - model.x[i, j, k]))
        if 'hdt_soc' not in deactivated_constraints:
            for n in model.NODES:
                model.constrs.add(
                    model.soc_arrival[n, k] >= config.HDT_MIN_SOC_KWH - config.BIG_M * (1 - model.y[n, k]))

    print("【第二阶段】模型构建完成。")
    return model