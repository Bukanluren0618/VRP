# HDT_Swapping_Optimization/src/modeling/model_final.py

from pyomo.environ import *
from src.common import config_final as config


def create_final_model(data):
    """
    创建最终版的、经过重构的、保证可行的物流-能源协同优化模型。
    【最终修复版】: 修复了载重、电量和电池库存传播的底层逻辑BUG。
    """
    print("开始构建最终交付版集成优化模型...")
    model = ConcreteModel(name="Robust_HDT_Model")

    # --- 1. 集合 ---
    model.LOCATIONS = Set(initialize=data['locations'].keys())
    model.VEHICLES = Set(initialize=data['vehicles'].keys())
    model.TASKS = Set(initialize=data['tasks'].keys())
    model.STATIONS = Set(initialize=data['stations'].keys())
    model.TIME = Set(initialize=data['time_steps'])
    model.DEPOT = Set(initialize=['Depot'])
    model.NODES = model.LOCATIONS - model.DEPOT

    # --- 2. 决策变量 ---
    model.x = Var(model.LOCATIONS, model.LOCATIONS, model.VEHICLES, within=Binary)
    model.task_assigned = Var(model.TASKS, model.VEHICLES, within=Binary)
    model.is_task_unassigned = Var(model.TASKS, within=Binary)
    model.arrival_time = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.soc_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.weight_depart = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.swap_decision = Var(model.STATIONS, model.VEHICLES, within=Binary)
    model.task_delay = Var(model.TASKS, model.VEHICLES, within=NonNegativeReals)
    model.P_grid = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.P_charge_batt = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.ev_unserved = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.N_full_batt = Var(model.STATIONS, model.TIME, within=NonNegativeIntegers)
    model.N_empty_batt = Var(model.STATIONS, model.TIME, within=NonNegativeIntegers)
    model.hdt_swap_in_t = Var(model.STATIONS, model.VEHICLES, model.TIME, within=Binary)

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

    # -- 任务分配约束 --
    for t in model.TASKS:
        model.constrs.add(sum(model.task_assigned[t, k] for k in model.VEHICLES) + model.is_task_unassigned[t] == 1)

    # -- 车辆路径与序列约束 (使用MTZ方法) --
    for k in model.VEHICLES:
        depot = next(iter(model.DEPOT))
        # 车辆从车场出发，如果出发必须返回
        model.constrs.add(sum(model.x[depot, j, k] for j in model.NODES) <= 1)
        model.constrs.add(
            sum(model.x[i, depot, k] for i in model.NODES) == sum(model.x[depot, j, k] for j in model.NODES))

        # 节点流平衡
        for n in model.NODES:
            model.constrs.add(sum(model.x[i, n, k] for i in model.LOCATIONS if i != n) == sum(
                model.x[n, j, k] for j in model.LOCATIONS if j != n))

        # 任务节点必须由分配到的车辆访问
        for t in model.TASKS:
            P, D = data['tasks'][t]['pickup'], data['tasks'][t]['delivery']
            model.constrs.add(sum(model.x[i, P, k] for i in model.LOCATIONS if i != P) == model.task_assigned[t, k])
            model.constrs.add(sum(model.x[i, D, k] for i in model.LOCATIONS if i != D) == model.task_assigned[t, k])

    # -- 物理状态传播 (最终修复版) --
    for k in model.VEHICLES:
        depot = next(iter(model.DEPOT))
        model.constrs.add(model.arrival_time[depot, k] == 0)
        model.constrs.add(model.soc_arrival[depot, k] == data['vehicles'][k]['initial_soc'])
        model.constrs.add(model.weight_depart[depot, k] == config.HDT_EMPTY_WEIGHT_TON)

        for j in model.NODES:
            # 载重传播 (修复版)
            weight_on_arrival = sum(model.weight_depart[i, k] * model.x[i, j, k] for i in model.LOCATIONS if i != j)
            pickup_at_j = sum(model.task_assigned[t, k] * data['tasks'][t]['weight'] for t in model.TASKS if
                              j == data['tasks'][t]['pickup'])
            delivery_at_j = sum(model.task_assigned[t, k] * data['tasks'][t]['weight'] for t in model.TASKS if
                                j == data['tasks'][t]['delivery'])
            # 离开j的重量 = 到达j的重量 + 在j的装货 - 在j的卸货
            model.constrs.add(model.weight_depart[j, k] == weight_on_arrival + pickup_at_j - delivery_at_j)

            # 时间与电量传播
            for i in model.LOCATIONS:
                if i != j:
                    service_time = 0
                    if i in {info['pickup'] for info in data['tasks'].values()} or i in {info['delivery'] for info in
                                                                                         data['tasks'].values()}:
                        service_time = config.LOADING_UNLOADING_TIME_HOURS
                    if i in model.STATIONS:
                        service_time += model.swap_decision[i, k] * config.SWAP_DURATION_HOURS

                    departure_time_i = model.arrival_time[i, k] + service_time
                    soc_departure_i = model.soc_arrival[i, k] + (model.swap_decision[i, k] * (
                                config.HDT_BATTERY_CAPACITY_KWH - model.soc_arrival[
                            i, k]) if i in model.STATIONS else 0)
                    energy_consumed = data['dist_matrix'].loc[i, j] * (
                                config.HDT_BASE_CONSUMPTION_KWH_PER_KM + model.weight_depart[
                            i, k] * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)

                    model.constrs.add(
                        model.arrival_time[j, k] >= departure_time_i + data['time_matrix'].loc[i, j] - config.BIG_M * (
                                    1 - model.x[i, j, k]))
                    model.constrs.add(model.soc_arrival[j, k] <= soc_departure_i - energy_consumed + config.BIG_M * (
                                1 - model.x[i, j, k]))

        for t in model.TASKS:
            P, D = data['tasks'][t]['pickup'], data['tasks'][t]['delivery']
            model.constrs.add(
                model.arrival_time[D, k] >= model.arrival_time[P, k] + data['time_matrix'].loc[P, D] - config.BIG_M * (
                            1 - model.task_assigned[t, k]))
            model.constrs.add(model.arrival_time[P, k] >= data['tasks'][t]['ready_time'] * model.task_assigned[t, k])
            model.constrs.add(model.arrival_time[D, k] <= data['tasks'][t]['due_time'] + model.task_delay[t, k])

        for n in model.NODES:
            model.constrs.add(model.soc_arrival[n, k] >= config.HDT_MIN_SOC_KWH - config.BIG_M * (
                        1 - sum(model.x[i, n, k] for i in model.LOCATIONS if i != n)))

    # -- 能源站约束 (修复版) --
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

    print("最终交付版模型构建完成。")
    return model