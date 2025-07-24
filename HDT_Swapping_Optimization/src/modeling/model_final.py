# HDT_Swapping_Optimization/src/modeling/model_final.py

from pyomo.environ import *
import numpy as np
from src.common import config_final as config


def create_operational_model(data, vehicle_ids, task_ids, deactivated_constraints=None):
    if deactivated_constraints is None:
        deactivated_constraints = []
    print(f"开始构建【第二阶段：运营执行 - 标准VRP逻辑版】模型...")
    print(f"  -> 规划车辆: {vehicle_ids}")
    print(f"  -> 规划任务: {task_ids}")

    model = ConcreteModel(name="Operational_VRP_Model")

    # --- Sets & Params ---
    model.LOCATIONS = Set(initialize=data['locations'].keys())
    model.VEHICLES = Set(initialize=vehicle_ids)
    model.TASKS = Set(initialize=task_ids)
    model.CUSTOMERS = Set(initialize=[data['tasks'][t]['delivery_to'] for t in model.TASKS if t in task_ids])
    model.STATIONS = Set(initialize=data['stations'].keys())
    model.TIME = Set(initialize=data['time_steps'])
    depot_ids = list({v['depot_id'] for k, v in data['vehicles'].items() if k in vehicle_ids})
    model.DEPOT = Set(initialize=depot_ids)
    model.NODES = model.LOCATIONS - model.DEPOT
    customer_demands = {data['tasks'][t]['delivery_to']: data['tasks'][t]['demand'] for t in model.TASKS}

    # --- Decision Variables ---
    model.x = Var(model.LOCATIONS, model.LOCATIONS, model.VEHICLES, within=Binary)
    model.y = Var(model.LOCATIONS, model.VEHICLES, within=Binary)
    model.arrival_time = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.departure_time = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.tour_duration = Var(model.VEHICLES, within=NonNegativeReals)
    model.soc_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals,
                            bounds=(0, config.HDT_BATTERY_CAPACITY_KWH))
    model.weight_on_arrival = Var(model.LOCATIONS, model.VEHICLES, within=NonNegativeReals)
    model.swap_decision = Var(model.STATIONS, model.VEHICLES, within=Binary)
    model.task_delay = Var(model.TASKS, model.VEHICLES, within=NonNegativeReals)
    model.soc_violation = Var(model.NODES, model.VEHICLES, within=NonNegativeReals)
    model.P_grid = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.P_charge_batt = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.ev_unserved = Var(model.STATIONS, model.TIME, within=NonNegativeReals)
    model.N_full_batt = Var(model.STATIONS, model.TIME, within=NonNegativeIntegers)
    model.N_empty_batt = Var(model.STATIONS, model.TIME, within=NonNegativeIntegers)
    model.hdt_swap_in_t = Var(model.STATIONS, model.VEHICLES, model.TIME, within=Binary)
    model.soc_swap_product = Var(model.STATIONS, model.VEHICLES, within=NonNegativeReals)

    # --- Objective Function ---
    def objective_rule(m):
        travel_cost = config.MANPOWER_COST_PER_HOUR * sum(m.tour_duration[k] for k in m.VEHICLES)
        swap_cost = config.FIXED_SWAP_COST * sum(m.swap_decision[s, k] for s in m.STATIONS for k in m.VEHICLES)
        grid_cost = sum(
            m.P_grid[s, t] * config.TIME_STEP_HOURS * data['electricity_prices'][t] for s in m.STATIONS for t in m.TIME)
        delay_penalty = config.DELAY_PENALTY_PER_HOUR * sum(m.task_delay[t, k] for t in m.TASKS for k in m.VEHICLES)
        ev_penalty = config.EV_UNSERVED_PENALTY_PER_KWH * sum(
            m.ev_unserved[s, t] * config.TIME_STEP_HOURS for s in m.STATIONS for t in m.TIME)
        unserved_task_penalty = config.UNASSIGNED_TASK_PENALTY * sum(
            1 - sum(m.y[data['tasks'][t]['delivery_to'], k] for k in m.VEHICLES)
            for t in m.TASKS
        )
        soc_violation_penalty = config.SOC_VIOLATION_PENALTY_PER_KWH * sum(
            m.soc_violation[n, k] for n in m.NODES for k in m.VEHICLES
        )
        return travel_cost + swap_cost + grid_cost + delay_penalty + ev_penalty + unserved_task_penalty + soc_violation_penalty

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- Standard VRP Routing Constraints ---
    if 'hdt_routing' not in deactivated_constraints:
        def task_served_rule(m, t):
            customer = data['tasks'][t]['delivery_to']
            return sum(m.y[customer, k] for k in m.VEHICLES) <= 1

        model.task_served_constr = Constraint(model.TASKS, rule=task_served_rule)

        def flow_balance_rule(m, k, n):
            return sum(m.x[i, n, k] for i in m.LOCATIONS if i != n) == sum(m.x[n, j, k] for j in m.LOCATIONS if j != n)

        model.flow_balance_constr = Constraint(model.VEHICLES, model.LOCATIONS, rule=flow_balance_rule)

        def start_from_depot_rule(m, k):
            depot_id = data['vehicles'][k]['depot_id']
            return sum(m.x[depot_id, j, k] for j in m.LOCATIONS if j != depot_id) <= 1

        model.start_from_depot_constr = Constraint(model.VEHICLES, rule=start_from_depot_rule)

        def y_x_relation_rule(m, k, n):
            return m.y[n, k] == sum(m.x[i, n, k] for i in m.LOCATIONS if i != n)

        model.y_x_relation_constr = Constraint(model.VEHICLES, model.NODES, rule=y_x_relation_rule)

        def task_delay_rule(m, t, k):
            customer = data['tasks'][t]['delivery_to']
            return m.task_delay[t, k] >= m.arrival_time[customer, k] - data['tasks'][t]['due_time']

        model.task_delay_constr = Constraint(model.TASKS, model.VEHICLES, rule=task_delay_rule)

    # --- Initial State Constraints ---
    def initial_time_rule(m, k):
        depot_id = data['vehicles'][k]['depot_id']
        return m.arrival_time[depot_id, k] == 0

    model.initial_time_constr = Constraint(model.VEHICLES, rule=initial_time_rule)

    def initial_soc_rule(m, k):
        depot_id = data['vehicles'][k]['depot_id']
        return m.soc_arrival[depot_id, k] == data['vehicles'][k]['initial_soc']

    model.initial_soc_constr = Constraint(model.VEHICLES, rule=initial_soc_rule)

    def initial_weight_rule(m, k):
        depot_id = data['vehicles'][k]['depot_id']
        return m.weight_on_arrival[depot_id, k] == config.HDT_EMPTY_WEIGHT_TON

    model.initial_weight_constr = Constraint(model.VEHICLES, rule=initial_weight_rule)

    # --- Big-M Constraints for State Propagation ---
    M_time = (max(data['time_steps']) + 1) * config.TIME_HORIZON_HOURS if data['time_steps'] else (24 * 4 + 1)
    M_weight = config.HDT_EMPTY_WEIGHT_TON + sum(customer_demands.values()) + 1
    M_soc = config.HDT_BATTERY_CAPACITY_KWH

    def valid_arcs_rule(m):
        for k in m.VEHICLES:
            for i in m.LOCATIONS:
                for j in m.LOCATIONS:
                    if i != j and np.isfinite(data['time_matrix'].loc[i, j]):
                        yield (i, j, k)

    model.ARCS = Set(dimen=3, initialize=valid_arcs_rule)
    model.ARCS_TO_NODES = Set(initialize=model.ARCS, filter=lambda m, i, j, k: j not in m.DEPOT)
    model.ARCS_TO_DEPOT = Set(initialize=model.ARCS, filter=lambda m, i, j, k: j in m.DEPOT)

    if 'hdt_weight' not in deactivated_constraints:
        def weight_propagate_upper_rule(m, i, j, k):
            depot_id = data['vehicles'][k]['depot_id']
            demand_at_i = customer_demands.get(i, 0)
            if i == depot_id:
                weight_depart_i = config.HDT_EMPTY_WEIGHT_TON + sum(
                    customer_demands.get(data['tasks'][t]['delivery_to'], 0) * m.y[data['tasks'][t]['delivery_to'], k]
                    for t in m.TASKS)
            else:
                weight_depart_i = m.weight_on_arrival[i, k] - demand_at_i * m.y[i, k]
            return m.weight_on_arrival[j, k] - weight_depart_i <= M_weight * (1 - m.x[i, j, k])

        model.weight_propagate_upper_constr = Constraint(model.ARCS, rule=weight_propagate_upper_rule)

        def weight_propagate_lower_rule(m, i, j, k):
            depot_id = data['vehicles'][k]['depot_id']
            demand_at_i = customer_demands.get(i, 0)
            if i == depot_id:
                weight_depart_i = config.HDT_EMPTY_WEIGHT_TON + sum(
                    customer_demands.get(data['tasks'][t]['delivery_to'], 0) * m.y[data['tasks'][t]['delivery_to'], k]
                    for t in m.TASKS)
            else:
                weight_depart_i = m.weight_on_arrival[i, k] - demand_at_i * m.y[i, k]
            return m.weight_on_arrival[j, k] - weight_depart_i >= -M_weight * (1 - m.x[i, j, k])

        model.weight_propagate_lower_constr = Constraint(model.ARCS, rule=weight_propagate_lower_rule)

    if 'hdt_time' not in deactivated_constraints:
        def link_swap_to_visit_rule(m, s, k):
            return m.swap_decision[s, k] <= m.y[s, k]

        model.link_swap_to_visit_constr = Constraint(model.STATIONS, model.VEHICLES, rule=link_swap_to_visit_rule)

        def bound_arrival_time_rule(m, n, k):
            return m.arrival_time[n, k] <= M_time * m.y[n, k]

        model.bound_arrival_time_constr = Constraint(model.NODES, model.VEHICLES, rule=bound_arrival_time_rule)

        # --- [修正] 这是解决崩溃问题的关键 ---
        # 我们需要用上下界两个约束把departure_time“锁死”在arrival_time + service_time上，
        # 避免其“浮动”导致计算爆炸。
        def departure_time_lower_rule(m, i, k):
            service_time = 0
            if i in m.CUSTOMERS:
                service_time = config.LOADING_UNLOADING_TIME_HOURS
            swap_time = 0
            if i in m.STATIONS:
                swap_time = m.swap_decision[i, k] * config.SWAP_DURATION_HOURS
            return m.departure_time[i, k] >= (m.arrival_time[i, k] + service_time + swap_time) - M_time * (
                        1 - m.y[i, k])

        model.departure_time_lower_constr = Constraint(model.LOCATIONS, model.VEHICLES, rule=departure_time_lower_rule)

        def departure_time_upper_rule(m, i, k):
            service_time = 0
            if i in m.CUSTOMERS:
                service_time = config.LOADING_UNLOADING_TIME_HOURS
            swap_time = 0
            if i in m.STATIONS:
                swap_time = m.swap_decision[i, k] * config.SWAP_DURATION_HOURS
            # 新增的上限约束
            return m.departure_time[i, k] <= (m.arrival_time[i, k] + service_time + swap_time) + M_time * (
                        1 - m.y[i, k])

        model.departure_time_upper_constr = Constraint(model.LOCATIONS, model.VEHICLES, rule=departure_time_upper_rule)

        def time_propagate_to_node_rule(m, i, j, k):
            travel_time = data['time_matrix'].loc[i, j]
            return m.arrival_time[j, k] >= m.departure_time[i, k] + travel_time - M_time * (1 - m.x[i, j, k])

        model.time_propagate_to_node_constr = Constraint(model.ARCS_TO_NODES, rule=time_propagate_to_node_rule)

        def time_propagate_to_depot_rule(m, i, j, k):
            travel_time = data['time_matrix'].loc[i, j]
            return m.tour_duration[k] >= m.departure_time[i, k] + travel_time - M_time * (1 - m.x[i, j, k])

        model.time_propagate_to_depot_constr = Constraint(model.ARCS_TO_DEPOT, rule=time_propagate_to_depot_rule)

    if 'hdt_soc' not in deactivated_constraints:
        def linearize_soc_swap_1(m, s, k):
            return m.soc_swap_product[s, k] <= M_soc * m.swap_decision[s, k]

        model.linearize_soc_swap_1_constr = Constraint(model.STATIONS, model.VEHICLES, rule=linearize_soc_swap_1)

        def linearize_soc_swap_2(m, s, k):
            return m.soc_swap_product[s, k] <= m.soc_arrival[s, k]

        model.linearize_soc_swap_2_constr = Constraint(model.STATIONS, model.VEHICLES, rule=linearize_soc_swap_2)

        def linearize_soc_swap_3(m, s, k):
            return m.soc_swap_product[s, k] >= m.soc_arrival[s, k] - M_soc * (1 - m.swap_decision[s, k])

        model.linearize_soc_swap_3_constr = Constraint(model.STATIONS, model.VEHICLES, rule=linearize_soc_swap_3)

        model.soc_propagate_constr = ConstraintList()
        for i, j, k in model.ARCS:
            depot_id = data['vehicles'][k]['depot_id']
            demand_at_i = customer_demands.get(i, 0)
            if i == depot_id:
                weight_depart_i = config.HDT_EMPTY_WEIGHT_TON + sum(
                    customer_demands.get(data['tasks'][t]['delivery_to'], 0) * model.y[
                        data['tasks'][t]['delivery_to'], k]
                    for t in model.TASKS)
            else:
                weight_depart_i = model.weight_on_arrival[i, k] - demand_at_i * model.y[i, k]
            soc_after_service = model.soc_arrival[i, k]
            if i in model.STATIONS:
                soc_after_service += model.swap_decision[i, k] * config.HDT_BATTERY_CAPACITY_KWH - \
                                     model.soc_swap_product[i, k]
            energy_consumed = data['dist_matrix'].loc[i, j] * (
                    config.HDT_BASE_CONSUMPTION_KWH_PER_KM + weight_depart_i * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
            soc_at_j = model.soc_arrival[j, k]
            soc_depart_i = soc_after_service - energy_consumed
            model.soc_propagate_constr.add(soc_at_j - soc_depart_i <= M_soc * (1 - model.x[i, j, k]))
            model.soc_propagate_constr.add(soc_at_j - soc_depart_i >= -M_soc * (1 - model.x[i, j, k]))

        def min_soc_rule(m, k, n):
            return m.soc_arrival[n, k] + m.soc_violation[n, k] >= config.HDT_MIN_SOC_KWH - M_soc * (1 - m.y[n, k])

        model.min_soc_constr = Constraint(model.VEHICLES, model.NODES, rule=min_soc_rule)

    print(f"【第二阶段 - 标准VRP逻辑版】模型构建完成。")
    return model