# src/modeling/model_final.py

from pyomo.environ import *
import numpy as np
from src.common import config_final as config


def create_operational_model(data, vehicle_ids, task_ids, fixed_route=None):
    model = ConcreteModel(name="Operational_VRP_Model_Final_Corrected")

    # --- Sets & Params ---
    model.LOCATIONS = Set(initialize=data['locations'].keys())
    model.VEHICLES = Set(initialize=vehicle_ids)
    model.TASKS = Set(initialize=task_ids)
    all_customer_nodes = {data['tasks'][t]['delivery_to'] for t in task_ids} if task_ids else set()
    model.CUSTOMERS = Set(initialize=list(all_customer_nodes))
    model.STATIONS = Set(initialize=data['stations'].keys())
    depot_ids = list({v['depot_id'] for k, v in data['vehicles'].items() if k in vehicle_ids})
    model.DEPOT = Set(initialize=depot_ids)
    model.NODES = model.LOCATIONS - model.DEPOT
    customer_demands = {data['tasks'][t]['delivery_to']: data['tasks'][t]['demand'] for t in model.TASKS}

    M_time = config.TIME_HORIZON_HOURS * 2
    M_weight = config.HDT_EMPTY_WEIGHT_TON + sum(d for d in customer_demands.values()) + 1
    M_soc = config.HDT_BATTERY_CAPACITY_KWH

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
    model.delay_hours = Var(model.TASKS, model.VEHICLES, within=NonNegativeReals)
    model.soc_deficit = Var(model.NODES, model.VEHICLES, within=NonNegativeReals)

    # --- Objective Function ---
    def objective_rule(m):
        travel_cost = summation(m.tour_duration) * config.MANPOWER_COST_PER_HOUR
        swap_cost = summation(m.swap_decision) * config.FIXED_SWAP_COST
        delay_penalty = summation(m.delay_hours) * config.DELAY_PENALTY_PER_HOUR
        soc_deficit_penalty = summation(m.soc_deficit) * 1e6  # High penalty
        unserved_task_penalty = 0
        if not fixed_route:
            unserved_task_penalty = (len(m.TASKS) - summation(m.y)) * config.UNASSIGNED_TASK_PENALTY
        return travel_cost + swap_cost + delay_penalty + soc_deficit_penalty + unserved_task_penalty

    model.objective = Objective(rule=objective_rule, sense=minimize)

    model.ARCS = Set(dimen=3,
                     initialize=[(i, j, k) for k in model.VEHICLES for i in model.LOCATIONS for j in model.LOCATIONS if
                                 i != j])
    # *** KEY FIX: Create a set for arcs that DO NOT end at a depot ***
    model.ARCS_TO_NODES = Set(initialize=model.ARCS, filter=lambda m, i, j, k: j not in m.DEPOT)

    # --- 固定路径 (用于验证模式) ---
    if fixed_route:
        vehicle_id = list(model.VEHICLES)[0]
        # Deactivate all arcs first
        for i, j, k in model.ARCS:
            if k == vehicle_id:
                model.x[i, j, k].fix(0)
        # Activate the arcs in the fixed route
        for i in range(len(fixed_route) - 1):
            u, v = fixed_route[i], fixed_route[i + 1]
            if u != v:
                model.x[u, v, vehicle_id].fix(1)

    # --- VRP Routing Constraints ---
    if not fixed_route:
        def task_served_rule(m, t):
            customer = data['tasks'][t]['delivery_to']
            return sum(m.y[customer, k] for k in m.VEHICLES) <= 1

        model.task_served_constr = Constraint(model.TASKS, rule=task_served_rule)

        def flow_balance_rule(m, k, n):
            return sum(m.x[i, n, k] for i in m.LOCATIONS if i != n) == sum(m.x[n, j, k] for j in m.LOCATIONS if j != n)

        model.flow_balance_constr = Constraint(model.VEHICLES, model.LOCATIONS, rule=flow_balance_rule)

        def start_from_depot_rule(m, k):
            return sum(m.x[data['vehicles'][k]['depot_id'], j, k] for j in m.LOCATIONS if
                       j != data['vehicles'][k]['depot_id']) <= 1

        model.start_from_depot_constr = Constraint(model.VEHICLES, rule=start_from_depot_rule)

    def y_x_relation_rule(m, k, n):
        return m.y[n, k] == sum(m.x[i, n, k] for i in m.LOCATIONS if i != n)

    model.y_x_relation_constr = Constraint(model.VEHICLES, model.NODES, rule=y_x_relation_rule)

    # --- Initial State Constraints (at Depot) ---
    def initial_time_rule(m, k):
        return m.arrival_time[data['vehicles'][k]['depot_id'], k] == 0

    model.initial_time_constr = Constraint(model.VEHICLES, rule=initial_time_rule)

    def departure_time_depot_rule(m, k):
        return m.departure_time[data['vehicles'][k]['depot_id'], k] == m.arrival_time[
            data['vehicles'][k]['depot_id'], k]

    model.departure_time_depot_constr = Constraint(model.VEHICLES, rule=departure_time_depot_rule)

    def initial_soc_rule(m, k):
        return m.soc_arrival[data['vehicles'][k]['depot_id'], k] == data['vehicles'][k]['initial_soc']

    model.initial_soc_constr = Constraint(model.VEHICLES, rule=initial_soc_rule)

    def initial_weight_rule(m, k):
        initial_load = sum(
            customer_demands[data['tasks'][t]['delivery_to']] * m.y[data['tasks'][t]['delivery_to'], k] for t in
            m.TASKS)
        return m.weight_on_arrival[data['vehicles'][k]['depot_id'], k] == config.HDT_EMPTY_WEIGHT_TON + initial_load

    model.initial_weight_constr = Constraint(model.VEHICLES, rule=initial_weight_rule)

    # --- State Propagation Constraints (for arcs to non-depot nodes) ---
    # *** KEY FIX: All propagation constraints now apply ONLY to ARCS_TO_NODES ***
    def weight_propagate_upper_rule(m, i, j, k):
        weight_depart_i = m.weight_on_arrival[i, k] - customer_demands.get(i, 0) * m.y[i, k]
        return m.weight_on_arrival[j, k] - weight_depart_i <= M_weight * (1 - m.x[i, j, k])

    model.weight_propagate_upper = Constraint(model.ARCS_TO_NODES, rule=weight_propagate_upper_rule)

    def weight_propagate_lower_rule(m, i, j, k):
        weight_depart_i = m.weight_on_arrival[i, k] - customer_demands.get(i, 0) * m.y[i, k]
        return m.weight_on_arrival[j, k] - weight_depart_i >= -M_weight * (1 - m.x[i, j, k])

    model.weight_propagate_lower = Constraint(model.ARCS_TO_NODES, rule=weight_propagate_lower_rule)

    def time_propagate_rule(m, i, j, k):
        travel_time = data['time_matrix'].loc[i, j]
        return m.arrival_time[j, k] >= m.departure_time[i, k] + travel_time - M_time * (1 - m.x[i, j, k])

    model.time_propagate_constr = Constraint(model.ARCS_TO_NODES, rule=time_propagate_rule)

    def soc_propagate_upper_rule(m, i, j, k):
        weight_depart_i = m.weight_on_arrival[i, k] - customer_demands.get(i, 0) * m.y[i, k] if i not in m.DEPOT else \
        m.weight_on_arrival[i, k]
        energy_consumed = data['dist_matrix'].loc[i, j] * (
                    config.HDT_BASE_CONSUMPTION_KWH_PER_KM + weight_depart_i * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
        soc_depart_i = m.soc_arrival[i, k]
        if i in m.STATIONS: soc_depart_i += m.swap_decision[i, k] * (
                    config.HDT_BATTERY_CAPACITY_KWH - m.soc_arrival[i, k])
        return m.soc_arrival[j, k] - (soc_depart_i - energy_consumed) <= M_soc * (1 - m.x[i, j, k])

    model.soc_propagate_upper = Constraint(model.ARCS_TO_NODES, rule=soc_propagate_upper_rule)

    def soc_propagate_lower_rule(m, i, j, k):
        weight_depart_i = m.weight_on_arrival[i, k] - customer_demands.get(i, 0) * m.y[i, k] if i not in m.DEPOT else \
        m.weight_on_arrival[i, k]
        energy_consumed = data['dist_matrix'].loc[i, j] * (
                    config.HDT_BASE_CONSUMPTION_KWH_PER_KM + weight_depart_i * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
        soc_depart_i = m.soc_arrival[i, k]
        if i in m.STATIONS: soc_depart_i += m.swap_decision[i, k] * (
                    config.HDT_BATTERY_CAPACITY_KWH - m.soc_arrival[i, k])
        return m.soc_arrival[j, k] - (soc_depart_i - energy_consumed) >= -M_soc * (1 - m.x[i, j, k])

    model.soc_propagate_lower = Constraint(model.ARCS_TO_NODES, rule=soc_propagate_lower_rule)

    # --- Node-based Constraints (at non-depot nodes) ---
    def link_swap_to_visit_rule(m, s, k):
        return m.swap_decision[s, k] <= m.y[s, k]

    model.link_swap_to_visit_constr = Constraint(model.STATIONS, model.VEHICLES, rule=link_swap_to_visit_rule)

    def departure_time_rule(m, i, k):
        service_time = config.LOADING_UNLOADING_TIME_HOURS if i in m.CUSTOMERS else 0
        swap_time = config.SWAP_DURATION_HOURS * m.swap_decision[i, k] if i in m.STATIONS else 0
        return m.departure_time[i, k] >= m.arrival_time[i, k] + service_time + swap_time - M_time * (1 - m.y[i, k])

    model.departure_time_constr = Constraint(model.NODES, model.VEHICLES, rule=departure_time_rule)

    def delay_calculation_rule(m, t, k):
        customer = data['tasks'][t]['delivery_to']
        return m.delay_hours[t, k] >= m.arrival_time[customer, k] - data['tasks'][t]['due_time'] - M_time * (
                    1 - m.y[customer, k])

    model.delay_calculation_constr = Constraint(model.TASKS, model.VEHICLES, rule=delay_calculation_rule)

    def min_soc_rule(m, k, n):
        return m.soc_deficit[n, k] >= config.HDT_MIN_SOC_KWH - m.soc_arrival[n, k] - M_soc * (1 - m.y[n, k])

    model.min_soc_constr = Constraint(model.VEHICLES, model.NODES, rule=min_soc_rule)

    # --- Final State Calculation (for arcs returning to depot) ---
    def tour_duration_rule(m, i, k):
        if i in m.DEPOT: return Constraint.Skip  # Should only trigger from a non-depot node
        depot_id = data['vehicles'][k]['depot_id']
        return m.tour_duration[k] >= m.departure_time[i, k] + data['time_matrix'].loc[i, depot_id] - M_time * (
                    1 - m.x[i, depot_id, k])

    model.tour_duration_constr = Constraint(model.NODES, model.VEHICLES, rule=tour_duration_rule)

    return model