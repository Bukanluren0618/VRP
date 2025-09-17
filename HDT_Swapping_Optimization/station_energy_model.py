# src/modeling/station_energy_model.py

from pyomo.environ import *


def create_station_energy_model(station_id, data, hdt_demand_series, config):
    """
    Creates a detailed energy dispatch optimization model for a single station.
    Its goal is to minimize energy costs while satisfying all charging demands (HDT + external EV).
    It intelligently decides when to buy from the grid, use PV, and charge/discharge the BESS.
    This model IS the implementation of the "basic grid constraints".
    """
    model = ConcreteModel(name=f"StationEnergyDispatch_{station_id}")

    # --- 1. Sets & Parameters ---
    model.T = Set(initialize=data['time_steps'])

    # Parameters
    model.pv_gen = Param(model.T, initialize=data['pv_generation'][station_id].to_dict())
    model.ev_demand = Param(model.T, initialize=data['ev_demand_timestep'][station_id].to_dict())
    model.hdt_demand = Param(model.T, initialize=hdt_demand_series.to_dict())
    model.price = Param(model.T, initialize=data['electricity_prices'].to_dict())

    # Battery Energy Storage System (BESS) Parameters
    model.BESS_CAPACITY = Param(initialize=config.BESS_CAPACITY_KWH)
    model.BESS_MAX_POWER = Param(initialize=config.BESS_MAX_POWER_KW)
    model.BESS_EFFICIENCY = Param(initialize=config.BESS_EFFICIENCY)

    model.TIME_STEP_HOURS = Param(initialize=config.TIME_STEP_HOURS)

    # --- 2. Decision Variables ---
    model.p_grid = Var(model.T, within=NonNegativeReals)
    model.p_bess_ch = Var(model.T, within=NonNegativeReals, bounds=(0, model.BESS_MAX_POWER))
    model.p_bess_dis = Var(model.T, within=NonNegativeReals, bounds=(0, model.BESS_MAX_POWER))
    model.e_bess = Var(model.T, within=NonNegativeReals, bounds=(0, model.BESS_CAPACITY))
    model.p_pv_curtail = Var(model.T, within=NonNegativeReals)

    # --- 3. Objective Function: Minimize Total Operational Cost ---
    def objective_rule(m):
        # Cost = (Power purchased from grid * duration * price)
        cost = sum(m.p_grid[t] * m.TIME_STEP_HOURS * m.price[t] for t in m.T)
        return cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- 4. Constraints ---
    # Constraint 1: Energy Balance (for each time step t)
    # Energy Sources == Energy Sinks
    def energy_balance_rule(m, t):
        energy_source = m.p_grid[t] + m.pv_gen[t] + m.p_bess_dis[t]
        energy_sink = m.hdt_demand[t] + m.ev_demand[t] + m.p_bess_ch[t] + m.p_pv_curtail[t]
        return energy_source == energy_sink

    model.energy_balance_constr = Constraint(model.T, rule=energy_balance_rule)

    # Constraint 2: BESS State of Charge (SOC) Dynamics
    def bess_soc_rule(m, t):
        if t == m.T.first():
            # Assume BESS starts at 50% capacity
            return m.e_bess[t] == m.BESS_CAPACITY / 2
        else:
            prev_t = m.T.prev(t)
            charge_energy = m.p_bess_ch[prev_t] * m.TIME_STEP_HOURS * m.BESS_EFFICIENCY
            discharge_energy = m.p_bess_dis[prev_t] * m.TIME_STEP_HOURS / m.BESS_EFFICIENCY
            return m.e_bess[t] == m.e_bess[prev_t] + charge_energy - discharge_energy

    model.bess_soc_constr = Constraint(model.T, rule=bess_soc_rule)

    # Constraint 3: PV curtailment cannot exceed PV generation
    def pv_curtailment_rule(m, t):
        return m.p_pv_curtail[t] <= m.pv_gen[t]

    model.pv_curtailment_constr = Constraint(model.T, rule=pv_curtailment_rule)

    return model