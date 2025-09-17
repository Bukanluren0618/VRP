# src/modeling/station_energy_model.py

from pyomo.environ import *


def create_station_energy_model(station_id, data, hdt_demand_series, config):
    """
    为单个换电站创建一个详细的能源调度优化模型。
    这个模型的目标是在满足所有充电需求（HDT + 外部EV）的前提下，最小化能源成本。
    它会智能地决定何时从电网买电、何时使用光伏、何时对储能系统充放电。
    """
    model = ConcreteModel(name=f"StationEnergyDispatch_{station_id}")

    # --- 1. 集合与参数 ---
    model.T = Set(initialize=data['time_steps'])

    # --- 参数 ---
    model.pv_gen = Param(model.T, initialize=data['pv_generation'][station_id].to_dict())
    model.ev_demand = Param(model.T, initialize=data['ev_demand_timestep'][station_id].to_dict())
    model.hdt_demand = Param(model.T, initialize=hdt_demand_series.to_dict())
    model.price = Param(model.T, initialize=data['electricity_prices'].to_dict())

    # 储能系统(BESS)参数
    model.BESS_CAPACITY = Param(initialize=config.BESS_CAPACITY_KWH)
    model.BESS_MAX_POWER = Param(initialize=config.BESS_MAX_POWER_KW)
    model.BESS_EFFICIENCY = Param(initialize=config.BESS_EFFICIENCY)

    # 时间步长（小时）
    model.TIME_STEP_HOURS = Param(initialize=config.TIME_STEP_HOURS)

    # --- 2. 决策变量 ---
    # 从电网购买的功率 (kW)
    model.p_grid = Var(model.T, within=NonNegativeReals)
    # 储能充电功率 (kW)
    model.p_bess_ch = Var(model.T, within=NonNegativeReals, bounds=(0, model.BESS_MAX_POWER))
    # 储能放电功率 (kW)
    model.p_bess_dis = Var(model.T, within=NonNegativeReals, bounds=(0, model.BESS_MAX_POWER))
    # 储能系统的能量状态 (State of Charge, kWh)
    model.e_bess = Var(model.T, within=NonNegativeReals, bounds=(0, model.BESS_CAPACITY))
    # 弃光功率 (kW) - 当光伏发电量超过需求和存储能力时
    model.p_pv_curtail = Var(model.T, within=NonNegativeReals)

    # --- 3. 目标函数: 最小化总运营成本 ---
    def objective_rule(m):
        # 成本 = (从电网购买的电量 * 电价)
        cost = sum(m.p_grid[t] * m.TIME_STEP_HOURS * m.price[t] for t in m.T)
        return cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- 4. 约束条件 ---
    # 约束1: 能量平衡约束 (对于每个时间步 t)
    # 能量来源 = 能量去向
    # (从电网来的 + 光伏发的 + 储能放的) = (给HDT的 + 给外部EV的 + 给储能充的 + 弃掉的光)
    def energy_balance_rule(m, t):
        energy_source = m.p_grid[t] + m.pv_gen[t] + m.p_bess_dis[t]
        energy_sink = m.hdt_demand[t] + m.ev_demand[t] + m.p_bess_ch[t] + m.p_pv_curtail[t]
        return energy_source == energy_sink

    model.energy_balance_constr = Constraint(model.T, rule=energy_balance_rule)

    # 约束2: 储能SOC (State of Charge) 动态变化
    # 当前SOC = 上一时刻SOC + 充电量 - 放电量
    def bess_soc_rule(m, t):
        if t == m.T.first():  # 初始状态
            # 假设储能系统初始电量为50%
            return m.e_bess[t] == m.BESS_CAPACITY / 2
        else:
            prev_t = m.T.prev(t)
            charge_energy = m.p_bess_ch[prev_t] * m.TIME_STEP_HOURS * m.BESS_EFFICIENCY
            discharge_energy = m.p_bess_dis[prev_t] * m.TIME_STEP_HOURS / m.BESS_EFFICIENCY
            return m.e_bess[t] == m.e_bess[prev_t] + charge_energy - discharge_energy

    model.bess_soc_constr = Constraint(model.T, rule=bess_soc_rule)

    # 约束3: 弃光功率不能超过当时的光伏发电量
    def pv_curtailment_rule(m, t):
        return m.p_pv_curtail[t] <= m.pv_gen[t]

    model.pv_curtailment_constr = Constraint(model.T, rule=pv_curtailment_rule)

    return model