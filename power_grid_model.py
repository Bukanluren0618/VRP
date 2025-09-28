# src/modeling/power_grid_model.py

from pyomo.environ import *
import pandapower as pp


def create_power_grid_model(net, station_bus_map, hdt_loads_kw, config):
    """
    创建一个包含简化线性潮流约束的配电网优化模型。

    Args:
        net (pandapowerNet): pandapower电网对象.
        station_bus_map (dict): 换电站名称到电网母线(bus)ID的映射.
        hdt_loads_kw (dict): {bus_id: load_in_kw} 形式的字典.
        config: 全局配置文件.

    Returns:
        Pyomo ConcreteModel: 构建好的电网优化模型.
    """
    model = ConcreteModel(name="DistributionGridFlowModel")

    # --- 1. 集合与参数 ---
    model.BUSES = Set(initialize=net.bus.index.tolist())
    model.LINES = Set(initialize=net.line.index.tolist())

    # 线路参数
    model.line_r_pu = Param(model.LINES, initialize=net.line.r_ohm_per_km.to_dict())
    model.line_x_pu = Param(model.LINES, initialize=net.line.x_ohm_per_km.to_dict())
    model.line_max_i_ka = Param(model.LINES, initialize=net.line.max_i_ka.to_dict())

    # 电压基准 (kV) 和功率基准 (MVA)
    # 我们假设只有一个电压等级，取第一个母线的vn_kv
    model.v_base_kv = Param(initialize=net.bus.vn_kv.iloc[0])
    model.s_base_mva = Param(initialize=1.0)  # 假设基准功率为1 MVA

    # 计算阻抗和最大电流的标幺值
    z_base = model.v_base_kv ** 2 / model.s_base_mva
    i_base_ka = model.s_base_mva / (sqrt(3) * model.v_base_kv)

    line_r_pu_dict = {i: r / z_base for i, r in model.line_r_pu.items()}
    line_x_pu_dict = {i: x / z_base for i, x in model.line_x_pu.items()}
    line_max_i_pu_dict = {i: i_ka / i_base_ka for i, i_ka in model.line_max_i_ka.items()}

    model.R = Param(model.LINES, initialize=line_r_pu_dict)
    model.X = Param(model.LINES, initialize=line_x_pu_dict)
    model.I_max_pu = Param(model.LINES, initialize=line_max_i_pu_dict)

    # 负荷参数
    # 将kW转换为MW，然后转换为标幺值
    p_loads_pu = {bus: kw / (1000 * model.s_base_mva) for bus, kw in hdt_loads_kw.items()}
    # 假设功率因数为0.95，计算无功负荷
    q_loads_pu = {bus: p * tan(acos(0.95)) for bus, p in p_loads_pu.items()}

    model.P_load = Param(model.BUSES, initialize=p_loads_pu, default=0)
    model.Q_load = Param(model.BUSES, initialize=q_loads_pu, default=0)

    # --- 2. 决策变量 ---
    # 节点电压的平方 (V_i^2), 标幺值
    model.v_sqr = Var(model.BUSES, within=NonNegativeReals, bounds=(0.95 ** 2, 1.05 ** 2))
    # 线路有功功率潮流 (P_ij), 标幺值
    model.p_flow = Var(model.LINES, within=Reals)
    # 线路无功功率潮流 (Q_ij), 标幺值
    model.q_flow = Var(model.LINES, within=Reals)
    # 发电机/外部电网注入的有功和无功功率
    model.p_gen = Var(model.BUSES, within=Reals)
    model.q_gen = Var(model.BUSES, within=Reals)

    # --- 3. 目标函数: 最小化网损 (近似为线路电流的平方) ---
    def objective_rule(m):
        # 网损近似于 R * I^2, 而 I^2 = (P^2 + Q^2)/V^2
        # 为了保持线性，我们简化目标为最小化潮流的绝对值
        return sum(abs(m.p_flow[l]) for l in m.LINES)

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- 4. 约束条件 ---
    model.constrs = ConstraintList()

    # 约束1: 节点功率平衡 (Kirchhoff's Current Law)
    for b in model.BUSES:
        # 流入节点的功率 = 流出节点的功率
        power_in_p = model.p_gen[b] + sum(model.p_flow[l] for l in model.LINES if net.line.to_bus[l] == b)
        power_out_p = model.P_load[b] + sum(model.p_flow[l] for l in model.LINES if net.line.from_bus[l] == b)
        model.constrs.add(power_in_p == power_out_p)

        power_in_q = model.q_gen[b] + sum(model.q_flow[l] for l in model.LINES if net.line.to_bus[l] == b)
        power_out_q = model.Q_load[b] + sum(model.q_flow[l] for l in model.LINES if net.line.from_bus[l] == b)
        model.constrs.add(power_in_q == power_out_q)

    # 约束2: 电压降 (Ohm's Law for power systems)
    for l in model.LINES:
        from_bus = net.line.from_bus[l]
        to_bus = net.line.to_bus[l]
        # V_j^2 ≈ V_i^2 - 2 * (R_ij * P_ij + X_ij * Q_ij)
        model.constrs.add(model.v_sqr[to_bus] == model.v_sqr[from_bus] - 2 * (
                    model.R[l] * model.p_flow[l] + model.X[l] * model.q_flow[l]))

    # 约束3: 线路热稳定约束 (近似)
    for l in model.LINES:
        # I^2 = (P^2 + Q^2) / V^2 <= I_max^2
        # 这是一个非线性约束，我们将其线性化为 P <= I_max * V_base 和 Q <= I_max * V_base
        # (假设V约等于1.0 p.u.)
        model.constrs.add(model.p_flow[l] <= model.I_max_pu[l])
        model.constrs.add(model.p_flow[l] >= -model.I_max_pu[l])
        model.constrs.add(model.q_flow[l] <= model.I_max_pu[l])
        model.constrs.add(model.q_flow[l] >= -model.I_max_pu[l])

    # 约束4: 发电机/电网连接点设置
    gen_buses = net.gen.bus.tolist()
    ext_grid_buses = net.ext_grid.bus.tolist()

    # 只有发电机或外部电网连接点才能注入功率
    for b in model.BUSES:
        if b not in gen_buses and b not in ext_grid_buses:
            model.p_gen[b].fix(0)
            model.q_gen[b].fix(0)

    # 参考节点 (slack bus) 电压固定为1.0 p.u.
    if ext_grid_buses:
        slack_bus = ext_grid_buses[0]
        model.v_sqr[slack_bus].fix(1.0)

    print("电网潮流模型构建完成。")
    return model