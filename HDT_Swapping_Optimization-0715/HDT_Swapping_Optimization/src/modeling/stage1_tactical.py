# HDT_Swapping_Optimization/src/modeling/stage1_tactical.py

from pyomo.environ import *
from src.common import config_final as config

def create_tactical_model(data):
    """
    创建第一阶段(Stage 1)的战术规划模型。
    """
    print("开始构建【第一阶段：战术规划】模型...")
    model = ConcreteModel(name="Tactical_Fleet_Allocation")

    depots_in_data = {v['depot_id'] for v in data['vehicles'].values()}
    model.DEPOTS = Set(initialize=list(depots_in_data))
    model.TASKS = Set(initialize=data['tasks'].keys())

    model.n_trucks = Var(model.DEPOTS, within=NonNegativeReals)
    model.is_task_served = Var(model.TASKS, within=Reals, bounds=(0, 1))

    model.constrs = ConstraintList()

    total_fleet_size = len(data['vehicles'])
    model.constrs.add(sum(model.n_trucks[d] for d in model.DEPOTS) <= total_fleet_size)

    tasks_by_depot = {d: [] for d in model.DEPOTS}
    for t_id, t_info in data['tasks'].items():
        if t_info['depot'] in tasks_by_depot:
            tasks_by_depot[t_info['depot']].append(t_id)

    for d in model.DEPOTS:
        served_tasks_in_depot = sum(model.is_task_served[t] for t in tasks_by_depot[d])
        capacity_of_depot = model.n_trucks[d] * config.AVG_TASKS_PER_TRUCK
        model.constrs.add(capacity_of_depot >= served_tasks_in_depot)

    def objective_rule(m):
        operating_cost = config.MANPOWER_COST_PER_HOUR * 10 * sum(m.n_trucks[d] for d in m.DEPOTS)
        unassigned_penalty = config.UNASSIGNED_TASK_PENALTY * sum(1 - m.is_task_served[t] for t in m.TASKS)
        binary_penalty = 1e-2 * sum(m.is_task_served[t] * (1 - m.is_task_served[t]) for t in m.TASKS)
        return operating_cost + unassigned_penalty + binary_penalty

    model.objective = Objective(rule=objective_rule, sense=minimize)

    print("【第一阶段】模型构建完成。")
    return model