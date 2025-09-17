# HDT_Swapping_Optimization/src/modeling/stage1_tactical.py

from pyomo.environ import *
from src.common import config_final as config

def create_tactical_model(data):
    """
    创建第一阶段(Stage 1)的战术规划模型。
    """
    print("开始构建【第一阶段：战术规划】模型...")
    model = ConcreteModel(name="Tactical_Fleet_Allocation")

    model.DEPOTS = Set(initialize=[name for name, info in data['locations'].items() if info['type'] == 'Depot'])
    model.TASKS = Set(initialize=data['tasks'].keys())

    model.n_trucks = Var(model.DEPOTS, within=NonNegativeIntegers)
    model.is_task_served = Var(model.TASKS, within=Binary)

    model.constrs = ConstraintList()

    total_fleet_size = len(data['vehicles'])
    model.constrs.add(sum(model.n_trucks[d] for d in model.DEPOTS) <= total_fleet_size)

    tasks_by_depot = {d: [] for d in model.DEPOTS}
    for t_id, t_info in data['tasks'].items():
        if t_info['depot'] in tasks_by_depot:
            tasks_by_depot[t_info['depot']].append(t_id)

    # --- [代码修改] ---
    # 使用从config文件中导入的参数，而不是硬编码
    for d in model.DEPOTS:
        # 约束1：任务数量容量
        served_tasks_in_depot = sum(model.is_task_served[t] for t in tasks_by_depot.get(d, []))
        capacity_of_depot_by_count = model.n_trucks[d] * config.AVG_TASKS_PER_TRUCK
        model.constrs.add(capacity_of_depot_by_count >= served_tasks_in_depot)

        # 约束2：任务预估时间容量
        total_estimated_time_in_depot = sum(
            model.is_task_served[t] * data['tasks'][t]['estimated_duration']
            for t in tasks_by_depot.get(d, [])
        )
        # 使用config中的硬性工作时间上限
        capacity_of_depot_by_time = model.n_trucks[d] * config.MAX_WORKING_HOURS_PER_TRUCK
        model.constrs.add(capacity_of_depot_by_time >= total_estimated_time_in_depot)


    def objective_rule(m):
        operating_cost = config.MANPOWER_COST_PER_HOUR * config.MAX_WORKING_HOURS_PER_TRUCK * sum(m.n_trucks[d] for d in m.DEPOTS)
        unassigned_penalty = config.UNASSIGNED_TASK_PENALTY * sum(1 - m.is_task_served[t] for t in m.TASKS)
        return operating_cost + unassigned_penalty

    model.objective = Objective(rule=objective_rule, sense=minimize)

    print("【第一阶段】模型构建完成。")
    return model