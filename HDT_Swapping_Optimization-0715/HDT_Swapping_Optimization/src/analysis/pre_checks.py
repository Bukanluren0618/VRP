import pandas as pd
from src.common import config_final as config


def check_task_feasibility(data):
    """\
    打印每个任务的往返距离、耗时和理论耗电量，并检查是否在单次行程内可行。
    返回一个DataFrame以供进一步分析或可视化。
    """
    records = []
    for t_id, t_info in data['tasks'].items():
        depot = t_info['depot']
        dest = t_info['delivery_to']
        demand = t_info['demand']
        due_time = t_info['due_time']

        dist_one = data['dist_matrix'].loc[depot, dest]
        time_one = data['time_matrix'].loc[depot, dest]
        round_dist = dist_one * 2
        round_time = time_one * 2 + config.LOADING_UNLOADING_TIME_HOURS

        weight_out = config.HDT_EMPTY_WEIGHT_TON + demand
        weight_back = config.HDT_EMPTY_WEIGHT_TON
        e_out = dist_one * (config.HDT_BASE_CONSUMPTION_KWH_PER_KM +
                            weight_out * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
        e_back = dist_one * (config.HDT_BASE_CONSUMPTION_KWH_PER_KM +
                             weight_back * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
        energy = e_out + e_back

        feasible_time = time_one <= due_time
        feasible_energy = energy <= config.HDT_BATTERY_CAPACITY_KWH - config.HDT_MIN_SOC_KWH
        feasible = feasible_time and feasible_energy

        records.append({
            '任务': t_id,
            '往返距离(km)': round_dist,
            '往返耗时(h)': round_time,
            '估算耗电量(kWh)': energy,
            '满足截止时间': feasible_time,
            '电量足够': feasible_energy,
            '整体可行': feasible
        })

    df = pd.DataFrame(records)
    print("\n========== 任务可达性检查 ==========")
    print(df.to_string(index=False))
    print("====================================\n")
    return df