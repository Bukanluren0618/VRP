# generate_data.py
# 生成 data.pkl：时间长度固定 1000，其余数量（仓库/站/客户/车辆等）依据 src/common/config_final.py
# 用法：与 algo_gurobi.py 同目录，运行本文件后生成 data.pkl

import os
import sys
import math
import pickle
import numpy as np
import pandas as pd

# ---- 1) 解析配置：优先 src/common/config_final.py，其次 config.py ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [BASE_DIR, os.path.join(BASE_DIR, "src")]
for p in CANDIDATE_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.common import config_final as cfg
except Exception:
    try:
        from common import config_final as cfg
    except Exception:
        import config as cfg  # 兜底（如果你项目里是 config.py）

# ---- 2) 固定时间长度为 1000；其余数量读取 config_final ----
TOTAL_STEPS = 96  # 固定 1000
TIME_STEP_HOURS = float(getattr(cfg, "TIME_STEP_HOURS", 0.25))

NUM_DEPOTS    = int(getattr(cfg, "NUM_DEPOTS", 1))
NUM_STATIONS  = int(getattr(cfg, "NUM_STATIONS", 1))
NUM_CUSTOMERS = int(getattr(cfg, "NUM_CUSTOMERS", 1))
NUM_TRUCKS    = int(getattr(cfg, "NUM_TRUCKS", 1))

HDT_BATTERY_CAPACITY_KWH = float(getattr(cfg, "HDT_BATTERY_CAPACITY_KWH", 200.0))
LOADING_UNLOADING_TIME_HOURS = float(getattr(cfg, "LOADING_UNLOADING_TIME_HOURS", 0.5))

VOLTAGE_MIN = float(getattr(cfg, "VOLTAGE_MIN", 0.95))
VOLTAGE_MAX = float(getattr(cfg, "VOLTAGE_MAX", 1.05))
PV_PEAK_POWER_KW = float(getattr(cfg, "PV_PEAK_POWER_KW", 400.0))

np.random.seed(42)

# ---- 3) 基本节点集合 ----
depots    = [f"Depot_{i+1}"    for i in range(NUM_DEPOTS)]
stations  = [f"Station_{i+1}"  for i in range(NUM_STATIONS)]
customers = [f"Customer_{i+1}" for i in range(NUM_CUSTOMERS)]
all_nodes = depots + stations + customers

# ---- 4) 随机几何点 -> 对称欧式距离矩阵（单位：km），平均速度 40km/h -> 时间矩阵（小时）
coords = {n: (float(np.random.uniform(0, 50)), float(np.random.uniform(0, 50))) for n in all_nodes}

def euclidean_km(a, b):
    ax, ay = coords[a]; bx, by = coords[b]
    return float(math.hypot(ax - bx, ay - by))

dist_matrix = {
    i: {j: (0.0 if i == j else euclidean_km(i, j)) for j in all_nodes}
    for i in all_nodes
}
AVG_SPEED_KMH = 40.0
time_matrix = {i: {j: dist_matrix[i][j] / AVG_SPEED_KMH for j in all_nodes} for i in all_nodes}

# ---- 5) locations 描述 ----
locations = {}
for d in depots:
    locations[d] = {"type": "Depot", "node_id": d}
for s in stations:
    locations[s] = {"type": "SwapStation", "node_id": s}
for c in customers:
    locations[c] = {"type": "Customer", "node_id": c}

# ---- 6) 车辆：初始 SOC 取电池容量，所属仓库循环分配 ----
vehicles = {
    f"HDT_{i+1}": {"initial_soc": HDT_BATTERY_CAPACITY_KWH, "depot_id": depots[i % NUM_DEPOTS]}
    for i in range(NUM_TRUCKS)
}

# ---- 7) 任务：为每个客户生成一个任务，选择其最近仓库，设置需求与截止时间 ----
def nearest_depot_of(cust):
    return min(depots, key=lambda d: time_matrix[d][cust])

tasks = {}
for i, c in enumerate(customers, 1):
    d = nearest_depot_of(c)
    one_way = time_matrix[d][c]
    due = round(one_way + LOADING_UNLOADING_TIME_HOURS + float(np.random.uniform(1.0, 4.0)), 3)
    tasks[f"Task_{i}"] = {
        "delivery_to": c,
        "demand": round(float(np.random.uniform(1.0, 3.0)), 3),
        "due_time": due,
        "depot": d,
    }

# ---- 8) 站点信息及映射 ----
stations_info = {s: {"initial_full": 10, "initial_empty": 5, "bus_id": i+1} for i, s in enumerate(stations)}
station_to_bus_map = {s: info["bus_id"] for s, info in stations_info.items()}

# ---- 9) 时间轴 / 电价 / PV / 电压前驱（长度固定 1000）----
time_steps = list(range(TOTAL_STEPS))

# 电价（简单平滑序列，避免 0）：可替换成你的分时规则
electricity_prices = pd.Series(
    0.9 + 0.3 * np.sin(np.linspace(0, 10 * math.pi, TOTAL_STEPS)),
    index=time_steps
)

# PV：日照样式（非负），峰值不超过 PV_PEAK_POWER_KW
pv_generation = {}
for s in stations:
    base = np.maximum(
        0.0, np.sin((np.array(time_steps) * TIME_STEP_HOURS - 6.0) / 12.0 * math.pi)
    ) ** 1.5
    # 正常化到峰值
    pv = base / (base.max() + 1e-9) * PV_PEAK_POWER_KW
    # 轻微噪声
    noise = np.clip(np.random.normal(0, 0.03, size=TOTAL_STEPS), -0.1, 0.1)
    pv = np.maximum(0.0, pv * (1.0 + noise))
    pv_generation[s] = [float(x) for x in pv]

# 电压前驱：取在 [VOLTAGE_MIN, VOLTAGE_MAX] 中间的稳定值
v_mid = (VOLTAGE_MIN + VOLTAGE_MAX) / 2.0
voltage_pre = {s: [float(v_mid)] * TOTAL_STEPS for s in stations}

# EV 需求占位（当前模型未用到，可置零）
ev_demand_timestep = {s: [0.0] * TOTAL_STEPS for s in stations}

# ---- 10) 组装 data 并写出 data.pkl ----
data = {
    "traffic_graph": None,
    "locations": locations,
    "tasks": tasks,
    "vehicles": vehicles,
    "stations": stations_info,
    "dist_matrix": dist_matrix,      # dict[str][str] -> float
    "time_matrix": time_matrix,      # dict[str][str] -> float
    "path_matrix": None,
    "power_grid_net": None,
    "station_to_bus_map": station_to_bus_map,
    "time_steps": time_steps,
    "electricity_prices": electricity_prices,         # pandas.Series, index=time_steps
    "pv_generation": pv_generation,                   # dict[station] -> list[float], len=TOTAL_STEPS
    "ev_demand_timestep": ev_demand_timestep,         # dict[station] -> list[float], len=TOTAL_STEPS
    "voltage_pre": voltage_pre,                       # dict[station] -> list[float], len=TOTAL_STEPS
}

OUT_PATH = os.path.join(BASE_DIR, "data.pkl")
with open(OUT_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"✅ 已生成 data.pkl（时间步=1000）")
print(f"   Depots={NUM_DEPOTS}, Stations={NUM_STATIONS}, Customers={NUM_CUSTOMERS}, Trucks={NUM_TRUCKS}")
print(f"📁 保存路径：{OUT_PATH}")

from src.common import config_final as cfg
print("Config read OK:")
print("NUM_DEPOTS =", cfg.NUM_DEPOTS)
print("NUM_STATIONS =", cfg.NUM_STATIONS)
print("NUM_CUSTOMERS =", cfg.NUM_CUSTOMERS)
print("NUM_TRUCKS =", cfg.NUM_TRUCKS)

