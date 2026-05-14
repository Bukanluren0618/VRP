# -*- coding: utf-8 -*-

import os
import sys
import math
import pickle
import random
import itertools
import builtins
import numpy as np
import pandas as pd
import networkx as nx


# ============================= Safe ASCII print for Windows =============================

_original_print = builtins.print


def safe_print(*args, **kwargs):
    """
    Avoid Windows GBK UnicodeEncodeError.
    """
    safe_args = []

    for arg in args:
        if isinstance(arg, str):
            arg = (
                arg.replace("✅", "[OK]")
                   .replace("📁", "[FILE]")
                   .replace("🖼️", "[FIG]")
                   .replace("⚠️", "[WARN]")
                   .replace("🔌", "[GRID]")
            )
            arg = arg.encode("ascii", errors="replace").decode("ascii")
        safe_args.append(arg)

    _original_print(*safe_args, **kwargs)


builtins.print = safe_print


# ============================= Basic configuration =============================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPORT_CSV_FILES = False
EXPORT_EXCEL_BUNDLE = True

CITY_NODE_COUNT = 80

NUM_CUSTOMERS = 60
NUM_UNIQUE_CUSTOMER_NODES = 40
NUM_DUPLICATED_CUSTOMERS = NUM_CUSTOMERS - NUM_UNIQUE_CUSTOMER_NODES

CITY_SCALE_KM = 45.0
AVG_SPEED_KMH = 40.0


# ============================= Task demand / unloading amount parameters =============================

# Here, task demand is treated as unloading amount.
DEMAND_SMALL_RANGE = (0.5, 2.0)
DEMAND_MEDIUM_RANGE = (2.0, 5.0)
DEMAND_LARGE_RANGE = (5.0, 8.0)

DEMAND_CLASS_PROBS = {
    "small": 0.35,
    "medium": 0.45,
    "large": 0.20
}


# ============================= Road network parameters =============================

CENTER = (0.5, 0.5)
RADIAL_POWER = 0.85
CENTER_JITTER = 0.055

K_CENTER = 5
K_EDGE = 4
K_FLOOR = 3

ADD_SHORTCUTS = True
SHORTCUT_PROB = 0.03
SHORTCUT_MAX_PER_NODE = 2


PLOT_NETWORK = True
LABEL_EVERY_K = 1
LABEL_FONTSIZE = 7.0


# ============================= BPR parameters =============================

BPR_ALPHA = 0.15
BPR_BETA = 4.0

CAPACITY_LOCAL = 600.0
CAPACITY_ARTERIAL = 1200.0
CAPACITY_SHORTCUT = 1500.0

FREE_SPEED_LOCAL = 40.0
FREE_SPEED_ARTERIAL = 55.0
FREE_SPEED_SHORTCUT = 65.0

BASE_OD_DEMAND_VEH_H = 8.0
MAX_K_SHORTEST_PATHS = 3
OD_MAX_DISTANCE_KM = 18.0
OD_TOPK_PER_ORIGIN = 20

USE_DIRECTED_ARCS = True

ENABLE_CONGESTION_EVENT = True
CONGESTION_CENTER = (0.5, 0.5)
CONGESTION_RADIUS_KM = 6.0
CONGESTION_CAPACITY_FACTOR = 0.35
CONGESTION_TIME_PENALTY = 3.0

FLEET_SIZE_EHDT = 0
IESS_TARGET_COORDS = [(0.22, 0.22), (0.78, 0.78)]
EVCS_TARGET_COORDS = [(0.20, 0.80), (0.80, 0.20)]


# ============================= Relaxed task time-window parameters =============================

PLANNING_HORIZON_H = 24.0

SERVICE_TIME_MIN_H = 0.25
SERVICE_TIME_MAX_H = 0.75


# ============================= Basic utility functions =============================

def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def round_up_to_step(x, step=0.25):
    return float(np.ceil(float(x) / step) * step)


def round_down_to_step(x, step=0.25):
    return float(np.floor(float(x) / step) * step)


def sample_center_to_sparse_positions(
    n,
    center=(0.5, 0.5),
    radial_power=0.85,
    jitter=0.055,
    seed=42
):
    rng = np.random.default_rng(seed)

    cx, cy = center

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(0.0, 1.0, size=n)

    r = np.power(u, radial_power)
    r = 0.48 * r

    x = cx + r * np.cos(theta) + rng.normal(0.0, jitter, size=n)
    y = cy + r * np.sin(theta) + rng.normal(0.0, jitter, size=n)

    x = np.array([_clip01(v) for v in x], dtype=float)
    y = np.array([_clip01(v) for v in y], dtype=float)

    pos = {
        i: (float(x[i]), float(y[i]))
        for i in range(n)
    }

    return pos


def radius_from_center(pos, center=(0.5, 0.5)):
    cx, cy = center

    return {
        i: float(math.hypot(x - cx, y - cy))
        for i, (x, y) in pos.items()
    }


def k_by_radius(r, k_center=5, k_edge=4, k_floor=3, r_max=0.48):
    t = min(1.0, max(0.0, r / (r_max + 1e-12)))

    k = (1.0 - t) * k_center + t * k_edge
    k = int(round(k))

    return max(k_floor, k)


def build_knn_graph(
    pos,
    center=(0.5, 0.5),
    k_center=5,
    k_edge=4,
    k_floor=3,
    add_shortcuts=True,
    shortcut_prob=0.03,
    shortcut_max_per_node=2,
    seed=42
):
    rng = np.random.default_rng(seed)

    nodes = list(pos.keys())
    n = len(nodes)

    XY = np.array([pos[i] for i in nodes], dtype=float)

    dx = XY[:, 0:1] - XY[:, 0:1].T
    dy = XY[:, 1:2] - XY[:, 1:2].T

    D2 = dx * dx + dy * dy
    np.fill_diagonal(D2, np.inf)

    rmap = radius_from_center(pos, center=center)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i in nodes:
        x, y = pos[i]

        G.nodes[i]["x"] = float(x)
        G.nodes[i]["y"] = float(y)
        G.nodes[i]["r"] = float(rmap[i])

    for idx_i, i in enumerate(nodes):
        k_i = k_by_radius(
            rmap[i],
            k_center=k_center,
            k_edge=k_edge,
            k_floor=k_floor
        )

        nn_idx = np.argpartition(D2[idx_i], kth=min(k_i, n - 2))[:k_i]

        for jdx in nn_idx:
            j = nodes[int(jdx)]

            if i == j:
                continue

            G.add_edge(i, j, kind="knn")

    if add_shortcuts and shortcut_prob > 0:
        for i in nodes:
            if rng.random() > shortcut_prob:
                continue

            candidates = rng.choice(nodes, size=min(25, n), replace=False)

            best = None
            best_d2 = -1.0

            for j in candidates:
                if i == j or G.has_edge(i, j):
                    continue

                xi, yi = pos[i]
                xj, yj = pos[j]

                d2 = (xi - xj) ** 2 + (yi - yj) ** 2

                if d2 > best_d2:
                    best_d2 = d2
                    best = int(j)

            if best is not None:
                G.add_edge(i, best, kind="shortcut")

        if shortcut_max_per_node is not None:
            for i in nodes:
                shortcut_edges = []

                for j in list(G.neighbors(i)):
                    if G[i][j].get("kind") == "shortcut":
                        xi, yi = pos[i]
                        xj, yj = pos[j]

                        d2 = (xi - xj) ** 2 + (yi - yj) ** 2
                        shortcut_edges.append((d2, j))

                if len(shortcut_edges) > shortcut_max_per_node:
                    shortcut_edges.sort()
                    remove_edges = shortcut_edges[shortcut_max_per_node:]

                    for _, j in remove_edges:
                        if G.has_edge(i, j) and G[i][j].get("kind") == "shortcut":
                            G.remove_edge(i, j)

    for u, v, ed in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        d_unit = math.hypot(x1 - x2, y1 - y2)
        d_km = max(1e-6, d_unit * CITY_SCALE_KM)

        ed["distance"] = max(1e-6, d_unit)
        ed["length_km"] = d_km

    return G


# ============================= BPR arc table =============================

def get_free_speed_capacity(edge_kind, length_km):
    if edge_kind == "shortcut":
        road_class = "shortcut"
        free_speed = FREE_SPEED_SHORTCUT
        capacity = CAPACITY_SHORTCUT
    else:
        if length_km >= 2.5:
            road_class = "arterial"
            free_speed = FREE_SPEED_ARTERIAL
            capacity = CAPACITY_ARTERIAL
        else:
            road_class = "local"
            free_speed = FREE_SPEED_LOCAL
            capacity = CAPACITY_LOCAL

    return road_class, free_speed, capacity


def build_bpr_arc_table(
    G,
    directed=True,
    alpha=BPR_ALPHA,
    beta=BPR_BETA
):
    rows = []
    arc_id = 0

    for u, v, ed in G.edges(data=True):
        length_km = float(ed["length_km"])
        edge_kind = ed.get("kind", "knn")

        road_class, free_speed_kmh, capacity_veh_h = get_free_speed_capacity(
            edge_kind=edge_kind,
            length_km=length_km
        )

        t0_h = length_km / max(free_speed_kmh, 1e-9)

        directions = [(u, v)]
        if directed:
            directions.append((v, u))

        for from_node, to_node in directions:
            rows.append({
                "arc_id": int(arc_id),
                "from_node": int(from_node),
                "to_node": int(to_node),
                "length_km": float(length_km),
                "t0_h": float(t0_h),
                "capacity_veh_h": float(capacity_veh_h),
                "alpha": float(alpha),
                "beta": float(beta),
                "free_speed_kmh": float(free_speed_kmh),
                "edge_kind": str(edge_kind),
                "road_class": str(road_class),
                "is_congested": False,
                "congestion_time_penalty": 1.0
            })

            arc_id += 1

    return pd.DataFrame(rows)


def apply_congestion_event_to_arcs(
    arc_df,
    pos,
    center=CONGESTION_CENTER,
    radius_km=CONGESTION_RADIUS_KM,
    capacity_factor=CONGESTION_CAPACITY_FACTOR,
    time_penalty=CONGESTION_TIME_PENALTY
):
    arc_df = arc_df.copy()

    cx, cy = center
    congested_arc_ids = []

    for _, row in arc_df.iterrows():
        u = int(row["from_node"])
        v = int(row["to_node"])

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        d_km = math.hypot(mx - cx, my - cy) * CITY_SCALE_KM

        if d_km <= radius_km:
            congested_arc_ids.append(int(row["arc_id"]))

    if len(congested_arc_ids) > 0:
        mask = arc_df["arc_id"].isin(congested_arc_ids)

        arc_df.loc[mask, "is_congested"] = True
        arc_df.loc[mask, "capacity_veh_h"] = (
            arc_df.loc[mask, "capacity_veh_h"] * capacity_factor
        )
        arc_df.loc[mask, "congestion_time_penalty"] = float(time_penalty)

    return arc_df, congested_arc_ids


def build_directed_graph_from_arcs(arc_df, avoid_congestion=False):
    DG = nx.DiGraph()

    for _, row in arc_df.iterrows():
        u = int(row["from_node"])
        v = int(row["to_node"])

        if avoid_congestion:
            weight = float(row["t0_h"]) * float(row["congestion_time_penalty"])
        else:
            weight = float(row["t0_h"])

        DG.add_edge(
            u,
            v,
            weight=weight,
            arc_id=int(row["arc_id"]),
            length_km=float(row["length_km"]),
            t0_h=float(row["t0_h"]),
            capacity_veh_h=float(row["capacity_veh_h"]),
            is_congested=bool(row["is_congested"])
        )

    return DG


# ============================= Customer and task generation =============================

def get_largest_component_nodes(G):
    components = list(nx.connected_components(G))
    components = sorted(components, key=len, reverse=True)
    return sorted(list(components[0]))


def build_customers_with_duplicates(
    G,
    num_customers=60,
    num_unique_customer_nodes=40,
    seed=42
):
    rng = np.random.default_rng(seed)

    largest_nodes = get_largest_component_nodes(G)

    if len(largest_nodes) < num_unique_customer_nodes:
        raise ValueError(
            f"Largest component has only {len(largest_nodes)} nodes, "
            f"less than required unique customer nodes={num_unique_customer_nodes}. "
            f"Increase K_CENTER/K_EDGE."
        )

    unique_nodes = rng.choice(
        largest_nodes,
        size=num_unique_customer_nodes,
        replace=False
    ).astype(int).tolist()

    duplicate_base_nodes = rng.choice(
        unique_nodes,
        size=num_customers - num_unique_customer_nodes,
        replace=True
    ).astype(int).tolist()

    customer_nodes = unique_nodes + duplicate_base_nodes

    rng.shuffle(customer_nodes)

    rows = []

    for cid, node_id in enumerate(customer_nodes):
        node_id = int(node_id)

        rows.append({
            "customer_id": f"Customer_{cid + 1}",
            "customer_index": int(cid),
            "node_id": int(node_id)
        })

    customers_df = pd.DataFrame(rows)

    multiplicity = customers_df.groupby("node_id").size().reset_index()
    multiplicity.columns = ["node_id", "customer_count_on_node"]

    customers_df = customers_df.merge(
        multiplicity,
        on="node_id",
        how="left"
    )

    return customers_df


def assign_task_demand(
    customers_df,
    seed=42
):
    """
    Assign task demand as unloading amount.

    Output:
        demand_df with columns:
        - customer_id
        - customer_index
        - node_id
        - demand_class
        - demand
    """
    rng = np.random.default_rng(seed)

    demand_classes = ["small", "medium", "large"]

    probs = np.array([
        DEMAND_CLASS_PROBS["small"],
        DEMAND_CLASS_PROBS["medium"],
        DEMAND_CLASS_PROBS["large"]
    ], dtype=float)

    probs = probs / probs.sum()

    rows = []

    for _, row in customers_df.iterrows():
        demand_class = rng.choice(demand_classes, p=probs)

        if demand_class == "small":
            low, high = DEMAND_SMALL_RANGE
        elif demand_class == "medium":
            low, high = DEMAND_MEDIUM_RANGE
        else:
            low, high = DEMAND_LARGE_RANGE

        demand = rng.uniform(low, high)

        rows.append({
            "customer_id": row["customer_id"],
            "customer_index": int(row["customer_index"]),
            "node_id": int(row["node_id"]),
            "demand_class": str(demand_class),
            "demand": round(float(demand), 3)
        })

    demand_df = pd.DataFrame(rows)

    return demand_df


def build_tasks_with_time_windows(
    customers_df,
    seed=42
):
    """
    Deadline-based relaxed task windows with within-class variations.

    All start times and deadlines are aligned to 15-minute intervals:
        15 min = 0.25 h

    Demand is treated as unloading amount.

    Rounding rule:
    - deadline_h: round up to nearest 0.25 h
    - start_time_h: round down to nearest 0.25 h

    Model usage:
    if has_start_constraint:
        arrival_i >= start_time_h

    always:
        arrival_i + service_time_i <= deadline_i
    """
    rng = np.random.default_rng(seed)

    demand_df = assign_task_demand(
        customers_df=customers_df,
        seed=seed + 100
    )

    demand_lookup = demand_df.set_index("customer_id").to_dict(orient="index")

    time_window_types = [
        "all_day",
        "business_hours",
        "before_20",
        "before_18",
        "before_12",
        "afternoon_or_evening"
    ]

    probs = np.array([
        0.20,
        0.24,
        0.22,
        0.17,
        0.08,
        0.09
    ], dtype=float)

    probs = probs / probs.sum()

    rows = []

    for _, row in customers_df.iterrows():
        tw_type = rng.choice(time_window_types, p=probs)

        if tw_type == "all_day":
            has_start_constraint = False
            start_time_h = np.nan
            ready_time_h = 0.0
            deadline_h = rng.uniform(22.5, 24.0)
            window_note = "All-day service with relaxed late deadline"

        elif tw_type == "business_hours":
            has_start_constraint = True
            start_time_h = rng.uniform(7.5, 9.0)
            deadline_h = rng.uniform(17.0, 19.0)

            if deadline_h - start_time_h < 6.0:
                deadline_h = min(19.0, start_time_h + rng.uniform(6.0, 9.0))

            ready_time_h = start_time_h
            window_note = "Service during approximate business hours"

        elif tw_type == "before_20":
            has_start_constraint = False
            start_time_h = np.nan
            ready_time_h = 0.0
            deadline_h = rng.uniform(19.0, 21.0)
            window_note = "Deadline around evening before 20:00"

        elif tw_type == "before_18":
            has_start_constraint = False
            start_time_h = np.nan
            ready_time_h = 0.0
            deadline_h = rng.uniform(17.0, 19.0)
            window_note = "Deadline around late afternoon"

        elif tw_type == "before_12":
            has_start_constraint = False
            start_time_h = np.nan
            ready_time_h = 0.0
            deadline_h = rng.uniform(11.0, 13.0)
            window_note = "Deadline around noon"

        elif tw_type == "afternoon_or_evening":
            has_start_constraint = True
            start_time_h = rng.uniform(11.5, 13.5)
            deadline_h = rng.uniform(21.0, 23.5)

            if deadline_h - start_time_h < 6.0:
                deadline_h = min(23.5, start_time_h + rng.uniform(6.0, 9.5))

            ready_time_h = start_time_h
            window_note = "Service after noon with evening deadline"

        else:
            has_start_constraint = False
            start_time_h = np.nan
            ready_time_h = 0.0
            deadline_h = rng.uniform(22.5, 24.0)
            window_note = "Default relaxed deadline"

        deadline_h = float(np.clip(deadline_h, 0.25, PLANNING_HORIZON_H))
        deadline_h = round_up_to_step(deadline_h, step=0.25)
        deadline_h = min(deadline_h, PLANNING_HORIZON_H)

        if has_start_constraint:
            start_time_h = float(np.clip(start_time_h, 0.0, deadline_h - 0.5))
            start_time_h = round_down_to_step(start_time_h, step=0.25)

            if deadline_h - start_time_h < 0.5:
                start_time_h = max(0.0, deadline_h - 0.5)
                start_time_h = round_down_to_step(start_time_h, step=0.25)

            ready_time_h = start_time_h
        else:
            start_time_h = np.nan
            ready_time_h = 0.0

        service_time = rng.uniform(
            SERVICE_TIME_MIN_H,
            SERVICE_TIME_MAX_H
        )

        service_time = round_up_to_step(service_time, step=0.25)

        demand_info = demand_lookup[row["customer_id"]]
        demand_class = demand_info["demand_class"]
        demand = float(demand_info["demand"])

        rows.append({
            "task_id": f"Task_{int(row['customer_index']) + 1}",
            "customer_id": row["customer_id"],
            "customer_index": int(row["customer_index"]),
            "node_id": int(row["node_id"]),

            "time_window_type": str(tw_type),
            "has_start_constraint": bool(has_start_constraint),
            "start_time_h": None if pd.isna(start_time_h) else round(float(start_time_h), 3),
            "deadline_h": round(float(deadline_h), 3),
            "window_note": str(window_note),

            "ready_time_h": round(float(ready_time_h), 3),
            "due_time_h": round(float(deadline_h), 3),
            "time_window_width_h": round(float(deadline_h - ready_time_h), 3),

            "service_time_h": round(float(service_time), 3),

            "demand_class": str(demand_class),
            "demand": round(float(demand), 3)
        })

    tasks_df = pd.DataFrame(rows)

    tasks_df = tasks_df.sort_values(
        by=["deadline_h", "has_start_constraint", "customer_index"]
    ).reset_index(drop=True)

    return tasks_df


# ============================= OD generation =============================

def build_od_pairs_from_customers(
    customers_df,
    tasks_df,
    pos,
    city_scale_km=CITY_SCALE_KM,
    base_demand_veh_h=BASE_OD_DEMAND_VEH_H,
    od_max_distance_km=OD_MAX_DISTANCE_KM,
    od_topk_per_origin=OD_TOPK_PER_ORIGIN
):
    task_lookup = tasks_df.set_index("customer_id").to_dict(orient="index")

    rows = []
    od_id = 0

    customers = customers_df.sort_values("customer_index").to_dict(orient="records")

    for o in customers:
        candidate_destinations = []
        ox, oy = pos[int(o["node_id"])]

        for d in customers:
            if o["customer_id"] == d["customer_id"]:
                continue

            dx, dy = pos[int(d["node_id"])]
            euclid_km = float(math.hypot(ox - dx, oy - dy) * city_scale_km)

            if (od_max_distance_km is not None) and (euclid_km > float(od_max_distance_km)):
                continue

            candidate_destinations.append((euclid_km, d))

        candidate_destinations.sort(key=lambda x: x[0])
        if (od_topk_per_origin is not None) and (od_topk_per_origin > 0):
            candidate_destinations = candidate_destinations[:int(od_topk_per_origin)]

        for euclid_km, d in candidate_destinations:

            o_task = task_lookup[o["customer_id"]]
            d_task = task_lookup[d["customer_id"]]

            task_demand_factor = 0.5 * (float(o_task["demand"]) + float(d_task["demand"]))
            distance_ratio = float(euclid_km) / max(float(od_max_distance_km), 1e-9)
            distance_factor = max(0.35, 1.0 - 0.45 * min(1.0, distance_ratio))
            od_demand_veh_h = max(0.1, float(base_demand_veh_h) * task_demand_factor * distance_factor)


            rows.append({
                "od_id": int(od_id),
                "origin_customer_id": o["customer_id"],
                "destination_customer_id": d["customer_id"],
                "origin_customer_index": int(o["customer_index"]),
                "destination_customer_index": int(d["customer_index"]),
                "origin_node": int(o["node_id"]),
                "destination_node": int(d["node_id"]),
                 "demand_veh_h": round(float(od_demand_veh_h), 3),
                "euclid_distance_km": round(float(euclid_km), 4),

                "origin_time_window_type": o_task["time_window_type"],
                "origin_has_start_constraint": bool(o_task["has_start_constraint"]),
                "origin_start_time_h": o_task["start_time_h"],
                "origin_deadline_h": float(o_task["deadline_h"]),
                "origin_demand_class": o_task["demand_class"],
                "origin_task_demand": float(o_task["demand"]),

                "destination_time_window_type": d_task["time_window_type"],
                "destination_has_start_constraint": bool(d_task["has_start_constraint"]),
                "destination_start_time_h": d_task["start_time_h"],
                "destination_deadline_h": float(d_task["deadline_h"]),
                "destination_demand_class": d_task["demand_class"],
                "destination_task_demand": float(d_task["demand"]),

                "origin_ready_time_h": float(o_task["ready_time_h"]),
                "origin_due_time_h": float(o_task["due_time_h"]),
                "destination_ready_time_h": float(d_task["ready_time_h"]),
                "destination_due_time_h": float(d_task["due_time_h"]),

                "same_physical_node": bool(int(o["node_id"]) == int(d["node_id"])),
                "od_type": "customer_to_customer"
            })

            od_id += 1

    return pd.DataFrame(rows)


# ============================= Candidate path generation =============================

def make_arc_lookup(arc_df):
    arc_by_uv = {}
    arc_len_by_uv = {}
    arc_t0_by_uv = {}

    for _, row in arc_df.iterrows():
        u = int(row["from_node"])
        v = int(row["to_node"])

        arc_by_uv[(u, v)] = int(row["arc_id"])
        arc_len_by_uv[(u, v)] = float(row["length_km"])
        arc_t0_by_uv[(u, v)] = float(row["t0_h"])

    return arc_by_uv, arc_len_by_uv, arc_t0_by_uv


def safe_shortest_path(DG, source, target):
    source = int(source)
    target = int(target)

    if source == target:
        return [source]

    try:
        return nx.shortest_path(
            DG,
            source=source,
            target=target,
            weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def safe_k_shortest_paths(DG, source, target, k=3):
    source = int(source)
    target = int(target)

    if source == target:
        return [[source]]

    try:
        gen = nx.shortest_simple_paths(
            DG,
            source=source,
            target=target,
            weight="weight"
        )

        paths = []

        for p in itertools.islice(gen, k):
            paths.append([int(x) for x in p])

        return paths

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def path_to_arcs(path_nodes, arc_by_uv):
    if path_nodes is None:
        return []

    if len(path_nodes) <= 1:
        return []

    arc_list = []

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        key = (int(u), int(v))

        if key not in arc_by_uv:
            return []

        arc_list.append(int(arc_by_uv[key]))

    return arc_list


def calc_path_length_time(path_nodes, arc_len_by_uv, arc_t0_by_uv):
    if path_nodes is None or len(path_nodes) <= 1:
        return 0.0, 0.0

    total_length = 0.0
    total_t0 = 0.0

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        key = (int(u), int(v))

        total_length += float(arc_len_by_uv.get(key, np.inf))
        total_t0 += float(arc_t0_by_uv.get(key, np.inf))

    return total_length, total_t0


def generate_candidate_paths_for_od(
    od_df,
    arc_df,
    max_k_shortest=MAX_K_SHORTEST_PATHS
):
    arc_by_uv, arc_len_by_uv, arc_t0_by_uv = make_arc_lookup(arc_df)

    DG_free = build_directed_graph_from_arcs(
        arc_df,
        avoid_congestion=False
    )

    DG_avoid = build_directed_graph_from_arcs(
        arc_df,
        avoid_congestion=True
    )

    path_rows = []
    path_arc_rows = []

    global_path_id = 0

    for _, od in od_df.iterrows():
        od_id = int(od["od_id"])
        origin = int(od["origin_node"])
        destination = int(od["destination_node"])

        raw_paths = []

        if origin == destination:
            raw_paths.append(("same_node", [origin]))
        else:
            shortest_path = safe_shortest_path(
                DG_free,
                origin,
                destination
            )

            if shortest_path is not None:
                raw_paths.append(("shortest", shortest_path))

            avoid_path = safe_shortest_path(
                DG_avoid,
                origin,
                destination
            )

            if avoid_path is not None:
                raw_paths.append(("avoid_congestion", avoid_path))

            k_paths = safe_k_shortest_paths(
                DG_free,
                origin,
                destination,
                k=max_k_shortest
            )

            for kk, p in enumerate(k_paths):
                if kk == 0:
                    continue

                raw_paths.append((f"detour_{kk + 1}", p))

        seen = set()
        unique_paths = []

        for path_type, p in raw_paths:
            if p is None:
                continue

            key = tuple(int(x) for x in p)

            if key in seen:
                continue

            seen.add(key)
            unique_paths.append((path_type, [int(x) for x in p]))

        for local_path_id, (path_type, node_path) in enumerate(unique_paths):
            arc_path = path_to_arcs(
                node_path,
                arc_by_uv
            )

            path_length_km, free_flow_time_h = calc_path_length_time(
                node_path,
                arc_len_by_uv,
                arc_t0_by_uv
            )

            if len(arc_path) > 0:
                congested_arc_count = int(
                    arc_df.loc[
                        arc_df["arc_id"].isin(arc_path),
                        "is_congested"
                    ].sum()
                )
            else:
                congested_arc_count = 0

            path_rows.append({
                "path_id": int(global_path_id),
                "od_id": int(od_id),
                "local_path_id": int(local_path_id),
                "path_type": str(path_type),
                "origin_customer_id": od["origin_customer_id"],
                "destination_customer_id": od["destination_customer_id"],
                "origin_node": int(origin),
                "destination_node": int(destination),
                "node_path": node_path,
                "arc_path": arc_path,
                "path_length_km": float(path_length_km),
                "free_flow_time_h": float(free_flow_time_h),
                "congested_arc_count": int(congested_arc_count)
            })

            for arc_id in arc_path:
                path_arc_rows.append({
                    "path_id": int(global_path_id),
                    "od_id": int(od_id),
                    "arc_id": int(arc_id),
                    "value": 1.0
                })

            global_path_id += 1

    path_df = pd.DataFrame(path_rows)
    path_arc_df = pd.DataFrame(path_arc_rows)

    return path_df, path_arc_df


def build_path_arc_matrix(path_df, path_arc_df, arc_df):
    path_ids = list(path_df["path_id"].astype(int))
    arc_ids = list(arc_df["arc_id"].astype(int))

    mat = pd.DataFrame(
        0.0,
        index=path_ids,
        columns=arc_ids
    )

    for _, row in path_arc_df.iterrows():
        p = int(row["path_id"])
        a = int(row["arc_id"])

        if p in mat.index and a in mat.columns:
            mat.loc[p, a] = float(row["value"])

    mat.index.name = "path_id"
    mat.columns.name = "arc_id"

    return mat

def select_fixed_facility_nodes(pos, target_coords):
    selected = []
    used = set()
    for tx, ty in target_coords:
        best_node = None
        best_d2 = float("inf")
        for nid, (x, y) in pos.items():
            if nid in used:
                continue
            d2 = (x - tx) ** 2 + (y - ty) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_node = int(nid)
        if best_node is not None:
            selected.append(best_node)
            used.add(best_node)
    return selected


def export_bundle_to_excel(base_dir, sheets):
    excel_path = os.path.join(base_dir, "vrp_outputs.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                continue
            safe_sheet = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
    return excel_path


def build_od_path_map(path_df):
    od_path_map = {}

    for od_id, sub in path_df.groupby("od_id"):
        od_path_map[int(od_id)] = [
            int(x)
            for x in sub["path_id"].tolist()
        ]

    return od_path_map

def build_path_bundle_for_od(od_df, arc_df):
    path_df, path_arc_df = generate_candidate_paths_for_od(
        od_df=od_df,
        arc_df=arc_df,
        max_k_shortest=MAX_K_SHORTEST_PATHS
    )
    path_arc_matrix = build_path_arc_matrix(
        path_df=path_df,
        path_arc_df=path_arc_df,
        arc_df=arc_df
    )
    od_path_map = build_od_path_map(path_df)
    return path_df, path_arc_df, path_arc_matrix, od_path_map




# ============================= BPR helper functions =============================

def bpr_travel_time(
    t0_h,
    flow_veh_h,
    capacity_veh_h,
    alpha=BPR_ALPHA,
    beta=BPR_BETA
):
    capacity = max(float(capacity_veh_h), 1e-9)
    ratio = max(float(flow_veh_h), 0.0) / capacity

    return float(
        t0_h * (1.0 + alpha * (ratio ** beta))
    )


def compute_link_flow_from_path_flow(
    path_arc_matrix,
    path_flow_series
):
    f = path_flow_series.reindex(path_arc_matrix.index)
    f = f.fillna(0.0).astype(float)

    link_flow = path_arc_matrix.T.dot(f)
    link_flow.name = "flow_veh_h"

    return link_flow


def evaluate_bpr_on_arcs(
    arc_df,
    link_flow_series
):
    out = arc_df.copy()

    flow_values = link_flow_series.reindex(
        out["arc_id"].astype(int)
    ).fillna(0.0).values

    out["flow_veh_h"] = flow_values

    out["bpr_time_h"] = [
        bpr_travel_time(
            t0_h=row["t0_h"],
            flow_veh_h=row["flow_veh_h"],
            capacity_veh_h=row["capacity_veh_h"],
            alpha=row["alpha"],
            beta=row["beta"]
        )
        for _, row in out.iterrows()
    ]

    return out


# ============================= OD matrix construction =============================

def build_od_demand_matrix(customers_df, od_df):
    customer_ids = customers_df.sort_values("customer_index")["customer_id"].tolist()

    mat = pd.DataFrame(
        0.0,
        index=customer_ids,
        columns=customer_ids
    )

    for _, row in od_df.iterrows():
        o = row["origin_customer_id"]
        d = row["destination_customer_id"]
        mat.loc[o, d] = float(row["demand_veh_h"])

    return mat


def build_od_time_matrix(customers_df, path_df):
    customer_ids = customers_df.sort_values("customer_index")["customer_id"].tolist()

    mat = pd.DataFrame(
        np.nan,
        index=customer_ids,
        columns=customer_ids
    )

    shortest_paths = path_df[
        path_df["path_type"].isin(["shortest", "same_node"])
    ].copy()

    for _, row in shortest_paths.iterrows():
        o = row["origin_customer_id"]
        d = row["destination_customer_id"]
        mat.loc[o, d] = float(row["free_flow_time_h"])

    return mat


# ============================= Visualization =============================

def plot_original_network(G, output_path):
    import matplotlib.pyplot as plt

    pos_plot = {
        n: (G.nodes[n]["x"], G.nodes[n]["y"])
        for n in G.nodes()
    }

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal")

    nx.draw_networkx_edges(
        G,
        pos_plot,
        ax=ax,
        edge_color="black",
        alpha=0.25,
        width=0.9
    )

    nx.draw_networkx_nodes(
        G,
        pos_plot,
        ax=ax,
        node_size=45,
        alpha=0.70
    )

    for n, (x, y) in pos_plot.items():
        if LABEL_EVERY_K > 1 and n % LABEL_EVERY_K != 0:
            continue

        ax.text(
            x,
            y,
            str(n),
            fontsize=LABEL_FONTSIZE,
            ha="center",
            va="center",
            alpha=0.95
        )

    ax.set_title(
        "Original 80-node Road Network",
        fontsize=12
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_customer_network(G, customers_df, output_path):
    import matplotlib.pyplot as plt

    pos_plot = {
        n: (G.nodes[n]["x"], G.nodes[n]["y"])
        for n in G.nodes()
    }

    customer_node_counts = customers_df.groupby("node_id").size().to_dict()
    customer_nodes = sorted(customer_node_counts.keys())

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal")

    nx.draw_networkx_edges(
        G,
        pos_plot,
        ax=ax,
        edge_color="black",
        alpha=0.20,
        width=0.9
    )

    nx.draw_networkx_nodes(
        G,
        pos_plot,
        ax=ax,
        node_size=35,
        alpha=0.45
    )

    node_sizes = [
        80 + 45 * customer_node_counts[n]
        for n in customer_nodes
    ]

    nx.draw_networkx_nodes(
        G,
        pos_plot,
        nodelist=customer_nodes,
        node_size=node_sizes,
        alpha=0.90,
        label="Customer nodes"
    )

    for n, (x, y) in pos_plot.items():
        ax.text(
            x,
            y,
            str(n),
            fontsize=LABEL_FONTSIZE,
            ha="center",
            va="center",
            alpha=0.85
        )

    for n in customer_nodes:
        count = customer_node_counts[n]
        x, y = pos_plot[n]

        if count > 1:
            ax.text(
                x,
                y + 0.025,
                f"x{count}",
                fontsize=9,
                ha="center",
                va="center",
                fontweight="bold",
                alpha=0.95
            )

    ax.set_title(
        "Customer Distribution: 60 Customers on 40 Unique Road Nodes",
        fontsize=12
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_od_matrix(od_matrix, output_path, title, cbar_label):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        od_matrix.values,
        aspect="auto"
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Destination customer")
    ax.set_ylabel("Origin customer")

    customer_ids = list(od_matrix.index)

    tick_step = max(1, len(customer_ids) // 12)
    ticks = list(range(0, len(customer_ids), tick_step))

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(
        [customer_ids[i].replace("Customer_", "C") for i in ticks],
        rotation=90,
        fontsize=7
    )

    ax.set_yticklabels(
        [customer_ids[i].replace("Customer_", "C") for i in ticks],
        fontsize=7
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_task_time_windows(tasks_df, output_path):
    import matplotlib.pyplot as plt

    plot_df = tasks_df.sort_values(
        by=["deadline_h", "has_start_constraint", "customer_index"]
    ).reset_index(drop=True)

    fig_height = max(8, 0.18 * len(plot_df))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    y_pos = np.arange(len(plot_df))

    for i, row in plot_df.iterrows():
        has_start = bool(row["has_start_constraint"])
        deadline = float(row["deadline_h"])

        if has_start and pd.notna(row["start_time_h"]):
            left = float(row["start_time_h"])
        else:
            left = 0.0

        width = max(0.0, deadline - left)

        ax.barh(
            y=i,
            width=width,
            left=left,
            height=0.65,
            alpha=0.80
        )

        if has_start:
            ax.plot(
                left,
                i,
                marker="|",
                markersize=8
            )

        ax.plot(
            deadline,
            i,
            marker="|",
            markersize=8
        )

    labels = [
        f"{row['task_id'].replace('Task_', 'T')} "
        f"({row['time_window_type']}, {row['demand_class']}, N{int(row['node_id'])})"
        for _, row in plot_df.iterrows()
    ]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Task")
    ax.set_xlim(0, PLANNING_HORIZON_H)

    ax.set_title(
        "Deadline-based Task Time Windows",
        fontsize=12
    )

    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_customer_node_multiplicity(customers_df, output_path):
    import matplotlib.pyplot as plt

    count_df = customers_df.groupby("node_id").size().reset_index()
    count_df.columns = ["node_id", "customer_count"]
    count_df = count_df.sort_values("node_id")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        count_df["node_id"].astype(str),
        count_df["customer_count"]
    )

    ax.set_xlabel("Road node")
    ax.set_ylabel("Number of customers")
    ax.set_title("Customer Multiplicity on Road Nodes")

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================= Main program =============================

def main():
    pos = sample_center_to_sparse_positions(
        n=CITY_NODE_COUNT,
        center=CENTER,
        radial_power=RADIAL_POWER,
        jitter=CENTER_JITTER,
        seed=SEED
    )

    G = build_knn_graph(
        pos=pos,
        center=CENTER,
        k_center=K_CENTER,
        k_edge=K_EDGE,
        k_floor=K_FLOOR,
        add_shortcuts=ADD_SHORTCUTS,
        shortcut_prob=SHORTCUT_PROB,
        shortcut_max_per_node=SHORTCUT_MAX_PER_NODE,
        seed=SEED + 7
    )

    degs = np.array([d for _, d in G.degree()], dtype=float)

    print(
        f"[OK] Road network generated: "
        f"nodes={G.number_of_nodes()}, "
        f"edges={G.number_of_edges()}, "
        f"connected={nx.is_connected(G)}, "
        f"components={nx.number_connected_components(G)}, "
        f"avg_degree={degs.mean():.3f}, "
        f"max_degree={int(degs.max())}"
    )

    rmap = radius_from_center(pos, center=CENTER)

    nodes_df = pd.DataFrame({
        "node_id": list(pos.keys()),
        "x": [pos[i][0] for i in pos.keys()],
        "y": [pos[i][1] for i in pos.keys()],
        "r_km": [rmap[i] * CITY_SCALE_KM for i in pos.keys()],
        "degree": [G.degree(i) for i in pos.keys()],
        "component_id": -1
    })

    for comp_id, comp in enumerate(nx.connected_components(G)):
        nodes_df.loc[
            nodes_df["node_id"].isin(list(comp)),
            "component_id"
        ] = comp_id

    nodes_csv = os.path.join(BASE_DIR, "nodes_xy.csv")

    if EXPORT_CSV_FILES:
        nodes_df.to_csv(nodes_csv, index=False, encoding="utf-8-sig")

    print("[FILE] nodes:", nodes_csv)

    graph_path = os.path.join(BASE_DIR, "traffic_graph_only.pkl")

    with open(graph_path, "wb") as f:
        pickle.dump(
            {
                "traffic_graph": G,
                "pos": pos,
                "params": {
                    "CITY_NODE_COUNT": CITY_NODE_COUNT,
                    "NUM_CUSTOMERS": NUM_CUSTOMERS,
                    "NUM_UNIQUE_CUSTOMER_NODES": NUM_UNIQUE_CUSTOMER_NODES,
                    "NUM_DUPLICATED_CUSTOMERS": NUM_DUPLICATED_CUSTOMERS,
                    "CITY_SCALE_KM": CITY_SCALE_KM,
                    "CENTER": CENTER,
                    "RADIAL_POWER": RADIAL_POWER,
                    "CENTER_JITTER": CENTER_JITTER,
                    "K_CENTER": K_CENTER,
                    "K_EDGE": K_EDGE,
                    "K_FLOOR": K_FLOOR,
                    "ADD_SHORTCUTS": ADD_SHORTCUTS,
                    "SHORTCUT_PROB": SHORTCUT_PROB,
                    "SHORTCUT_MAX_PER_NODE": SHORTCUT_MAX_PER_NODE,
                    "OD_MAX_DISTANCE_KM": OD_MAX_DISTANCE_KM,
                    "OD_TOPK_PER_ORIGIN": OD_TOPK_PER_ORIGIN,
                    "SEED": SEED
                }
            },
            f
        )

    print("[FILE] graph:", graph_path)

    arc_df = build_bpr_arc_table(
        G=G,
        directed=USE_DIRECTED_ARCS,
        alpha=BPR_ALPHA,
        beta=BPR_BETA
    )

    if ENABLE_CONGESTION_EVENT:
        arc_df, congested_arc_ids = apply_congestion_event_to_arcs(
            arc_df=arc_df,
            pos=pos,
            center=CONGESTION_CENTER,
            radius_km=CONGESTION_RADIUS_KM,
            capacity_factor=CONGESTION_CAPACITY_FACTOR,
            time_penalty=CONGESTION_TIME_PENALTY
        )
    else:
        congested_arc_ids = []

    arcs_csv = os.path.join(BASE_DIR, "arcs_bpr.csv")

    if EXPORT_CSV_FILES:
        arc_df.to_csv(arcs_csv, index=False, encoding="utf-8-sig")



    print(
        f"[OK] Arc table generated: "
        f"arcs={len(arc_df)}, "
        f"congested_arcs={len(congested_arc_ids)}"
    )
    print("[FILE] arcs:", arcs_csv)

    customers_df = build_customers_with_duplicates(
        G=G,
        num_customers=NUM_CUSTOMERS,
        num_unique_customer_nodes=NUM_UNIQUE_CUSTOMER_NODES,
        seed=SEED + 11
    )

    customers_csv = os.path.join(BASE_DIR, "customers.csv")
    if EXPORT_CSV_FILES:
        customers_df.to_csv(customers_csv, index=False, encoding="utf-8-sig")

    

    unique_customer_nodes = customers_df["node_id"].nunique()
    duplicate_customer_count = NUM_CUSTOMERS - unique_customer_nodes

    print(
        f"[OK] Customers generated: "
        f"customers={len(customers_df)}, "
        f"unique_customer_nodes={unique_customer_nodes}, "
        f"duplicated_customers={duplicate_customer_count}"
    )
    print("[FILE] customers:", customers_csv)

    tasks_df = build_tasks_with_time_windows(
        customers_df=customers_df,
        seed=SEED + 21
    )

    tasks_csv = os.path.join(BASE_DIR, "tasks.csv")
    if EXPORT_CSV_FILES:
        tasks_df.to_csv(tasks_csv, index=False, encoding="utf-8-sig")

    print("[OK] Deadline-based tasks generated")
    print("[FILE] tasks:", tasks_csv)

    od_df = build_od_pairs_from_customers(
        customers_df=customers_df,
        tasks_df=tasks_df,
        pos=pos,
        city_scale_km=CITY_SCALE_KM,
        base_demand_veh_h=BASE_OD_DEMAND_VEH_H,
        od_max_distance_km=OD_MAX_DISTANCE_KM,
        od_topk_per_origin=OD_TOPK_PER_ORIGIN
    )

    od_csv = os.path.join(BASE_DIR, "od_pairs.csv")
    if EXPORT_CSV_FILES:
        od_df.to_csv(od_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] OD pairs generated: OD={len(od_df)}")
    print("[FILE] od:", od_csv)

    od_df_full = od_df.copy()
    od_df_filtered = od_df[
        (od_df["same_physical_node"] == False) &
        (od_df["euclid_distance_km"] <= float(OD_MAX_DISTANCE_KM) * 0.75)
    ].copy()
    if len(od_df_filtered) == 0:
        od_df_filtered = od_df_full.copy()

    path_df, path_arc_df, path_arc_matrix, od_path_map = build_path_bundle_for_od(od_df, arc_df)
    path_df_full, path_arc_df_full, path_arc_matrix_full, od_path_map_full = build_path_bundle_for_od(od_df_full, arc_df)
    path_df_filtered, path_arc_df_filtered, path_arc_matrix_filtered, od_path_map_filtered = build_path_bundle_for_od(od_df_filtered, arc_df)

    path_csv = os.path.join(BASE_DIR, "candidate_paths.csv")
    path_arc_csv = os.path.join(BASE_DIR, "path_arc_incidence.csv")
    path_dense_csv = os.path.join(BASE_DIR, "path_arc_incidence_dense.csv")

    path_full_csv = os.path.join(BASE_DIR, "candidate_paths_full.csv")
    path_arc_full_csv = os.path.join(BASE_DIR, "path_arc_incidence_full.csv")
    path_dense_full_csv = os.path.join(BASE_DIR, "path_arc_incidence_dense_full.csv")

    path_filtered_csv = os.path.join(BASE_DIR, "candidate_paths_filtered.csv")
    path_arc_filtered_csv = os.path.join(BASE_DIR, "path_arc_incidence_filtered.csv")
    path_dense_filtered_csv = os.path.join(BASE_DIR, "path_arc_incidence_dense_filtered.csv")

    if EXPORT_CSV_FILES:
        path_df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    if EXPORT_CSV_FILES:
        path_arc_df.to_csv(path_arc_csv, index=False, encoding="utf-8-sig")

    if EXPORT_CSV_FILES:
        path_arc_matrix.to_csv(path_dense_csv, encoding="utf-8-sig")

    if EXPORT_CSV_FILES:
        path_df_full.to_csv(path_full_csv, index=False, encoding="utf-8-sig")
    if EXPORT_CSV_FILES:
        path_arc_df_full.to_csv(path_arc_full_csv, index=False, encoding="utf-8-sig")
    if EXPORT_CSV_FILES:
        path_arc_matrix_full.to_csv(path_dense_full_csv, encoding="utf-8-sig")

    if EXPORT_CSV_FILES:
        path_df_filtered.to_csv(path_filtered_csv, index=False, encoding="utf-8-sig")
    if EXPORT_CSV_FILES:
        path_arc_df_filtered.to_csv(path_arc_filtered_csv, index=False, encoding="utf-8-sig")
    if EXPORT_CSV_FILES:
        path_arc_matrix_filtered.to_csv(path_dense_filtered_csv, encoding="utf-8-sig")

    print(
        f"[OK] Candidate paths generated: "
        f"paths={len(path_df)}, "
        f"path_arc_rows={len(path_arc_df)}"
    )
    print("[FILE] paths:", path_csv)
    print("[FILE] path_arc:", path_arc_csv)

    print("[FILE] path_arc_dense:", path_dense_csv)
    print(f"[FILE] full paths/path_arc/path_dense: {path_full_csv} | {path_arc_full_csv} | {path_dense_full_csv}")
    print(f"[FILE] filtered paths/path_arc/path_dense: {path_filtered_csv} | {path_arc_filtered_csv} | {path_dense_filtered_csv}")
    print(f"[INFO] OD full={len(od_df_full)}, filtered={len(od_df_filtered)}")

    path_arc_dense_csv = os.path.join(BASE_DIR, "path_arc_incidence_dense.csv")
    if EXPORT_CSV_FILES:
        path_arc_matrix.to_csv(path_arc_dense_csv, encoding="utf-8-sig")
    print("[FILE] path_arc_dense:", path_arc_dense_csv)

    od_path_map = build_od_path_map(path_df)

    baseline_path_flow = pd.Series(
        0.0,
        index=path_arc_matrix.index,
        name="path_flow_veh_h"
    )



    arc_bpr_eval_df = evaluate_bpr_on_arcs(
        arc_df=arc_df,
        link_flow_series=compute_link_flow_from_path_flow(
            path_arc_matrix=path_arc_matrix,
            path_flow_series=baseline_path_flow
        )
    )
    baseline_link_flow = arc_bpr_eval_df.set_index("arc_id")["flow_veh_h"].copy()


    arc_bpr_eval_csv = os.path.join(BASE_DIR, "arcs_bpr_with_baseline_flow.csv")
    if EXPORT_CSV_FILES:
        arc_bpr_eval_df.to_csv(arc_bpr_eval_csv, index=False, encoding="utf-8-sig")
    print("[FILE] arcs with baseline flow:", arc_bpr_eval_csv)

    od_demand_matrix = build_od_demand_matrix(
        customers_df=customers_df,
        od_df=od_df
    )

    od_time_matrix = build_od_time_matrix(
        customers_df=customers_df,
        path_df=path_df
    )

    od_demand_matrix_csv = os.path.join(BASE_DIR, "od_demand_matrix.csv")
    od_time_matrix_csv = os.path.join(BASE_DIR, "od_free_flow_time_matrix.csv")

    if EXPORT_CSV_FILES:
        od_demand_matrix.to_csv(od_demand_matrix_csv, encoding="utf-8-sig")
    if EXPORT_CSV_FILES:
        od_time_matrix.to_csv(od_time_matrix_csv, encoding="utf-8-sig")

    print("[FILE] od demand matrix:", od_demand_matrix_csv)
    print("[FILE] od time matrix:", od_time_matrix_csv)

    iess_nodes = []
    evcs_nodes = []

    iess_nodes = select_fixed_facility_nodes(pos, IESS_TARGET_COORDS)
    evcs_nodes = select_fixed_facility_nodes(pos, EVCS_TARGET_COORDS)
    facility_df = pd.DataFrame(
        [{"facility_type": "IESS", "node_id": n} for n in iess_nodes] +
        [{"facility_type": "EVCS", "node_id": n} for n in evcs_nodes]
    )
    print(f"[INFO] Fixed IESS nodes: {list(iess_nodes)}")
    print(f"[INFO] Fixed EVCS nodes: {list(evcs_nodes)}")
    print(f"[INFO] Fleet size EHDT: {FLEET_SIZE_EHDT}")

    if EXPORT_EXCEL_BUNDLE:
        excel_path = export_bundle_to_excel(
            base_dir=BASE_DIR,
            sheets={
                "nodes": nodes_df,
                "arcs_bpr": arc_df,
                "customers": customers_df,
                "tasks": tasks_df,
                "od_pairs": od_df,
                "candidate_paths": path_df,
                "path_arc_incidence": path_arc_df,
                "path_arc_dense": path_arc_matrix.reset_index(),
                "arcs_bpr_with_flow": arc_bpr_eval_df,
                "od_demand_matrix": od_demand_matrix.reset_index().rename(columns={"index": "origin_customer_id"}),
                "od_free_flow_time_matrix": od_time_matrix.reset_index().rename(columns={"index": "origin_customer_id"})
            }
        )
        print("[FILE] excel bundle:", excel_path)



    initial_path_flow = baseline_path_flow
    initial_link_flow = baseline_link_flow

    data = {
        "traffic_graph": G,
        "pos": pos,

        "nodes_df": nodes_df,

        "customers_df": customers_df,
        "tasks_df": tasks_df,

        "arc_df": arc_df,
        "od_df": od_df,
        "path_df_full": path_df_full,
        "path_df_filtered": path_df_filtered,
        "path_df": path_df,
        "path_arc_df_full": path_arc_df_full,
        "path_arc_df_filtered": path_arc_df_filtered,
        "path_arc_df": path_arc_df,
        "path_arc_matrix_full": path_arc_matrix_full,
        "path_arc_matrix_filtered": path_arc_matrix_filtered,
        "path_arc_matrix": path_arc_matrix,
        "od_path_map_full": od_path_map_full,
        "od_path_map_filtered": od_path_map_filtered,
        "od_path_map": od_path_map,

        "od_demand_matrix": od_demand_matrix,
        "od_time_matrix": od_time_matrix,

        "initial_path_flow": initial_path_flow,
        "initial_link_flow": initial_link_flow,
        "arc_bpr_eval_df": arc_bpr_eval_df,

        "congested_arc_ids": congested_arc_ids,

        "params": {
            "CITY_NODE_COUNT": CITY_NODE_COUNT,
            "NUM_CUSTOMERS": NUM_CUSTOMERS,
            "NUM_UNIQUE_CUSTOMER_NODES": NUM_UNIQUE_CUSTOMER_NODES,
            "NUM_DUPLICATED_CUSTOMERS": NUM_DUPLICATED_CUSTOMERS,
            "CITY_SCALE_KM": CITY_SCALE_KM,
            "AVG_SPEED_KMH": AVG_SPEED_KMH,

            "DEMAND_SMALL_RANGE": DEMAND_SMALL_RANGE,
            "DEMAND_MEDIUM_RANGE": DEMAND_MEDIUM_RANGE,
            "DEMAND_LARGE_RANGE": DEMAND_LARGE_RANGE,
            "DEMAND_CLASS_PROBS": DEMAND_CLASS_PROBS,

            "BPR_ALPHA": BPR_ALPHA,
            "BPR_BETA": BPR_BETA,
            "BASE_OD_DEMAND_VEH_H": BASE_OD_DEMAND_VEH_H,
            "MAX_K_SHORTEST_PATHS": MAX_K_SHORTEST_PATHS,
            "USE_DIRECTED_ARCS": USE_DIRECTED_ARCS,

            "ENABLE_CONGESTION_EVENT": ENABLE_CONGESTION_EVENT,
            "CONGESTION_CENTER": CONGESTION_CENTER,
            "CONGESTION_RADIUS_KM": CONGESTION_RADIUS_KM,
            "CONGESTION_CAPACITY_FACTOR": CONGESTION_CAPACITY_FACTOR,
            "CONGESTION_TIME_PENALTY": CONGESTION_TIME_PENALTY,

            "PLANNING_HORIZON_H": PLANNING_HORIZON_H,
            "SEED": SEED
        }
    }

    out_path = os.path.join(BASE_DIR, "data_bpr.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print("[FILE] data:", out_path)

    if PLOT_NETWORK:
        original_network_png = os.path.join(
            BASE_DIR,
            "fig_1_original_road_network.png"
        )

        customer_network_png = os.path.join(
            BASE_DIR,
            "fig_2_customer_distribution_on_network.png"
        )

        od_demand_png = os.path.join(
            BASE_DIR,
            "fig_3_od_demand_matrix.png"
        )

        od_time_png = os.path.join(
            BASE_DIR,
            "fig_4_od_free_flow_time_matrix.png"
        )

        task_time_window_png = os.path.join(
            BASE_DIR,
            "fig_5_task_time_windows.png"
        )

        customer_multiplicity_png = os.path.join(
            BASE_DIR,
            "fig_6_customer_node_multiplicity.png"
        )

        plot_original_network(
            G=G,
            output_path=original_network_png
        )

        plot_customer_network(
            G=G,
            customers_df=customers_df,
            output_path=customer_network_png
        )

        plot_od_matrix(
            od_matrix=od_demand_matrix,
            output_path=od_demand_png,
            title="OD Demand Matrix",
            cbar_label="Demand (veh/h)"
        )

        plot_od_matrix(
            od_matrix=od_time_matrix,
            output_path=od_time_png,
            title="OD Free-flow Travel Time Matrix",
            cbar_label="Travel time (h)"
        )


        plot_task_time_windows(
            tasks_df=tasks_df,
            output_path=task_time_window_png
        )

        plot_customer_node_multiplicity(
            customers_df=customers_df,
            output_path=customer_multiplicity_png
        )

        print("[FIG] saved:", original_network_png)
        print("[FIG] saved:", customer_network_png)
        print("[FIG] saved:", od_demand_png)
        print("[FIG] saved:", od_time_png)
        print("[FIG] saved:", task_time_window_png)
        print("[FIG] saved:", customer_multiplicity_png)

    same_node_od_count = int(od_df["same_physical_node"].sum())

    print("\n================ Data structure check ================")
    print(f"Road nodes: {G.number_of_nodes()}")
    print(f"Road edges: {G.number_of_edges()}")
    print(f"Connected: {nx.is_connected(G)}")
    print(f"Components: {nx.number_connected_components(G)}")
    print(f"Customers: {len(customers_df)}")
    print(f"Unique customer road nodes: {customers_df['node_id'].nunique()}")
    print(f"Duplicated customers: {len(customers_df) - customers_df['node_id'].nunique()}")
    print(f"OD pairs: {len(od_df)}")
    print(f"Same-physical-node OD pairs: {same_node_od_count}")
    print(f"Candidate paths: {len(path_df)}")
    print(f"Arcs: {len(arc_df)}")
    print("======================================================")

    print("\n================ Task structure ================")
    print("Main task columns:")
    print("  tasks_df[['task_id','customer_id','node_id','time_window_type','has_start_constraint','start_time_h','deadline_h','service_time_h','demand_class','demand']]")
    print("\nDemand meaning:")
    print("  demand is treated as unloading amount.")
    print("\nDeadline constraint:")
    print("  arrival_i + service_time_i <= deadline_i")
    print("\nOptional start constraint:")
    print("  if has_start_constraint=True: arrival_i >= start_time_i")
    print("================================================")

    print("\n================ BPR traffic optimization structure ================")
    print("Arc set A:")
    print("  arc_df[['arc_id','from_node','to_node','length_km','t0_h','capacity_veh_h','alpha','beta']]")
    print("\nOD set W:")
    print("  od_df[['od_id','origin_customer_id','destination_customer_id','origin_node','destination_node','demand_veh_h']]")
    print("\nPath set K_w:")
    print("  path_df[['path_id','od_id','path_type','path_length_km','free_flow_time_h']]")
    print("\nPath-arc incidence:")
    print("  path_arc_matrix.loc[path_id, arc_id] = 1 if path uses arc")
    print("\nBPR:")
    print("  t_a(x_a) = t0_a * (1 + alpha_a * (x_a / capacity_a)^beta_a)")
    print("\nFlow relation:")
    print("  x_a = sum_w sum_k delta_{a,k} f_wk")
    print("====================================================================")


if __name__ == "__main__":
    main()