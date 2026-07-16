import ast
import csv
import os
import pickle
import zipfile
import xml.etree.ElementTree as ET


SELECTION_LOGIC_TEXT = """
选址逻辑:
1) 候选节点: 全部路网节点 - depot/显式排除节点。
2) OD几何中心: 读取 OD 对，把每个 OD 的起终点中点作为设施服务中心参考。
3) IESS: 先选，只看 OD 几何中心覆盖效果，并加少量分散项，避免多个 IESS 重合。
4) EVCS: 后选，以已选 EVCS 之间的间距为主，同时稍微向 OD 几何中心收拢。
5) 电网映射: 按 IESS + EVCS 的顺序，将路网设施映射到 35kV bus 列表。
""".strip()


class PreviewGraph:
    def __init__(self):
        self._nodes = {}

    def add_node(self, node_id, **attrs):
        self._nodes[int(node_id)] = dict(attrs)

    def nodes(self, data=False):
        if data:
            return list(self._nodes.items())
        return list(self._nodes.keys())


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _node_coords(traffic_graph):
    return {
        int(n): (float(attrs.get("x", 0.0)), float(attrs.get("y", 0.0)))
        for n, attrs in traffic_graph.nodes(data=True)
    }


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def select_facility_nodes(
    traffic_graph,
    locations=None,
    n_iess=4,
    n_evcs=4,
    exclude_node_ids=None,
    od_pairs=None,
    verbose=False,
):
    """Select deterministic road-network nodes for IESS and EVCS."""
    pos = _node_coords(traffic_graph)
    if not pos:
        return [], []

    excluded = {int(n) for n in (exclude_node_ids or [])}
    if locations:
        excluded.update(
            int(info["node_id"])
            for info in locations.values()
            if info.get("type") == "Depot" and "node_id" in info
        )

    candidates = sorted(n for n in pos if n not in excluded)
    if not candidates:
        candidates = sorted(pos)

    demand_nodes = []
    if locations:
        for info in locations.values():
            if info.get("type") == "Customer" and int(info["node_id"]) in pos:
                demand_nodes.append(int(info["node_id"]))
    if not demand_nodes:
        demand_nodes = candidates

    center = (
        _mean(pos[n][0] for n in demand_nodes),
        _mean(pos[n][1] for n in demand_nodes),
    )
    od_center_points = []
    for origin, destination in od_pairs or []:
        origin = int(origin)
        destination = int(destination)
        if origin in pos and destination in pos:
            od_center_points.append((
                (pos[origin][0] + pos[destination][0]) / 2.0,
                (pos[origin][1] + pos[destination][1]) / 2.0,
            ))
    if od_center_points:
        od_centroid = (
            _mean(p[0] for p in od_center_points),
            _mean(p[1] for p in od_center_points),
        )
    else:
        od_centroid = center

    nearest_demand_score = {}
    service_points = od_center_points or [pos[n] for n in demand_nodes]
    for n in candidates:
        nearest = sorted(_dist2(pos[n], point) for point in service_points)[:10]
        nearest_demand_score[n] = -_mean(nearest) if nearest else 0.0

    def pick_iess(n_select, used):
        selected = []
        for _ in range(min(int(n_select), len(candidates) - len(used))):
            best_node = None
            best_score = float("inf")
            for n in candidates:
                if n in used:
                    continue
                trial = selected + [n]
                nearest_od_to_iess = [
                    min(_dist2(point, pos[s]) for s in trial)
                    for point in service_points
                ]
                avg_od_cover = _mean(nearest_od_to_iess)
                od_centroid_dist = _dist2(pos[n], od_centroid)
                spread = min((_dist2(pos[n], pos[u]) for u in used), default=1.0)
                score = (
                    0.75 * avg_od_cover
                    + 0.15 * od_centroid_dist
                    - 0.10 * spread
                )
                if score < best_score:
                    best_node = n
                    best_score = score
            if best_node is None:
                break
            selected.append(int(best_node))
            used.add(int(best_node))
        return selected

    def pick_evcs(n_select, used, weight_center, weight_demand, weight_spread):
        selected = []
        for _ in range(min(int(n_select), len(candidates) - len(used))):
            best_node = None
            best_score = float("-inf")
            for n in candidates:
                if n in used:
                    continue
                spread = min((_dist2(pos[n], pos[u]) for u in selected), default=0.0)
                score = (
                    weight_center * (-_dist2(pos[n], center))
                    + weight_demand * nearest_demand_score[n]
                    + weight_spread * spread
                )
                if score > best_score:
                    best_node = n
                    best_score = score
            if best_node is None:
                break
            selected.append(int(best_node))
            used.add(int(best_node))
        return selected

    used = set()
    iess_nodes = pick_iess(n_iess, used)
    evcs_nodes = pick_evcs(n_evcs, used, weight_center=0.4, weight_demand=0.3, weight_spread=0.55)
    if verbose:
        print_selection_logic()
        print(f"[facility_siting] candidates={len(candidates)}, demand_nodes={len(demand_nodes)}")
        print(f"[facility_siting] od_center_points={len(od_center_points)}")
        print(f"[facility_siting] demand_center=({center[0]:.4f}, {center[1]:.4f})")
        print(f"[facility_siting] od_centroid=({od_centroid[0]:.4f}, {od_centroid[1]:.4f})")
        print(f"[facility_siting] selected IESS road nodes: {iess_nodes}")
        print(f"[facility_siting] selected EVCS road nodes: {evcs_nodes}")
    return iess_nodes, evcs_nodes


def print_selection_logic():
    print("\n" + "=" * 80)
    print("【IESS / EVCS 路网选址逻辑】")
    print("=" * 80)
    print(SELECTION_LOGIC_TEXT)
    print("=" * 80 + "\n")


def _read_xlsx_rows(xlsx_path, sheet_name=None):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as zf:
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_name is not None:
            rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
            book_ns = {
                "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
                "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            }
            workbook = ET.fromstring(zf.read("xl/workbook.xml"))
            rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rel_targets = {
                rel.get("Id"): rel.get("Target")
                for rel in rels.findall("r:Relationship", rel_ns)
            }
            for sheet in workbook.findall(".//a:sheet", book_ns):
                if sheet.get("name") == sheet_name:
                    rel_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                    target = rel_targets[rel_id]
                    sheet_path = "xl/" + target.lstrip("/")
                    break
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        sheet = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in sheet.findall(".//a:row", ns):
            values = []
            for c in row.findall("a:c", ns):
                value = c.find("a:v", ns)
                raw = "" if value is None else value.text or ""
                if c.get("t") == "s" and raw:
                    raw = shared[int(raw)]
                values.append(raw)
            rows.append(values)
        return rows


def _read_xlsx_sheet1_rows(xlsx_path):
    return _read_xlsx_rows(xlsx_path)


def get_35kv_buses(grid_input_data=None, voltage_xlsx="grid_bus_voltage.xlsx"):
    """Return available 35 kV bus ids, preferring the explicit voltage file."""
    buses = []
    if os.path.exists(voltage_xlsx):
        rows = _read_xlsx_sheet1_rows(voltage_xlsx)
        if rows:
            header = [str(v).strip() for v in rows[0]]
            if "bus" in header or "nidou" in header:
                bus_idx = header.index("bus") if "bus" in header else header.index("nidou")
                kv_idx = header.index("Vbase_kV") if "Vbase_kV" in header else header.index("vbase_kv")
                for row in rows[1:]:
                    if len(row) <= max(bus_idx, kv_idx):
                        continue
                    if round(float(row[kv_idx]), 6) == 35.0:
                        buses.append(int(float(row[bus_idx])))

    if not buses and grid_input_data is None:
        buses = list(range(40))

    if not buses and grid_input_data is not None and "gridcat_buses" in grid_input_data:
        bus_df = grid_input_data["gridcat_buses"]
        if hasattr(bus_df, "columns") and "Vbase_kV" in bus_df.columns:
            buses = bus_df.loc[
                bus_df["Vbase_kV"].astype(float).round(6) == 35.0, "nidou"
            ].astype(int).tolist()
        elif hasattr(bus_df, "__getitem__"):
            buses = bus_df["nidou"].astype(int).tolist()

    return sorted(dict.fromkeys(buses))


def map_facilities_to_buses(iess_nodes, evcs_nodes, grid_input_data=None, voltage_xlsx="grid_bus_voltage.xlsx"):
    buses = get_35kv_buses(grid_input_data=grid_input_data, voltage_xlsx=voltage_xlsx)
    if not buses:
        raise ValueError("No grid buses available for IESS/EVCS mapping.")

    facility_nodes = list(iess_nodes) + list(evcs_nodes)
    if len(facility_nodes) > len(buses):
        raise ValueError(f"Need {len(facility_nodes)} grid buses but only {len(buses)} available.")
    return {int(node): int(bus) for node, bus in zip(facility_nodes, buses)}


def print_facility_siting_summary(traffic_graph, iess_nodes, evcs_nodes, bus_map=None):
    pos = _node_coords(traffic_graph)
    rows = []
    for facility_type, nodes in (("IESS", iess_nodes), ("EVCS", evcs_nodes)):
        for node in nodes:
            x, y = pos.get(int(node), (None, None))
            rows.append({
                "type": facility_type,
                "road_node": int(node),
                "x": None if x is None else round(float(x), 6),
                "y": None if y is None else round(float(y), 6),
                "grid_bus_35kv": None if bus_map is None else bus_map.get(int(node)),
            })

    print("\n" + "=" * 80)
    print("【IESS / EVCS 选址结果】")
    print("=" * 80)
    if not rows:
        print("[facility_siting] No facilities selected.")
    else:
        headers = ["type", "road_node", "x", "y", "grid_bus_35kv"]
        print(" ".join(f"{h:>14}" for h in headers))
        for row in rows:
            print(" ".join(f"{str(row[h]):>14}" for h in headers))
    print("=" * 80 + "\n")
    return rows


def visualize_facility_siting(
    traffic_graph,
    iess_nodes,
    evcs_nodes,
    bus_map=None,
    output_path="facility_siting.svg",
    locations=None,
    label_all_nodes=True,
):
    pos = _node_coords(traffic_graph)
    if not pos:
        return None
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height, pad = 1000, 760, 50

    def sx(x):
        return pad + (x - min_x) / (max_x - min_x or 1.0) * (width - 2 * pad)

    def sy(y):
        return height - pad - (y - min_y) / (max_y - min_y or 1.0) * (height - 2 * pad)

    iess_set = {int(n) for n in iess_nodes}
    evcs_set = {int(n) for n in evcs_nodes}
    depot_set = set()
    if locations:
        depot_set = {
            int(info["node_id"])
            for info in locations.values()
            if info.get("type") == "Depot" and int(info["node_id"]) in pos
        }

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="50" y="30" font-size="22" font-family="Arial">IESS / EVCS facility siting</text>')
    for node, (x, y) in pos.items():
        color, r = "#c7c7c7", 3
        if node in depot_set:
            color, r = "#444444", 7
        if node in iess_set:
            color, r = "#d62728", 10
        if node in evcs_set:
            color, r = "#1f77b4", 8
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{r}" fill="{color}" stroke="black" stroke-width="0.5"/>')
        if label_all_nodes:
            parts.append(
                f'<text x="{sx(x)+4:.2f}" y="{sy(y)+4:.2f}" '
                f'font-size="9" font-family="Arial" fill="#555555">{node}</text>'
            )
        if node in iess_set or node in evcs_set:
            label = "IESS" if node in iess_set else "EVCS"
            bus = "" if bus_map is None else f" / bus {bus_map.get(node)}"
            parts.append(f'<text x="{sx(x)+8:.2f}" y="{sy(y)-8:.2f}" font-size="12" font-family="Arial">{label} {node}{bus}</text>')
    parts.append('<text x="50" y="720" font-size="14" font-family="Arial" fill="#d62728">● IESS</text>')
    parts.append('<text x="150" y="720" font-size="14" font-family="Arial" fill="#1f77b4">● EVCS</text>')
    parts.append('<text x="250" y="720" font-size="14" font-family="Arial" fill="#444444">● Depot</text>')
    parts.append('</svg>')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[facility_siting] visualization saved to: {output_path}")
    return output_path


def get_existing_evcs_count(default=10):
    """Read the existing EVCS count without adding a new config knob."""
    try:
        with open("NetworkGen.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "EVCS_TARGET_COORDS" for t in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                return len(node.value.elts)
    except Exception:
        pass
    return int(default)


def relocate_stations_to_35kv_buses(data, grid_input_data, n_iess=None, n_evcs=None):
    """Relocate IESS stations and map both IESS/EVCS to 35 kV grid buses."""
    station_names = sorted(data["stations"].keys())
    n_iess = len(station_names) if n_iess is None else min(int(n_iess), len(station_names))
    n_evcs = get_existing_evcs_count() if n_evcs is None else int(n_evcs)
    iess_nodes, evcs_nodes = select_facility_nodes(data["traffic_graph"], locations=data.get("locations", {}), n_iess=n_iess, n_evcs=n_evcs, verbose=True)
    bus_map = map_facilities_to_buses(iess_nodes, evcs_nodes, grid_input_data)
    print_facility_siting_summary(data["traffic_graph"], iess_nodes, evcs_nodes, bus_map)
    visualize_facility_siting(data["traffic_graph"], iess_nodes, evcs_nodes, bus_map=bus_map, locations=data.get("locations", {}))
    for station_name, road_node in zip(station_names[:n_iess], iess_nodes):
        data["locations"][station_name]["type"] = "SwapStation"
        data["locations"][station_name]["node_id"] = int(road_node)
        data["stations"][station_name]["facility_type"] = "IESS"
        data["stations"][station_name]["bus_id"] = int(bus_map[int(road_node)])
    data["evcs"] = {}
    for idx, road_node in enumerate(evcs_nodes, start=1):
        evcs_name = f"EVCS_{idx}"
        data["locations"][evcs_name] = {"type": "EVCS", "node_id": int(road_node)}
        data["evcs"][evcs_name] = {"facility_type": "EVCS", "node_id": int(road_node), "bus_id": int(bus_map[int(road_node)])}
    data["facility_siting"] = {"IESS_nodes": [int(n) for n in iess_nodes], "EVCS_nodes": [int(n) for n in evcs_nodes], "road_to_35kv_bus": {int(k): int(v) for k, v in bus_map.items()}}
    return data


def _load_preview_road_graph(node_csv="node_df.csv", vrp_xlsx="vrp_outputs.xlsx"):
    graph = PreviewGraph()
    if os.path.exists(node_csv):
        with open(node_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    node_id = int(row["node_id"])
                except ValueError:
                    continue
                graph.add_node(node_id, x=float(row["x"]), y=float(row["y"]))
        print(f"[facility_siting] loaded {len(graph.nodes())} road nodes from {node_csv}")
        return graph, None

    if os.path.exists(vrp_xlsx):
        rows = _read_xlsx_rows(vrp_xlsx, sheet_name="nodes")
        header = rows[0]
        idx = {name: header.index(name) for name in ("node_id", "x", "y")}
        for row in rows[1:]:
            try:
                node_id = int(float(row[idx["node_id"]]))
            except (ValueError, IndexError):
                continue
            graph.add_node(node_id, x=float(row[idx["x"]]), y=float(row[idx["y"]]))
        print(f"[facility_siting] loaded {len(graph.nodes())} road nodes from {vrp_xlsx}:nodes")
        return graph, None

    raise FileNotFoundError(f"Cannot find {node_csv} or {vrp_xlsx}")


def _load_preview_od_pairs(vrp_xlsx="vrp_outputs.xlsx", candidate_paths_csv="candidate_paths.csv"):
    if os.path.exists(vrp_xlsx):
        rows = _read_xlsx_rows(vrp_xlsx, sheet_name="od_pairs")
        header = rows[0]
        origin_col = "origin_node" if "origin_node" in header else "origin_customer_index"
        destination_col = "destination_node" if "destination_node" in header else "destination_customer_index"
        origin_idx = header.index(origin_col)
        destination_idx = header.index(destination_col)
        pairs = []
        for row in rows[1:]:
            try:
                pairs.append((int(float(row[origin_idx])), int(float(row[destination_idx]))))
            except (ValueError, IndexError):
                continue
        print(f"[facility_siting] loaded {len(pairs)} OD pairs from {vrp_xlsx}:od_pairs")
        return pairs

    if not os.path.exists(candidate_paths_csv):
        return []
    pairs = {}
    with open(candidate_paths_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            od_id = row.get("od_id")
            if od_id in pairs:
                continue
            try:
                pairs[od_id] = (int(row["origin_node"]), int(row["destination_node"]))
            except (KeyError, ValueError):
                continue
    pairs = list(pairs.values())
    print(f"[facility_siting] loaded {len(pairs)} OD pairs from {candidate_paths_csv}")
    return pairs


def preview_facility_siting(n_iess=5, n_evcs=10, output_path="facility_siting.svg"):
    """Run only the siting preview: print logic/results and save a figure."""
    road_graph, locations = _load_preview_road_graph()
    od_pairs = _load_preview_od_pairs()
    iess_nodes, evcs_nodes = select_facility_nodes(
        road_graph,
        locations=locations,
        n_iess=n_iess,
        n_evcs=n_evcs,
        od_pairs=od_pairs,
        verbose=True,
    )
    bus_map = map_facilities_to_buses(iess_nodes, evcs_nodes)
    summary_rows = print_facility_siting_summary(road_graph, iess_nodes, evcs_nodes, bus_map)
    visualize_facility_siting(road_graph, iess_nodes, evcs_nodes, bus_map=bus_map, output_path=output_path, locations=locations)
    return {"iess_nodes": iess_nodes, "evcs_nodes": evcs_nodes, "road_to_35kv_bus": bus_map, "summary_rows": summary_rows, "figure_path": output_path}


if __name__ == "__main__":
    preview_facility_siting()
