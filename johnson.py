"""Johnson shortest paths for the project's sectioned CSV files (Python 3.9+).

Run interactively: python johnson.py
Run directly: python johnson.py Datasets/Dataset1.csv --source 0 --goal 3
From a notebook: distances, parents = johnson(nodes, edges)
"""
import argparse
import csv
import heapq
import math
from pathlib import Path


class NegativeCycleError(ValueError):
    pass


def read_dataset(path):
    """Read nonnegative node IDs, directed integer-weight edges and reference costs."""
    node_lines, edges, expected = {}, [], {}
    section = header = None
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            row = [cell.strip() for cell in row]
            if not any(row):
                continue
            if row[0].startswith("#"):
                section, header = row[0], None
                continue
            if header is None:
                header = row
                continue
            fields = dict(zip(header, row))
            if section == "# Nodes":
                location = f"{Path(path).name}, line {reader.line_num}"
                raw = fields.get("node_id", "")
                try:
                    node = int(raw)
                except ValueError:
                    raise ValueError(f"{location}: expected an integer node_id, found {raw!r}.") from None
                if node < 0:
                    raise ValueError(f"{location}: node ID {node} must be 0 or greater.")
                if node in node_lines:
                    raise ValueError(f"{location}: duplicate node ID {node} in # Nodes "
                                     f"(first listed on line {node_lines[node]}).")
                node_lines[node] = reader.line_num
            elif section == "# Edges":
                edges.append((int(fields["source_node"]), int(fields["target_node"]),
                              int(fields["weight"])))
            elif section == "# Expected Distances" and fields["expected_distance"]:
                pair = (int(fields["source_node"]), int(fields["target_node"]))
                expected[pair] = float(fields["expected_distance"])
    nodes = sorted(node_lines)
    if not nodes:
        raise ValueError("No nodes found in the CSV.")
    if any(u not in node_lines or v not in node_lines for u, v, _ in edges):
        raise ValueError("An edge references an unknown node.")
    return nodes, edges, expected


def bellman_ford(nodes, edges):
    # Starting every potential at 0 has the effect of relaxing a zero-weight
    # edge from a temporary super-source to each vertex.
    h = dict.fromkeys(nodes, 0)
    for _ in range(len(nodes) - 1):
        changed = False
        for u, v, weight in edges:
            candidate = h[u] + weight
            if candidate < h[v]:
                h[v] = candidate
                changed = True
        if not changed:
            break
    if any(h[u] + weight < h[v] for u, v, weight in edges):
        raise NegativeCycleError("Negative Cycle Detected")
    return h


def dijkstra(adjacency, source):
    distance = dict.fromkeys(adjacency, math.inf)
    distance[source] = 0
    parent = {}
    queue = [(0, source)]
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != distance[u]:
            continue
        for v, weight in adjacency[u]:
            candidate = cost + weight
            if candidate < distance[v]:
                distance[v] = candidate
                parent[v] = u
                heapq.heappush(queue, (candidate, v))
    return distance, parent


def johnson(nodes, edges):
    """Return nested dictionaries: distances[source][goal] and parents[source]."""
    nodes, edges = tuple(nodes), list(edges)
    h = bellman_ford(nodes, edges)
    adjacency = {u: [] for u in nodes}
    for u, v, weight in edges:
        adjacency[u].append((v, weight + h[u] - h[v]))

    distances, parents = {}, {}
    for source in nodes:
        reweighted, parents[source] = dijkstra(adjacency, source)
        distances[source] = {v: cost - h[source] + h[v]
                             for v, cost in reweighted.items()}
    return distances, parents


def reconstruct_path(parents, source, goal):
    if source != goal and goal not in parents[source]:
        return []
    route = [goal]
    while route[-1] != source:
        route.append(parents[source][route[-1]])
    return route[::-1]


def choose_csv():
    folder = Path(__file__).resolve().parent / "Datasets"
    files = sorted(folder.glob("*.csv"))
    for index, path in enumerate(files, 1):
        print(f"{index}. {path.name}")
    choice = input("Select a file number or enter a CSV path: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return files[int(choice) - 1]
    return Path(choice).expanduser()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path)
    parser.add_argument("--source", type=int)
    parser.add_argument("--goal", type=int)
    args = parser.parse_args()
    nodes, edges, expected = read_dataset(args.csv or choose_csv())
    try:
        distances, parents = johnson(nodes, edges)
    except NegativeCycleError as error:
        print(error)
        if expected:
            matched = -math.inf in expected.values()
            print("CSV cycle check:", "PASS" if matched else "FAIL")
            return 0 if matched else 1
        return 0

    if len(nodes) <= 20:
        print("Distance matrix (row and column order):", nodes)
        for source in nodes:
            print([distances[source][goal] for goal in nodes])
    source = args.source if args.source is not None else int(input("Source node: "))
    goal = args.goal if args.goal is not None else int(input("Goal node: "))
    if source not in distances or goal not in distances:
        raise ValueError("Choose source and goal IDs present in the CSV.")
    route = reconstruct_path(parents, source, goal)
    print("Path:", " -> ".join(map(str, route)) if route else "unreachable")
    print("Cost:", distances[source][goal])
    mismatches = 0
    for (u, v), reference in expected.items():
        if u not in distances or v not in distances[u] or distances[u][v] != reference:
            mismatches += 1
            if mismatches <= 5:
                print(f"Mismatch {u} -> {v}: expected {reference}, got {distances.get(u, {}).get(v)}")
    if expected:
        print(f"CSV verification: {len(expected) - mismatches}/{len(expected)} distances matched")
    else:
        print("No expected distances supplied.")
    return int(bool(mismatches))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))



