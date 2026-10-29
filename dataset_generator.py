"""
Dataset generator for Johnson's algorithm and A* search.

Use it three ways
-----------------
1. Interactive prompts:      python dataset_generator.py
                             Select Random Datasets, Test Cases, or View Saved Datasets.
                             Interactive datasets are saved by algorithm under
                             ./simulation/johnson/ and ./simulation/astar/.
2. Command line (scripted):  python dataset_generator.py johnson --nodes 50 --edges 200 --seed 7
                             python dataset_generator.py grid --rows 100 --cols 100 --obstacles 0.2
                             python dataset_generator.py geometric --nodes 300 --extra-edges 600
                             python dataset_generator.py tests
                             python dataset_generator.py suite --out datasets
3. As a module (Jupyter):    import dataset_generator as gen
                             ds = gen.johnson_dataset(n=12, edge_count=30, seed=42)
                             gen.save_dataset(ds, "datasets")
                             ds = gen.load_dataset("datasets/<file>.txt")

A dataset ("ds") is a plain dict:
    name, mode ("johnson"|"astar"), kind, seed, coords {id:(x,y)},
    edges [(u,v,w)], queries [(s,t,expected_dist)], expected
    (Johnson: distance matrix, "NEGATIVE_CYCLE", or None if skipped)

File format (identical for both algorithms):
    # mode=johnson kind=random seed=42 nodes=4 edges=5 directed=true
    NODES
    node_id,x,y
    EDGES
    source,target,weight
    QUERIES                      (A* files only)
    source,target,expected_distance        ("inf" = unreachable)
Johnson files ignore x,y (dummy coordinates are written).
Johnson expected output is saved next to the file as <name>.expected.
"""
import argparse
import heapq
import math
import os
import random
import sys

INF = float("inf")
FW_LIMIT = 400  # Floyd-Warshall reference is O(n^3); skipped above this size
SIM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Datasets"
)
import os


# --------------------------------------------------------------------------
# Low-level graph generators
# --------------------------------------------------------------------------
def generate_johnson_graph(n, edge_count, seed, connected=True):
    """Edges as (u, v, w). Weight = base(>=0) + p[v] - p[u], so p is a valid
    Johnson potential and no negative cycle can exist. Edge 0->1 is always
    present with weight -20 (guaranteed negative edge).
    connected=False skips the Hamiltonian cycle, which allows unreachable
    pairs (guaranteed whenever edge_count < n)."""
    rng = random.Random(seed)
    min_edges = n if connected else 1
    max_edges = n * (n - 1)
    if n < 2:
        raise ValueError("n must be at least 2")
    if not min_edges <= edge_count <= max_edges:
        raise ValueError(f"edge_count must be in [{min_edges}, {max_edges}]")

    p = [rng.randint(-10, 10) for _ in range(n)]
    p[0], p[1] = 10, -10

    edges = {(0, 1)}
    if connected:
        edges |= {(u, (u + 1) % n) for u in range(n)}
    if edge_count > max_edges // 2:  # dense: sample from all remaining pairs
        pool = [(u, v) for u in range(n) for v in range(n)
                if u != v and (u, v) not in edges]
        edges |= set(rng.sample(pool, edge_count - len(edges)))
    else:
        while len(edges) < edge_count:
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v:
                edges.add((u, v))

    out = []
    for u, v in sorted(edges):
        base = 0 if (u, v) == (0, 1) else rng.randint(0, 10)
        out.append((u, v, base + p[v] - p[u]))
    return out


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_grid(rows, cols, obstacle_prob=0.0, seed=0, keep_open=None):
    """4-neighbour grid, edges in both directions, weight 1 (>= Euclid 1).
    Blocked cells are removed (node ids stay r*cols+c). Cells in keep_open
    (default: the two corners) are never blocked."""
    rng = random.Random(seed)
    blocked = {(r, c) for r in range(rows) for c in range(cols)
               if rng.random() < obstacle_prob}
    blocked -= set(keep_open or [(0, 0), (rows - 1, cols - 1)])
    coords, edges = {}, []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            coords[r * cols + c] = (float(c), float(r))
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in blocked:
                    edges.append((r * cols + c, rr * cols + cc, 1.0))
    return coords, edges


def generate_geometric_graph(n, extra_edges, seed, isolate_last=False):
    """Random points in 100x100. Weight = ceil4(euclid * U(1, 1.3)).
    isolate_last=True leaves the last node unreachable (for 'inf' queries)."""
    m = n - 1 if isolate_last else n
    max_extra = m * (m - 1) // 2 - (m - 1)
    if m < 2 or not 0 <= extra_edges <= max_extra:
        raise ValueError(f"extra_edges must be in [0, {max(max_extra, 0)}]")
    rng = random.Random(seed)
    coords = {i: (round(rng.uniform(0, 100), 3), round(rng.uniform(0, 100), 3))
              for i in range(n)}
    pairs = set()
    for i in range(1, m):  # random spanning tree -> connected
        pairs.add((i, rng.randrange(i)))
    while len(pairs) < (m - 1) + extra_edges:
        a, b = rng.randrange(m), rng.randrange(m)
        if a != b:
            pairs.add((a, b) if a > b else (b, a))
    edges = []
    for a, b in sorted(pairs):
        w = math.ceil(euclid(coords[a], coords[b]) * rng.uniform(1.0, 1.3) * 1e4) / 1e4
        edges += [(a, b, w), (b, a, w)]
    return coords, edges


# --------------------------------------------------------------------------
# Reference solvers (used only to produce expected outputs)
# --------------------------------------------------------------------------
def floyd_warshall(n, edges):
    """Distance matrix, or None if a negative cycle exists."""
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[u][v] = min(d[u][v], w)
    for k in range(n):
        for i in range(n):
            if d[i][k] == INF:
                continue
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    if any(d[i][i] < 0 for i in range(n)):
        return None
    return d


def dijkstra(adj, s):
    dist = {s: 0.0}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_astar(coords, edges):
    for u, v, w in edges:
        assert w >= 0, f"negative weight {u}->{v}"
        assert w + 1e-9 >= euclid(coords[u], coords[v]), f"inadmissible {u}->{v}"


# --------------------------------------------------------------------------
# Dataset builders (return dicts; nothing is written to disk)
# --------------------------------------------------------------------------
def _seed(seed):
    return random.randrange(1_000_000) if seed is None else seed


def _dataset(name, mode, kind, seed, coords, edges, queries=None, expected=None):
    return dict(name=name, mode=mode, kind=kind, seed=seed, coords=coords,
                edges=edges, queries=queries or [], expected=expected)


def johnson_from_edges(name, n, edges, seed="fixed", kind="fixed"):
    """Wrap an explicit edge list as a Johnson dataset with its expected output."""
    expected = None
    if n <= FW_LIMIT:
        matrix = floyd_warshall(n, edges)
        expected = "NEGATIVE_CYCLE" if matrix is None else matrix
    coords = {i: (float(i), 0.0) for i in range(n)}  # dummy, ignored by Johnson
    return _dataset(name, "johnson", kind, seed, coords, edges, expected=expected)


def johnson_dataset(n, edge_count, seed=None, connected=True,
                    negative_cycle=False, name=None):
    """Random Johnson graph. negative_cycle=True adds edge 1->0 (w=19) which
    closes the cycle 0->1->0 of total weight -1 (algorithm must reject it);
    that extra edge means the result can have one more edge than requested."""
    seed = _seed(seed)
    edges = generate_johnson_graph(n, edge_count, seed, connected)
    if negative_cycle:
        edges = [e for e in edges if e[:2] != (1, 0)] + [(1, 0, 19)]
    tag = "negcycle" if negative_cycle else "johnson"
    name = name or f"{tag}_n{n}_e{len(edges)}_s{seed}"
    ds = johnson_from_edges(name, n, edges, seed,
                            kind="negcycle" if negative_cycle else "random")
    if not negative_cycle:
        assert ds["expected"] != "NEGATIVE_CYCLE", "generator produced a negative cycle"
        assert any(w < 0 for _, _, w in edges), "no negative edge present"
    return ds


def _astar_dataset(name, kind, seed, coords, edges, pairs):
    validate_astar(coords, edges)
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
    queries = [(s, t, dijkstra(adj, s).get(t, INF)) for s, t in pairs]
    return _dataset(name, "astar", kind, seed, coords, edges, queries)


def grid_dataset(rows, cols, obstacle_prob=0.0, seed=None, start=None,
                 goal=None, name=None):
    """A* grid map. start/goal are (row, col); default = opposite corners."""
    seed = _seed(seed)
    start = start or (0, 0)
    goal = goal or (rows - 1, cols - 1)
    coords, edges = generate_grid(rows, cols, obstacle_prob, seed, [start, goal])
    pair = (start[0] * cols + start[1], goal[0] * cols + goal[1])
    name = name or f"astar_grid_{rows}x{cols}_p{obstacle_prob}_s{seed}"
    return _astar_dataset(name, "grid", seed, coords, edges, [pair])


def random_pairs(nodes, k, seed):
    rng = random.Random(f"queries-{seed}")
    nodes = list(nodes)
    return [tuple(rng.sample(nodes, 2)) for _ in range(k)]


def geometric_dataset(n, extra_edges, seed=None, isolate_last=False,
                      num_queries=3, name=None):
    """A* graph on random 2-D points. isolate_last adds an unreachable query."""
    seed = _seed(seed)
    coords, edges = generate_geometric_graph(n, extra_edges, seed, isolate_last)
    m = n - 1 if isolate_last else n
    pairs = random_pairs(range(m), num_queries, seed)
    if isolate_last:
        pairs.append((0, n - 1))
    name = name or f"astar_geo_n{n}_x{extra_edges}_s{seed}"
    return _astar_dataset(name, "geometric", seed, coords, edges, pairs)


def fixed_cases():
    """Two fixed Johnson test cases (normal and negative-cycle graphs)."""
    tc1 = [(0, 1, -2), (1, 2, -1), (2, 0, 4), (0, 3, 3), (2, 3, 1)]
    return [johnson_from_edges("johnson_tc1_negative_weights", 4, tc1),
            johnson_from_edges("johnson_tc2_negative_cycle", 4, tc1 + [(2, 1, -2)])]


def build_suite():
    """The default fixed-seed suite (10 datasets), returned in memory."""
    suite = fixed_cases()
    suite.append(johnson_dataset(20, 40, 1, name="johnson_sparse_20"))
    suite.append(johnson_dataset(100, 300, 2, name="johnson_sparse_100"))
    suite.append(johnson_dataset(50, 1500, 3, name="johnson_dense_50"))
    suite.append(johnson_dataset(30, 25, 4, connected=False,
                                 name="johnson_disconnected_30"))
    suite.append(grid_dataset(100, 100, 0.0, seed=0, name="astar_grid100_open"))
    suite.append(grid_dataset(100, 100, 0.2, seed=5, name="astar_grid100_obstacles20"))
    suite.append(geometric_dataset(500, 1500, 6, name="astar_geometric_500"))
    suite.append(geometric_dataset(200, 400, 7, isolate_last=True, num_queries=1,
                                   name="astar_unreachable_200"))
    return suite


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------
def fmt(x):
    return "inf" if x == INF else (str(int(x)) if float(x).is_integer() else repr(x))


def save_dataset(ds, out_dir=SIM_DIR):
    """Write <name>.txt (and <name>.expected for Johnson). Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, ds["name"] + ".txt")
    with open(path, "w") as f:
        f.write(f"# mode={ds['mode']} kind={ds['kind']} seed={ds['seed']} "
                f"nodes={len(ds['coords'])} edges={len(ds['edges'])} directed=true\n")
        f.write("NODES\n")
        for i in sorted(ds["coords"]):
            f.write(f"{i},{ds['coords'][i][0]},{ds['coords'][i][1]}\n")
        f.write("EDGES\n")
        for u, v, w in ds["edges"]:
            f.write(f"{u},{v},{fmt(w)}\n")
        if ds["queries"]:
            f.write("QUERIES\n")
            for s, t, d in ds["queries"]:
                f.write(f"{s},{t},{fmt(d)}\n")
    exp = ds["expected"]
    if exp is not None:
        with open(os.path.join(out_dir, ds["name"] + ".expected"), "w") as f:
            if exp == "NEGATIVE_CYCLE":
                f.write("NEGATIVE_CYCLE\n")
            else:
                for row in exp:
                    f.write(",".join(fmt(x) for x in row) + "\n")
    return path


def _num(text):
    t = text.strip()
    return int(t) if t.lstrip("-").isdigit() else float(t)


def load_dataset(path):
    """Read a dataset file (and its .expected file if present) into a dict."""
    meta, coords, edges, queries, section = {}, {}, [], [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                meta = dict(tok.split("=", 1) for tok in line.lstrip("# ").split()
                            if "=" in tok)
            elif line in ("NODES", "EDGES", "QUERIES"):
                section = line
            elif line:
                a, b, c = line.split(",")
                if section == "NODES":
                    coords[int(a)] = (float(b), float(c))
                elif section == "EDGES":
                    edges.append((int(a), int(b), _num(c)))
                else:
                    queries.append((int(a), int(b), float(c)))
    expected, exp_path = None, os.path.splitext(path)[0] + ".expected"
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            rows = [ln.strip() for ln in f if ln.strip()]
        expected = "NEGATIVE_CYCLE" if rows[0] == "NEGATIVE_CYCLE" else \
            [[_num(x) if x != "inf" else INF for x in r.split(",")] for r in rows]
    return _dataset(os.path.splitext(os.path.basename(path))[0],
                    meta.get("mode", "johnson"), meta.get("kind", "unknown"),
                    meta.get("seed"), coords, edges, queries, expected)


def describe(ds):
    neg = sum(1 for e in ds["edges"] if e[2] < 0)
    txt = (f"{ds['name']}: {len(ds['coords'])} nodes, {len(ds['edges'])} edges, "
           f"{neg} negative, seed={ds['seed']}")
    if ds["expected"] == "NEGATIVE_CYCLE":
        txt += ", contains a NEGATIVE CYCLE"
    if ds["queries"]:
        txt += f", {len(ds['queries'])} query(ies)"
    return txt


def list_datasets(folder=SIM_DIR):
    """List saved datasets in this folder and its subfolders, newest first."""
    if not os.path.isdir(folder):
        return []
    rows = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(root, fn)
            with open(path) as f:
                head = f.readline()
            if not head.startswith("#"):
                continue
            meta = dict(t.split("=", 1) for t in head.lstrip("# ").split()
                        if "=" in t)
            rows.append(dict(
                name=fn[:-4], path=path, mtime=os.path.getmtime(path),
                **{k: meta.get(k, "?") for k in
                   ("mode", "kind", "nodes", "edges", "seed")}
            ))
    return sorted(rows, key=lambda r: -r["mtime"])


def choose_dataset(folder=SIM_DIR):
    """Prompt for a saved dataset (including nested folders)."""
    rows = list_datasets(folder)
    if not rows:
        print(f"No datasets in '{folder}/' yet. Generate some first.")
        return None
    print(f"\nSaved datasets in '{folder}/' (newest first):")
    for i, r in enumerate(rows, 1):
        location = os.path.relpath(r["path"], folder)
        print(f"  {i:>2}) {location:<50} {r['mode']:<7} "
              f"nodes={r['nodes']:<6} edges={r['edges']}")
    while True:
        raw = input("Pick a number (blank = back): ").strip()
        if raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            return load_dataset(rows[int(raw) - 1]["path"])
        print(f"  Enter a number between 1 and {len(rows)}.")


def full_suite(out=SIM_DIR):
    for ds in build_suite():
        save_dataset(ds, out)
    print(f"Wrote the full suite to {out}/")


# --------------------------------------------------------------------------
# Interactive prompts
# --------------------------------------------------------------------------
def ask(prompt, cast=int, lo=None, hi=None, default=None):
    """Prompt until a valid value in [lo, hi] is entered (blank = default)."""
    rng_txt = f" [{lo}..{hi}]" if lo is not None and hi is not None else ""
    dflt_txt = f" (default {default})" if default is not None else ""
    while True:
        raw = input(f"{prompt}{rng_txt}{dflt_txt}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast(raw)
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return val


def ask_yes_no(prompt, default=False):
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} ({d}): ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def ask_seed():
    raw = input("Seed (blank = random, it is saved in the file header): ").strip()
    if raw == "":
        seed = random.randrange(1_000_000)
        print(f"  Using random seed {seed}")
        return seed
    try:
        return int(raw)
    except ValueError:
        print("  Not an integer, using it as a text seed.")
        return raw

def _finish(ds, out, show=None):
    """Save the dataset and print its exact location."""
    path = save_dataset(ds, out)

    print("  ->", describe(ds))
    print("Saved at:", os.path.abspath(path))

    if show:
        show(ds)

def prompt_johnson(out, with_cycle=False, show=None):
    n = ask("Number of vertices", int, 2, 5000, 20)
    max_e = n * (n - 1)
    print(f"  Edges below {n} give a graph with unreachable pairs; "
          f"{n} or more can be made strongly connected.")
    e = ask("Number of edges", int, 1, max_e, min(max_e, 2 * n))
    connected = e >= n and ask_yes_no("Force strongly connected?", True)
    ds = johnson_dataset(n, e, ask_seed(), connected, with_cycle)
    _finish(ds, out, show)


def prompt_grid(out, show=None):
    rows = ask("Rows", int, 2, 1000, 100)
    cols = ask("Columns", int, 2, 1000, 100)
    p = ask("Obstacle probability", float, 0.0, 0.6, 0.0)
    start = (ask("Start row", int, 0, rows - 1, 0), ask("Start col", int, 0, cols - 1, 0))
    goal = (ask("Goal row", int, 0, rows - 1, rows - 1),
            ask("Goal col", int, 0, cols - 1, cols - 1))
    ds = grid_dataset(rows, cols, p, ask_seed(), start, goal)
    _finish(ds, out, show)


def prompt_geometric(out, show=None):
    n = ask("Number of nodes", int, 3, 5000, 200)
    isolate = ask_yes_no("Leave the last node unreachable (tests 'inf')?", False)
    m = n - 1 if isolate else n
    max_extra = m * (m - 1) // 2 - (m - 1)
    extra = ask("Extra undirected edges beyond the spanning tree", int, 0,
                max_extra, min(max_extra, 2 * n))
    k = ask("Number of random queries", int, 1, 50, 3)
    ds = geometric_dataset(n, extra, ask_seed(), isolate, k)
    _finish(ds, out, show)


MAIN_MENU = """
========== DATASET GENERATOR ==========
1) Generate Random Datasets
2) Generate Test Cases
3) View Saved Datasets
0) Exit"""

RANDOM_MENU = """
========== RANDOM DATASETS ==========
1) Johnson - Random weighted graph
2) Johnson - Negative cycle graph
3) A* - Random grid
4) A* - Random geometric graph
0) Back"""

TEST_MENU = """
========== TEST CASES ==========
1) Johnson - Negative Weights (No Negative Cycle)
2) Johnson - Negative Cycle Detection
3) A* - 100x100 Grid (A* vs Dijkstra)
4) Generate All Test Cases
0) Back"""


def test_cases():
    """Return the three fixed test datasets for algorithm verification.

    The A* dataset contains an expected path distance of 198, but generating
    it does not run the user's A* or Dijkstra algorithms or count expansions.
    """
    tc1, tc2 = fixed_cases()
    tc3 = grid_dataset(100, 100, obstacle_prob=0.0, seed=0,
                       start=(0, 0), goal=(99, 99),
                       name="astar_grid100_open")
    return [tc1, tc2, tc3]


def _save_by_algorithm(ds, root, show=None):
    """Save interactive datasets by mode so both collaborators stay organized."""
    target = os.path.join(root, ds["mode"])
    _finish(ds, target, show)


def random_datasets_menu(out=SIM_DIR, show=None):
    """Prompt for custom Johnson/A* graphs; 0 returns to main menu."""
    while True:
        print(RANDOM_MENU)
        choice = input("Choose a random dataset [0-4]: ").strip()
        if choice in ("0", "q", ""):
            return
        if choice == "1":
            prompt_johnson(os.path.join(out, "johnson"), show=show)
        elif choice == "2":
            prompt_johnson(os.path.join(out, "johnson"), with_cycle=True,
                           show=show)
        elif choice == "3":
            prompt_grid(os.path.join(out, "astar"), show=show)
        elif choice == "4":
            prompt_geometric(os.path.join(out, "astar"), show=show)
        else:
            print("  Unknown choice. Choose 0-4.")


def test_cases_menu(out=SIM_DIR, show=None):
    """Save selected predefined test case(s); 0 returns to main menu."""
    while True:
        print(TEST_MENU)
        choice = input("Choose a test case [0-4]: ").strip()
        if choice in ("0", "q", ""):
            return
        if choice in ("1", "2", "3", "4"):
            cases = test_cases() if choice in ("3", "4") else fixed_cases()
            selected = cases if choice == "4" else [cases[int(choice) - 1]]
            for ds in selected:
                _save_by_algorithm(ds, out, show)
            if choice in ("3", "4"):
                print("  Note: the 100x100 A* test dataset has expected distance 198.")
                print("  Run your A* and Dijkstra implementations to compare expanded nodes.")
        else:
            print("  Unknown choice. Choose 0-4.")


def view_saved_datasets(out=SIM_DIR):
    """Inspect existing .txt graph datasets without modifying them."""
    ds = choose_dataset(out)
    if ds is None:
        return
    print("\n", describe(ds))
    print("First 10 edges:", ds["edges"][:10])
    if ds["mode"] == "johnson":
        if ds["expected"] == "NEGATIVE_CYCLE":
            print("Expected result: NEGATIVE_CYCLE")
        elif isinstance(ds["expected"], list):
            print(f"Expected result: {len(ds['expected'])} x "
                  f"{len(ds['expected'])} distance matrix (.expected file)")
        else:
            print("Expected result: not saved for this graph size")
    elif ds["queries"]:
        print("A* queries (start, goal, expected distance):", ds["queries"][:5])


def interactive(out=SIM_DIR, show=None):
    """Organized two-level menu; keeps random datasets and test cases separate.

    Optionally pass show=<function> to draw each newly generated dataset.
    """
    while True:
        print(MAIN_MENU)
        choice = input("Choose an option [0-3]: ").strip()
        if choice in ("0", "q", ""):
            print("Done.")
            return
        if choice == "1":
            random_datasets_menu(out, show=show)
        elif choice == "2":
            test_cases_menu(out, show=show)
        elif choice == "3":
            view_saved_datasets(out)
        else:
            print("  Unknown choice. Choose 0-3.")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _rc(text):
    r, c = text.split(",")
    return int(r), int(c)


def cli(argv):
    ap = argparse.ArgumentParser(description="Johnson / A* dataset generator "
                                 "(run with no arguments for interactive prompts)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=SIM_DIR)
        p.add_argument("--seed", type=int, default=None,
                       help="omit for a random seed (saved in the file header)")
        p.add_argument("--name", default=None)

    p = sub.add_parser("johnson", help="random Johnson graph")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--edges", type=int, required=True)
    p.add_argument("--disconnected", action="store_true",
                   help="do not force strong connectivity")
    p.add_argument("--negative-cycle", action="store_true")
    common(p)
    p = sub.add_parser("grid", help="A* grid map")
    p.add_argument("--rows", type=int, default=100)
    p.add_argument("--cols", type=int, default=100)
    p.add_argument("--obstacles", type=float, default=0.0)
    p.add_argument("--start", type=_rc, help="row,col (default 0,0)")
    p.add_argument("--goal", type=_rc, help="row,col (default opposite corner)")
    common(p)
    p = sub.add_parser("geometric", help="A* graph on random 2-D points")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--extra-edges", type=int, required=True)
    p.add_argument("--queries", type=int, default=3)
    p.add_argument("--isolate-last", action="store_true")
    common(p)
    p = sub.add_parser("suite", help="original 10-dataset suite")
    p.add_argument("--out", default=SIM_DIR)
    p = sub.add_parser("tests", help="generate the three predefined test cases")
    p.add_argument("--out", default=SIM_DIR)

    a = ap.parse_args(argv)
    if a.cmd == "suite":
        return full_suite(a.out)
    if a.cmd == "tests":
        for ds in test_cases():
            _save_by_algorithm(ds, a.out)
        return
    try:
        if a.cmd == "johnson":
            ds = johnson_dataset(a.nodes, a.edges, a.seed, not a.disconnected,
                                 a.negative_cycle, a.name)
        elif a.cmd == "grid":
            ds = grid_dataset(a.rows, a.cols, a.obstacles, a.seed, a.start,
                              a.goal, a.name)
        else:
            ds = geometric_dataset(a.nodes, a.extra_edges, a.seed, a.isolate_last,
                                   a.queries, a.name)
    except ValueError as e:
        ap.error(str(e))
    save_dataset(ds, a.out)
    print(describe(ds))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--suite":  # backward-compatible alias
        args = ["suite"] + args[1:]
    cli(args) if args else interactive()
