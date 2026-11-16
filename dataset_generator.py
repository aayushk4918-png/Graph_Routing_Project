"""Interactive CSV dataset generator. Python 3.9+, standard library only.

Run: python graph_generator.py
Optional batch mode: python graph_generator.py --presets
Importing this file never starts the menus.

All CSVs are saved directly in one Datasets folder beside this script,
independently of the terminal working directory. No algorithm subfolders are created.
Each dataset is a named CSV. Existing files are never overwritten.
Use menu option 3 to copy old datasets into these shared folders.
The notebook imports DATASETS_DIR from this file as its single path source.

Generated node IDs run from 0 to N-1 for both algorithms.
Johnson node section: node_id (no coordinates generated or saved).
A* node section: node_id,x_coordinate,y_coordinate.
Shared edge section: source_node,target_node,weight.
Random datasets include all N*N expected distances (self=0, unreachable=inf).
These are calculated at save time from the final edge weights. Large random
graphs may take substantial time and disk space. For the negative-cycle test, -inf marks pairs whose cost is unbounded below;
Johnson must still reject that graph with Negative Cycle Detected.
The only other CSV section is Expected Distances. Metadata and Queries are not saved.
Choose source and goal when running your algorithm or simulation.
The Johnson credit cases use the original labels 0..3.
Grid coordinates still start at (0,0); grid node IDs start at 0.

Future notebooks can import load_dataset and heuristic from this same file.
The loader returns nodes={id: None} for Johnson and nodes={id: (x,y)} for A*.
For Johnson visualization, assign a layout in the notebook, not in the CSV.
A* uses edge weights for actual path cost and coordinates for its heuristic.
On the 100x100 grid, expected cost is 198; priority (f,h,node_id) gives useful
heuristic tie-breaking. Expansion savings are not guaranteed by data alone.
"""
import argparse
import csv
import math
import random
import shutil
import hashlib
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / 'Datasets'
def ensure_dataset_folders():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


def list_saved_datasets():
    ensure_dataset_folders()
    return sorted(p for p in DATASETS_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() == '.csv')


SECTIONS = {
    '# Nodes': ['node_id', 'x_coordinate', 'y_coordinate'],
    '# Edges': ['source_node', 'target_node', 'weight'],
    '# Expected Distances': ['source_node', 'target_node', 'expected_distance'],
}


def graph(nodes, edges, queries, **metadata):
    """Nodes: {id: (x,y) or None}; edges: [(u,v,w)]; queries: [(s,t,expected)]."""
    return dict(nodes=nodes, edges=edges, queries=queries, expected=[],
                metadata={str(k): str(v) for k, v in metadata.items()})


def johnson_case(negative_cycle=False):
    edges = [(0, 1, -2), (1, 2, -1), (2, 0, 4), (0, 3, 3), (2, 3, 1)]
    if negative_cycle:
        edges.append((2, 1, -2))
    g = graph({u: None for u in range(4)},
              edges, [(0, 3, '' if negative_cycle else -2)], algorithm='johnson',
              kind='johnson_negative_cycle' if negative_cycle else 'johnson_negative_edges',
              heuristic='none', expected_negative_cycle=str(negative_cycle).lower(),
              expected_error='Negative Cycle Detected' if negative_cycle else '',
              coordinate_role='none')
    if not negative_cycle:
        matrix = [[0, -2, -3, -2], [3, 0, -1, 0], [4, 2, 0, 1],
                  [math.inf, math.inf, math.inf, 0]]
        g['expected'] = [(s, t, matrix[s][t]) for s in range(4) for t in range(4)]
    return g


def grid_graph(rows=100, cols=100, start=0, goal=None):
    """Obstacle-free 4-neighbor unit-cost grid; each direction is an arc."""
    if rows < 1 or cols < 1:
        raise ValueError('Grid dimensions must be positive.')
    n = rows * cols
    goal = n - 1 if goal is None else goal
    if not (0 <= start < n and 0 <= goal < n):
        raise ValueError('Start and goal must be valid node IDs.')
    nodes = {r * cols + c: (c, r) for r in range(rows) for c in range(cols)}
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if rr < rows and cc < cols:
                    v = rr * cols + cc
                    edges.extend([(u, v, 1), (v, u, 1)])
    sx, sy = nodes[start]
    tx, ty = nodes[goal]
    return graph(nodes, edges, [(start, goal, abs(sx-tx)+abs(sy-ty))],
                 kind='grid', algorithm='astar', rows=rows, cols=cols, heuristic='manhattan',
                 astar_tie_break='lowest_f_then_lowest_h_then_node_id',
                 expected_negative_cycle='false', coordinate_role='heuristic_and_visualization')


def random_graph(n, m, seed=42, query_count=1, ensure_connected=True, extra_max=10, spatial=True, max_weight=20):
    """Directed simple graph, exactly m arcs; no self-loops or duplicates.

    A directed Hamiltonian cycle guarantees strong connectivity when requested.
    Sampling integer edge indices avoids allocating all n*(n-1) candidates.
    For A*, weights >= Euclidean distance make that heuristic consistent.
    For Johnson (spatial=False), no coordinates are generated; weights are 1..max_weight.
    """
    if n < 1 or not 0 <= m <= n * (n - 1):
        raise ValueError('Require n >= 1 and 0 <= m <= n*(n-1).')
    if ensure_connected and n > 1 and m < n:
        raise ValueError('Strong connectivity requires at least n directed edges.')
    if query_count < 1 or extra_max < 0 or max_weight < 1:
        raise ValueError('Query count and max weight must be positive; extra cost nonnegative.')
    rng = random.Random(seed)
    nodes = {u: ((rng.randrange(1000), rng.randrange(1000)) if spatial else None)
             for u in range(n)}
    # Integer lattice avoids rounding uncertainty in ceil(Euclidean distance).
    def encode(u, v):
        return u * (n - 1) + (v if v < u else v - 1)
    required = set()
    if ensure_connected and n > 1:
        order = list(range(n))
        rng.shuffle(order)
        required = {encode(order[i], order[(i+1) % n]) for i in range(n)}
    # Sample m indices once. Removing forced indices leaves enough to fill m.
    candidates = rng.sample(range(n * (n - 1)), m)
    selected = sorted(required)
    selected.extend([e for e in candidates if e not in required][:m-len(required)])
    edges = []
    for index in sorted(selected):
        u, offset = divmod(index, n - 1)
        v = offset if offset < u else offset + 1
        if spatial:
            x, y = nodes[u]
            xx, yy = nodes[v]
            squared = (x-xx)**2 + (y-yy)**2
            root = math.isqrt(squared)
            ceil_distance = root + (root * root != squared)
            weight = max(1, ceil_distance) + rng.randint(0, extra_max)
        else:
            weight = rng.randint(1, max_weight)
        edges.append((u, v, weight))
    queries = []
    for _ in range(query_count):
        s, t = rng.sample(range(n), 2) if n > 1 else (0, 0)
        queries.append((s, t, ''))
    return graph(nodes, edges, queries,
                 kind='random_spatial' if spatial else 'random_weighted', seed=seed,
                 algorithm='astar' if spatial else 'johnson',
                 heuristic='euclidean' if spatial else 'none', strongly_connected_guaranteed=str(ensure_connected).lower(),
                 extra_cost_max=extra_max if spatial else '', max_base_weight=max_weight if not spatial else '',
                 expected_negative_cycle='false',
                 coordinate_role='heuristic_and_visualization' if spatial else 'none')


def validate(g):
    nodes = g['nodes']
    base = min(nodes, default=0)  # Also accept earlier saved CSVs numbered from 1.
    if not nodes or base not in (0, 1) or set(nodes) != set(range(base, base + len(nodes))):
        raise ValueError('Node IDs must be contiguous integers starting at zero or one.')
    seen = set()
    for u, v, w in g['edges']:
        if u not in nodes or v not in nodes or u == v or (u, v) in seen:
            raise ValueError('Invalid, duplicate, or self-loop edge.')
        if not math.isfinite(w):
            raise ValueError('Edge weights must be finite.')
        seen.add((u, v))
    for s, t, _ in g['queries']:
        if s not in nodes or t not in nodes:
            raise ValueError('Invalid query endpoints.')


def reference_distances(g):
    """Floyd–Warshall all-pairs oracle. O(V³) time and O(V²) memory."""
    ids = sorted(g['nodes'])
    index = {node: i for i, node in enumerate(ids)}
    size = len(ids)
    distance = [[math.inf] * size for _ in ids]
    for i in range(size):
        distance[i][i] = 0
    for source, target, weight in g['edges']:
        i, j = index[source], index[target]
        distance[i][j] = min(distance[i][j], weight)

    # Permit one additional intermediate vertex on each pass.
    for k in range(size):
        via_k = distance[k]
        for row in distance:
            to_k = row[k]
            if to_k == math.inf:
                continue
            for j, from_k in enumerate(via_k):
                if from_k != math.inf:
                    candidate = to_k + from_k
                    if candidate < row[j]:
                        row[j] = candidate

    # Only pairs that can travel through a negative cycle are unbounded.
    cycles = [k for k in range(size) if distance[k][k] < 0]
    for k in cycles:
        reachable = [j for j in range(size) if distance[k][j] != math.inf]
        for row in distance:
            if row[k] != math.inf:
                for j in reachable:
                    row[j] = -math.inf

    for i, source in enumerate(ids):
        for j, target in enumerate(ids):
            yield source, target, distance[i][j]


def valid_dataset_name(name):
    return bool(name and name not in ('.', '..') and not name.endswith((' ', '.'))
                and not any(c in name for c in '<>:"/\\|?*')
                and not name.lower().endswith('.csv'))


def csv_algorithm(path):
    """Classify by the Nodes header, not by a possibly renamed filename."""
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        awaiting = False
        for row in csv.reader(handle):
            if not row: continue
            row = [cell.strip() for cell in row]
            if row == ['# Nodes']:
                awaiting = True
                continue
            if awaiting:
                if row == ['node_id']: return 'johnson'
                if row == ['node_id', 'x_coordinate', 'y_coordinate']: return 'astar'
                raise ValueError('Unrecognized Nodes header.')
    raise ValueError('No sectioned Nodes header found; use the current CSV format.')


def content_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def import_datasets(source_folder):
    """Copy old CSVs into the shared folders. Originals remain untouched.

    Identical content is skipped even if its filename differs. Name collisions
    receive numeric suffixes; algorithm is inferred from the Nodes header.
    """
    source = Path(source_folder).expanduser().resolve()
    if not source.is_dir(): raise ValueError('Choose an existing folder containing CSVs.')
    ensure_dataset_folders()
    known = {content_digest(path) for path in list_saved_datasets()}
    copied = []; skipped = 0; rejected = []
    candidates = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower()=='.csv')
    for path in candidates:
        try:
            algorithm = csv_algorithm(path)
            digest = content_digest(path)
            if digest in known:
                skipped += 1
                continue
            # Prefix the parent folder, since earlier versions named every CSV johnson.csv/astar.csv.
            raw_name = path.parent.name + '_' + path.stem
            name = ''.join('_' if c in '<>:"/\\|?*' else c for c in raw_name).strip(' .') or 'imported_dataset'
            destination = DATASETS_DIR / f'{name}.csv'
            suffix = 2
            while destination.exists():
                destination = destination.with_name(f'{name}_{suffix}.csv')
                suffix += 1
            shutil.copy2(path, destination)
            known.add(digest); copied.append(destination)
        except (OSError, ValueError) as error:
            rejected.append((str(path), str(error)))
    print(f'Imported {len(copied)} CSVs; skipped {skipped} identical files; rejected {len(rejected)} files.')
    for path in copied: print('  Saved:', path)
    for path, error in rejected: print('  Not imported:', path, '-', error)
    return copied, skipped, rejected


def import_menu():
    print('\nIMPORT EXISTING DATASETS')
    print('Copies compatible CSVs into Datasets. Originals are kept; duplicates are skipped.')
    source = ask_text('Folder containing your old CSVs', str(Path(__file__).resolve().parent / 'datasets'))
    try: import_datasets(source)
    except (OSError, ValueError) as error: print('Import failed:', error)


def save_dataset(g, output, name):
    """Write one named CSV directly in Datasets; return its path.

    The interactive/batch entry points always supply DATASETS_DIR. The output
    parameter exists for isolated tests and notebook use.
    """
    validate(g)
    algorithm = g['metadata'].get('algorithm')
    if algorithm not in ('johnson', 'astar'):
        raise ValueError('Dataset must select johnson or astar.')
    if algorithm == 'astar' and any(w < 0 for _, _, w in g['edges']):
        raise ValueError('Standard A* requires nonnegative weights.')
    if algorithm == 'astar' and any(xy is None for xy in g['nodes'].values()):
        raise ValueError('A* requires node coordinates.')
    if not valid_dataset_name(name):
        raise ValueError('Dataset name must be a single folder name.')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    folder = root
    destination = folder / f'{name}.csv'
    suffix = 2
    while destination.exists():
        destination = folder / f'{name}_{suffix}.csv'
        suffix += 1
    headers = dict(SECTIONS)
    headers['# Nodes'] = ['node_id'] if algorithm == 'johnson' else SECTIONS['# Nodes']
    node_rows = ((u,) for u in sorted(g['nodes'])) if algorithm == 'johnson' else (
        (u, *xy) for u, xy in sorted(g['nodes'].items()))
    expected = list(g['expected'])
    known_pairs = {(u, v) for u, v, _ in expected}
    for u, v, distance in g['queries']:
        if distance not in ('', None) and (u, v) not in known_pairs:
            expected.append((u, v, distance))
            known_pairs.add((u, v))
    if g['metadata'].get('kind') != 'grid' or not expected:
        print(f'Computing {len(g["nodes"]) ** 2:,} expected distances using Floyd–Warshall...')
        expected = reference_distances(g)
    temporary = destination.with_suffix('.csv.tmp')
    try:
        expected_count, preview = _write_sections(temporary, headers, node_rows, g['edges'], expected)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Saved {len(g["nodes"])} vertices, {len(g["edges"])} directed edges:')
    print(f'  {destination.resolve()}')
    print(f'  Expected Distances: {expected_count:,} rows saved inside this CSV.')
    print('  Find the # Expected Distances section after the edges. Preview:')
    print('  source_node,target_node,expected_distance')
    for row in preview:
        print('  ' + ','.join(map(str, row)))
    if g['metadata'].get('expected_negative_cycle') == 'true':
        print('  -inf means no finite shortest distance (reachable negative cycle).')
        print('  Expected Johnson behavior: Negative Cycle Detected.')
    return destination


def _write_sections(path, headers, node_rows, edges, expected):
    expected_count = 0
    preview = []
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        sections = {
            '# Nodes': node_rows,
            '# Edges': edges,
            '# Expected Distances': expected,
        }
        for section, records in sections.items():
            writer.writerow([section])
            writer.writerow(headers[section])
            if section == '# Expected Distances':
                for row in records:
                    writer.writerow(row)
                    expected_count += 1
                    if len(preview) < 5:
                        preview.append(row)
            else:
                writer.writerows(records)
            writer.writerow([])
    if expected_count == 0:
        raise ValueError('Expected Distances cannot be empty; dataset was not saved.')
    return expected_count, preview


def load_dataset(path, for_execution=True):
    """Shared loader for future notebooks; expected values never enter adjacency.

    Johnson nodes map to None; A* nodes map to coordinate pairs.
    Johnson negative-cycle cases are deliberately loadable for detection tests.
    The optional for_execution argument is retained for visualization workflows.
    """
    g = graph({}, [], [])
    with Path(path).open(newline='', encoding='utf-8') as f:
        headers = dict(SECTIONS)
        section = None
        needs_header = False
        seen_sections = set()
        for cells in csv.reader(f):
            if not cells or all(not cell.strip() for cell in cells):
                continue
            if cells[0].startswith('#'):
                if needs_header:
                    raise ValueError('Missing section header.')
                if len(cells) != 1 or cells[0] not in SECTIONS or cells[0] in seen_sections:
                    raise ValueError('Unknown or duplicate section marker.')
                section = cells[0]
                seen_sections.add(section)
                needs_header = True
                continue
            if section is None:
                raise ValueError('Expected a section marker such as # Nodes.')
            if needs_header:
                if section == '# Nodes' and cells == ['node_id']:
                    headers[section] = cells
                if cells != headers[section]:
                    raise ValueError(f'Incorrect column names in {section}.')
                needs_header = False
                continue
            if len(cells) != len(headers[section]):
                raise ValueError(f'Incorrect column count in {section}.')
            row = dict(zip(headers[section], cells))
            if section == '# Nodes':
                u = int(row['node_id'])
                if u in g['nodes']:
                    raise ValueError('Duplicate node ID.')
                g['nodes'][u] = ((float(row['x_coordinate']), float(row['y_coordinate']))
                                 if 'x_coordinate' in row else None)
            elif section == '# Edges':
                g['edges'].append((int(row['source_node']), int(row['target_node']), float(row['weight'])))
            else:
                expected = float(row['expected_distance']) if row['expected_distance'] else None
                g['expected'].append(
                    (int(row['source_node']), int(row['target_node']), expected))
        if needs_header or seen_sections != set(SECTIONS):
            raise ValueError('Missing section or column header.')
    validate(g)
    # Infer runtime settings from the graph; none are read from CSV metadata.
    algorithm = 'johnson' if headers['# Nodes'] == ['node_id'] else 'astar'
    kind = 'none'
    if algorithm == 'astar':
        if for_execution and any(w < 0 for _, _, w in g['edges']):
            raise ValueError('Standard A* requires nonnegative weights.')
        # Use a geometric heuristic only when every edge satisfies its bound.
        manhattan_ok = euclidean_ok = True
        for u, v, w in g['edges']:
            x, y = g['nodes'][u]
            xx, yy = g['nodes'][v]
            manhattan_ok &= abs(x-xx) + abs(y-yy) <= w
            euclidean_ok &= math.hypot(x-xx, y-yy) <= w
        kind = 'manhattan' if manhattan_ok else ('euclidean' if euclidean_ok else 'none')
    g['metadata'].update(algorithm=algorithm, heuristic=kind)
    g['adjacency'] = {u: [] for u in g['nodes']}
    for u, v, w in g['edges']:
        g['adjacency'][u].append((v, w))
    return g


def heuristic(g, node, goal):
    """Goal-dependent estimate; do not cache across different query goals."""
    kind = g['metadata'].get('heuristic', 'none')
    if kind == 'none':
        return 0.0
    x, y = g['nodes'][node]
    gx, gy = g['nodes'][goal]
    if kind == 'manhattan':
        return abs(x-gx) + abs(y-gy)
    if kind == 'euclidean':
        return math.hypot(x-gx, y-gy)
    return 0.0


class BackToMain(Exception):
    """Cancel the current configuration without writing files."""


def ask_int(label, low, high, default=None):
    while True:
        hint = f' [{default}]' if default is not None else ''
        text = input(f'{label}{hint} ({low}..{high}; B = main menu): ').strip()
        if text.lower() == 'b':
            raise BackToMain
        try:
            value = int(text) if text else default
            if value is not None and low <= value <= high:
                return value
        except ValueError:
            pass
        print('Please enter a number within the shown range.')


def menu(title, options, back='Back'):
    print(f'\n{title}\n' + '-' * len(title))
    for index, label in enumerate(options, 1):
        print(f'  {index}. {label}')
    print(f'  0. {back}')
    # Menus deliberately have no default: Enter never chooses a dataset.
    return ask_int('Choose an option', 0, len(options))


def ask_text(label, default):
    text = input(f'{label} [{default}] (B = main menu): ').strip()
    if text.lower() == 'b':
        raise BackToMain
    return text or default


def presets(output):
    save_dataset(johnson_case(), output, 'johnson_negative_edges')
    save_dataset(johnson_case(True), output, 'johnson_negative_cycle')
    save_dataset(grid_graph(), output, 'grid_100x100')


def review_and_save(g, output, name):
    print('\nREVIEW DATASET')
    print(f'  Type: {g["metadata"]["kind"]}')
    print(f'  Vertices: {len(g["nodes"]):,}')
    print(f'  Directed edges: {len(g["edges"]):,}')
    if g['metadata'].get('kind', '').startswith('random_'):
        print(f'  Expected distances: all {len(g["nodes"]) ** 2:,} ordered node pairs')
        if len(g['nodes']) > 500:
            print('  All-pairs verification can take time and produce a large CSV, because Floyd–Warshall takes cubic time and quadratic memory.')
    algorithm = g['metadata']['algorithm']
    print(f'  Algorithm: {algorithm}')
    print(f'  Node IDs: 0 to {len(g["nodes"]) - 1}')
    print('  Coordinates: ' + ('included' if algorithm == 'astar' else 'not generated'))
    while True:
        name = input('Enter dataset name (without .csv; B = main menu): ').strip()
        if name.lower() == 'b':
            raise BackToMain
        if valid_dataset_name(name):
            break
        print('A dataset name is required. Do not include a folder path or .csv extension.')
    print(f'  Destination: {Path(output) / (name + ".csv")}')
    choice = menu('Ready to generate?', ['Generate and save this dataset'], 'Cancel')
    if choice == 1:
        save_dataset(g, output, name)
    else:
        print('Cancelled. No files were written.')


def test_cases_menu(output):
    while True:
        algorithm = menu('TEST CASES', ["Johnson’s algorithm", 'A* search'])
        if algorithm == 0:
            return
        while True:
            if algorithm == 1:
                case = menu('TEST CASES / JOHNSON', [
                    'Negative weights without a negative cycle (4 vertices, 5 edges)',
                    'Negative cycle detection (4 vertices, 6 edges)'])
                if case == 0:
                    break
                g = johnson_case(case == 2)
                name = g['metadata']['kind']
            else:
                case = menu('TEST CASES / A*', [
                    'A* vs Dijkstra: 100 x 100 grid, corner to corner'])
                if case == 0:
                    break
                print('Fixed credit case: 10,000 vertices, 39,600 directed edges, expected cost 198.')
                g, name = grid_graph(), 'grid_100x100'
            review_and_save(g, output, name)


def configure_random(algorithm):
    print(f'\nGENERATE DATASETS / {algorithm.upper()} / RANDOM GRAPH')
    print('Enter graph size first. Edges are directed; no duplicates or self-loops.')
    n = ask_int('1. Number of vertices', 1, 10000)
    connected = bool(ask_int('2. Guarantee strong connectivity? 1=yes, 0=no', 0, 1))
    minimum = n if connected and n > 1 else 0
    maximum = min(n*(n-1), 1000000)
    m = ask_int('3. Number of directed edges', minimum, maximum)
    negative = False
    if algorithm == 'Johnson' and m:
        negative = ask_int('4. Weight type: 1=positive, 2=negative edges without negative cycles', 1, 2) == 2
    seed = ask_int('Random seed (same settings + seed reproduce the graph)', 0, 2147483647, 42)
    q = 1  # Internal placeholder only; queries are never exported.
    if algorithm == 'Johnson':
        weight_max = ask_int('Maximum positive base edge weight', 1, 1000000, 20)
        g = random_graph(n, m, seed, q, connected, spatial=False, max_weight=weight_max)
    else:
        extra = ask_int('Maximum extra cost above geometric distance', 0, 1000000, 10)
        g = random_graph(n, m, seed, q, connected, extra)
    if negative:
        # Potential differences telescope to zero around every cycle. Starting
        # from positive weights therefore cannot introduce a negative cycle.
        source = g['edges'][0][0]
        potential = {u: 0 for u in g['nodes']}
        potential[source] = max(w for _, _, w in g['edges']) + 1
        g['edges'] = [(u, v, w + potential[v] - potential[u]) for u, v, w in g['edges']]
        g['metadata'].update(kind='random_negative_edges', heuristic='none',
                             coordinate_role='none',
                             weight_method='positive_base_plus_target_potential_minus_source_potential')
    return g, f'{algorithm.lower()}_random_v{n}_e{m}_seed{seed}'


def configure_grid():
    print('\nGENERATE DATASETS / A* / GRID')
    rows = ask_int('1. Grid rows', 1, 300)
    cols = ask_int('2. Grid columns', 1, 300)
    n = rows * cols
    edges = 2 * (rows*(cols-1) + cols*(rows-1))
    print(f'This four-neighbor grid has {n:,} vertices and {edges:,} directed edges.')
    print(f'Node ID = row * {cols} + column. Coordinates start at zero.')
    print('Choose source and goal later when running A*.')
    return grid_graph(rows, cols), f'astar_grid_{rows}x{cols}'


def generate_menu(output):
    while True:
        algorithm = menu('GENERATE DATASETS', ["Generate a dataset for Johnson", 'Generate a dataset for A*'])
        if algorithm == 0:
            return
        if algorithm == 1:
            g, name = configure_random('Johnson')
            review_and_save(g, output, name)
        else:
            while True:
                kind = menu('GENERATE DATASETS / A*', ['Random spatial graph (choose vertices and edges)',
                                                       'Grid map (choose rows and columns)'])
                if kind == 0:
                    break
                g, name = configure_random('Astar') if kind == 1 else configure_grid()
                review_and_save(g, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--presets', action='store_true', help='Explicit batch mode: generate all credit cases without prompts')
    args = parser.parse_args()
    ensure_dataset_folders()
    if args.presets:
        presets(DATASETS_DIR)
        return
    print('SHORTEST-PATH DATASET GENERATOR - SINGLE FOLDER EDITION')
    print('Shared dataset folder:', DATASETS_DIR)
    print('Choose a category, choose an algorithm, then configure and review.')
    print('No files are written until you select Generate and save.')
    while True:
        try:
            choice = menu('MAIN MENU', ['Test cases (predefined credit cases)',
                                        'Generate datasets (custom settings)',
                                        'Import existing datasets into shared folders'], 'Exit')
            if choice == 0:
                print('Goodbye.')
                return
            if choice == 1:
                test_cases_menu(DATASETS_DIR)
            elif choice == 2:
                generate_menu(DATASETS_DIR)
            else:
                import_menu()
        except BackToMain:
            print('Returned to main menu. No pending dataset was saved.')


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nGeneration stopped.')
    except OSError as error:
        raise SystemExit(f'Could not write/read dataset: {error}')



