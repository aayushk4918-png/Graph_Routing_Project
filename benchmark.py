"""Run beside johnson.py and Datasets/. Requires matplotlib.
CSV format: # Nodes, # Edges, # Expected Distances sections.
Solver contract: johnson(nodes, edges) -> (distance_dict, parents).
Run: python benchmark_johnson.py
"""
from pathlib import Path
import csv
import math
import statistics
import time
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_DIR / 'Datasets'
IMAGES_DIR = PROJECT_DIR / 'Images'


def read_csv(path):
    nodes, edges, expected = [], [], {}
    section = ''
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            first = row[0].strip()
            if first.startswith('#'):
                section = first[1:].strip().lower()
                continue
            if first.lower() in ('node_id', 'source_node', 'key', 'query_id'):
                continue
            if section == 'nodes':
                node = int(first)
                if node < 0 or node in nodes:
                    raise ValueError(f'invalid or duplicate node ID: {node}')
                nodes.append(node)
            elif section == 'edges':
                weight = float(row[2])
                if not math.isfinite(weight):
                    raise ValueError('edge weights must be finite')
                edges.append((int(first), int(row[1]), weight))
            elif section == 'expected distances':
                pair = (int(first), int(row[1]))
                if pair in expected:
                    raise ValueError(f'duplicate expected pair: {pair}')
                expected[pair] = float(row[2])
    if len(nodes) < 2:
        raise ValueError('density experiment requires at least two vertices')
    pairs = [(u, v) for u, v, _ in edges]
    if len(set(pairs)) != len(pairs):
        raise ValueError('parallel edges are not supported for this density experiment')
    if any(u not in nodes or v not in nodes or u == v for u, v in pairs):
        raise ValueError('unknown endpoints or self-loops')
    return nodes, edges, expected


def verify(nodes, distances, expected):
    required = {(u, v) for u in nodes for v in nodes}
    if set(expected) != required:
        raise ValueError('CSV must contain exactly V² expected distance pairs')
    if any(value == -math.inf for value in expected.values()):
        raise ValueError('negative-cycle dataset: use a separate cycle-detection test')
    mismatches = sum(
        not math.isclose(distances[u][v], expected[u, v], rel_tol=1e-9, abs_tol=1e-9)
        for u, v in required
    )
    if mismatches:
        raise ValueError(f'{mismatches} distance mismatches; excluded from plot')


def measure_runtime(solver, nodes, edges, expected, repeats):
    # Warm-up and correctness verification are outside the timer.
    distances, parents = solver(nodes, edges)
    verify(nodes, distances, expected)
    del distances, parents
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = solver(nodes, edges)
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
        del result
    return timings


def positive_integer(prompt, default):
    while True:
        value = input(f'{prompt} [{default}]: ').strip()
        try:
            number = int(value or default)
            if number > 0:
                return number
        except ValueError:
            pass
        print('Enter a positive integer.')


def main():
    try:
        from johnson import johnson
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f'Missing dependency: {error}')
        print('Place this file beside johnson.py; install matplotlib if missing.')
        return

    print(f'Dataset folder: {DATASETS_DIR}')
    files = sorted(DATASETS_DIR.glob('*.csv'))
    if not files:
        print('No CSV files found. Save datasets in the folder shown above.')
        return
    graphs = []
    for path in files:
        try:
            nodes, edges, expected = read_csv(path)
            graphs.append((path, nodes, edges, expected))
        except (ValueError, IndexError, OSError) as error:
            print(f'Skipping {path.name}: {error}')
    if not graphs:
        return
    sizes = sorted({len(g[1]) for g in graphs})
    print('Available vertex counts:', ', '.join(map(str, sizes)))
    vertices = positive_integer('Choose fixed vertex count', sizes[0])
    selected = [g for g in graphs if len(g[1]) == vertices]
    if not selected:
        print('No datasets with that vertex count.')
        return
    print('Datasets to benchmark:')
    for path, nodes, edges, _ in selected:
        print(f'  {path.name}: {len(edges)} edges, {100*len(edges)/(vertices*(vertices-1)):.2f}% density')
    repeats = positive_integer('Timed runs per dataset', 7)
    rows, grouped = [], defaultdict(list)
    for path, nodes, edges, expected in selected:
        try:
            times = measure_runtime(johnson, nodes, edges, expected, repeats)
        except Exception as error:
            print(f'Skipping {path.name}: {type(error).__name__}: {error}')
            continue
        edge_count = len(edges)
        density = 100 * edge_count / (vertices * (vertices - 1))
        median = statistics.median(times)
        grouped[edge_count].append(median)
        print(f'{path.name}: PASS ({vertices**2} pairs), median {median:.3f} ms')
        for run, elapsed in enumerate(times, 1):
            rows.append([path.name, vertices, edge_count, density, run, elapsed, median, 100])
    if not rows:
        print('No valid datasets to plot.')
        return
    IMAGES_DIR.mkdir(exist_ok=True)
    # Unique output names preserve earlier experiments.
    stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns() % 1000000000:09d}'
    stem = IMAGES_DIR / f'johnson_density_v{vertices}_{stamp}'
    csv_path = stem.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'vertices', 'edges', 'density_percent', 'run',
                         'runtime_ms', 'dataset_median_ms', 'correctness_percent'])
        writer.writerows(rows)
    edge_counts = sorted(grouped)
    densities = [100 * e / (vertices * (vertices - 1)) for e in edge_counts]
    medians = [statistics.median(grouped[e]) for e in edge_counts]
    fig, ax = plt.subplots(figsize=(9, 5))
    for e, density in zip(edge_counts, densities):
        ax.scatter([density] * len(grouped[e]), grouped[e], color='#94a3b8', alpha=.7)
    ax.plot(densities, medians, 'o-', color='#2563eb', linewidth=2,
            label='Median of dataset medians')
    ax.set(title=f"Johnson runtime vs density — {vertices} vertices",
           xlabel='Directed graph density (%)', ylabel='Median execution time (ms)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    image_path = stem.with_suffix('.png')
    fig.savefig(image_path, dpi=300)
    print(f'Timings saved: {csv_path}\nPlot saved: {image_path}')
    if len(edge_counts) < 2:
        print('Only one density found. Add datasets with other edge counts to compare densities.')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nBenchmark cancelled.')
