
import random
import statistics


def generate_graph(n, edge_count, seed=None):
    """
    Generate a directed weighted graph for Johnson's Algorithm.

    Parameters:
        n: Number of vertices
        edge_count: Exact number of directed edges
        seed: Random seed (None generates a new random graph)

    Returns:
        Adjacency-list dictionary:
        {
            0: [(destination, weight), ...],
            1: [(destination, weight), ...],
            ...
        }

    Properties:
        - Strongly connected
        - Contains negative edge weights
        - No negative-weight cycles
        - No self-loops or duplicate edges
        - Generates a new random graph every run
    """

    if n < 2:
        raise ValueError("Number of vertices must be at least 2")

    max_edges = n * (n - 1)

    if not n <= edge_count <= max_edges:
        raise ValueError(
            f"Edge count must be between {n} and {max_edges}"
        )

    # Different random graph each run when seed=None
    rng = random.Random(seed)

    # STEP 1: Create random vertex potentials
    potentials = [
        rng.randint(-10, 10) for _ in range(n)
    ]

    # Ensure the graph contains a negative edge
    potentials[0] = 10
    potentials[1] = -10

    # STEP 2: Create a directed ring
    # This guarantees strong connectivity
    edges = set()

    for u in range(n):
        v = (u + 1) % n
        edges.add((u, v))

    # STEP 3: Add random directed edges
    while len(edges) < edge_count:
        u = rng.randrange(n)
        v = rng.randrange(n)

        if u != v:
            edges.add((u, v))

    # STEP 4: Assign edge weights
    graph = {i: [] for i in range(n)}

    for u, v in sorted(edges):

        # Nonnegative base weight
        base_weight = rng.randint(0, 10)

        # Guarantee a negative edge: 0 -> 1
        if u == 0 and v == 1:
            base_weight = 0

        # Potential-based weight assignment
        weight = (
            base_weight
            + potentials[v]
            - potentials[u]
        )

        graph[u].append((v, weight))

    return graph


def graph_statistics(graph):
    """
    Calculate descriptive statistics for a graph.
    """

    vertices = len(graph)

    weights = [
        weight
        for neighbors in graph.values()
        for _, weight in neighbors
    ]

    edges = len(weights)

    negative_edges = sum(
        1 for weight in weights if weight < 0
    )

    possible_edges = vertices * (vertices - 1)

    density = (
        edges / possible_edges
        if possible_edges > 0 else 0
    )

    return {
        "vertices": vertices,
        "edges": edges,
        "density": density,
        "min_weight": min(weights) if weights else None,
        "max_weight": max(weights) if weights else None,
        "mean_weight": (
            statistics.mean(weights) if weights else None
        ),
        "std_weight": (
            statistics.pstdev(weights) if weights else None
        ),
        "negative_edges": negative_edges,
        "negative_percentage": (
            100 * negative_edges / edges
            if edges > 0 else 0
        )
    }


def generate_negative_cycle_graph():
    """
    Professor's negative-cycle verification case.
    This graph must be rejected by Johnson's Algorithm.
    """

    return {
        0: [(1, -2), (3, 3)],
        1: [(2, -1)],
        2: [(0, 4), (3, 1), (1, -2)],
        3: []
    }


if __name__ == "__main__":

    # Generate a NEW random graph every run
    graph = generate_graph(
        n=10,
        edge_count=30,
        seed=None
    )

    print("========== GENERATED GRAPH ==========")

    for vertex, neighbors in graph.items():
        print(f"{vertex}: {neighbors}")

    print("\n========== GRAPH STATISTICS ==========")

    stats = graph_statistics(graph)

    for key, value in stats.items():
        print(f"{key}: {value}")
