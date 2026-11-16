
"""
Johnson's Algorithm for All-Pairs Shortest Paths
================================================

Description:
    This program implements Johnson's Algorithm to calculate the shortest
    paths between every pair of vertices in a weighted, directed graph.

    Johnson's Algorithm supports graphs containing negative edge weights,
    provided that the graph does not contain a negative-weight cycle.

    The algorithm combines Bellman-Ford and Dijkstra's algorithms:

    1. Bellman-Ford computes a potential value for each vertex and detects
       negative-weight cycles.
    2. Edge reweighting uses these potentials to make all edge weights
       non-negative without changing the shortest paths.
    3. Dijkstra's algorithm runs from every vertex to determine shortest
       path distances in the reweighted graph.
    4. The computed distances are converted back to their original values.

    The program also supports:
        - Reading graph data from a CSV file.
        - Selecting a dataset interactively or through command-line arguments.
        - Reconstructing the shortest path between two selected vertices.
        - Displaying the all-pairs distance matrix for small graphs.
        - Comparing calculated distances against expected CSV results.

Time Complexity:
    O(VE + V(E + V) log V), using a binary heap for Dijkstra's algorithm.

Space Complexity:
    O(V^2 + V + E), including the stored all-pairs distances and paths.

    V = number of vertices
    E = number of directed edges

Usage:
    python johnson.py graph.csv --source 0 --goal 4

    Alternatively, run without arguments to select a CSV file
    interactively and enter the source and destination vertices.
"""

import argparse
import csv
import heapq
import math
from pathlib import Path


class NegativeCycleError(ValueError):
    """
    Exception raised when a negative-weight cycle is detected.

    A negative-weight cycle makes shortest-path distances undefined
    for paths that can repeatedly traverse the cycle, because the
    total path cost can decrease without limit.

    Johnson's Algorithm cannot operate on graphs containing
    negative-weight cycles.
    """
    pass


def read_dataset(path):
    """
    Read graph vertices, weighted edges, and expected distances from a CSV.

    The input CSV is divided into sections identified by:
        # Nodes
        # Edges
        # Expected Distances

    Each section contains a header row followed by data rows.

    The Nodes section must contain non-negative, unique integer IDs.
    The Edges section contains directed edges and integer weights.
    Expected Distances is optional and is used for result verification.

    Args:
        path (str or pathlib.Path):
            Location of the CSV dataset.

    Returns:
        tuple:
            nodes (list[int]):
                Sorted list of vertex IDs.

            edges (list[tuple[int, int, int]]):
                Directed edges represented as
                (source, destination, weight).

            expected (dict[tuple[int, int], float]):
                Expected shortest-path distances, indexed by
                (source, destination).

    Raises:
        ValueError:
            If a vertex ID is invalid or duplicated, no nodes are
            found, or an edge references an unknown vertex.

        OSError:
            If the CSV file cannot be opened or read.
    """

    # Store node IDs and their CSV line numbers for duplicate detection.
    node_lines, edges, expected = {}, [], {}

    # Track the currently active CSV section and its column headers.
    section = header = None

    # utf-8-sig supports UTF-8 files with or without a byte-order mark.
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)

        for row in reader:
            # Remove surrounding whitespace from every CSV field.
            row = [cell.strip() for cell in row]

            # Ignore completely empty rows.
            if not any(row):
                continue

            # A row beginning with '#' identifies a new data section.
            if row[0].startswith("#"):
                section, header = row[0], None
                continue

            # The first nonempty row in a section defines column names.
            if header is None:
                header = row
                continue

            # Associate each column header with its corresponding value.
            fields = dict(zip(header, row))

            if section == "# Nodes":
                location = f"{Path(path).name}, line {reader.line_num}"
                raw = fields.get("node_id", "")

                # Convert the node identifier to an integer.
                try:
                    node = int(raw)
                except ValueError:
                    raise ValueError(
                        f"{location}: expected an integer node_id, "
                        f"found {raw!r}."
                    ) from None

                # Node identifiers must be non-negative.
                if node < 0:
                    raise ValueError(
                        f"{location}: node ID {node} must be 0 or greater."
                    )

                # Reject duplicate node identifiers.
                if node in node_lines:
                    raise ValueError(
                        f"{location}: duplicate node ID {node} in # Nodes "
                        f"(first listed on line {node_lines[node]})."
                    )

                # Record the line on which the node was declared.
                node_lines[node] = reader.line_num

            elif section == "# Edges":
                # Store each directed weighted edge as (u, v, weight).
                edges.append((
                    int(fields["source_node"]),
                    int(fields["target_node"]),
                    int(fields["weight"])
                ))

            elif section == "# Expected Distances" and fields["expected_distance"]:
                # Store reference distances for later verification.
                pair = (
                    int(fields["source_node"]),
                    int(fields["target_node"])
                )
                expected[pair] = float(fields["expected_distance"])

    # Sort vertices to produce a consistent output order.
    nodes = sorted(node_lines)

    # Ensure that the dataset contains at least one vertex.
    if not nodes:
        raise ValueError("No nodes found in the CSV.")

    # Verify that every edge endpoint exists in the node list.
    if any(u not in node_lines or v not in node_lines for u, v, _ in edges):
        raise ValueError("An edge references an unknown node.")

    return nodes, edges, expected


def bellman_ford(nodes, edges):
    """
    Compute vertex potentials using the Bellman-Ford algorithm.

    In Johnson's Algorithm, Bellman-Ford determines a potential h(v)
    for every vertex. These potentials are later used to reweight
    edges so that all transformed edge weights are non-negative.

    Conceptually, a temporary super-source is connected to every
    vertex by a zero-weight edge. Instead of explicitly creating
    that vertex, all potentials are initialized to zero.

    Each edge is relaxed up to |V| - 1 times, with early termination
    if an iteration produces no changes.

    Args:
        nodes (iterable[int]):
            Collection of graph vertices.

        edges (iterable[tuple[int, int, int]]):
            Directed weighted edges (u, v, weight).

    Returns:
        dict[int, int]:
            Potential h[v] for every vertex v.

    Raises:
        NegativeCycleError:
            If the graph contains a negative-weight cycle.

    Time Complexity:
        O(VE)

    Space Complexity:
        O(V)
    """

    # Initialize every potential to zero.
    # This is equivalent to relaxing zero-weight edges from a
    # temporary super-source to all vertices.
    h = dict.fromkeys(nodes, 0)

    # A graph without negative cycles requires at most V-1 passes
    # after the conceptual super-source initialization.
    for _ in range(len(nodes) - 1):
        changed = False

        # Relax every directed edge (u, v).
        for u, v, weight in edges:
            candidate = h[u] + weight

            # Update the potential when a lower cost is found.
            if candidate < h[v]:
                h[v] = candidate
                changed = True

        # Stop early if no potential changed during this pass.
        if not changed:
            break

    # If an edge can still be relaxed, a negative cycle exists.
    if any(h[u] + weight < h[v] for u, v, weight in edges):
        raise NegativeCycleError("Negative Cycle Detected")

    return h


def dijkstra(adjacency, source):
    """
    Find shortest paths from one source using Dijkstra's algorithm.

    This implementation uses a binary min-heap to efficiently select
    the vertex with the smallest tentative distance.

    It operates on the non-negative edge weights generated by
    Johnson's reweighting step.

    Args:
        adjacency (dict[int, list[tuple[int, int]]]):
            Adjacency list mapping each vertex to its outgoing
            (neighbor, weight) pairs.

        source (int):
            Starting vertex for the shortest-path search.

    Returns:
        tuple:
            distance (dict[int, int | float]):
                Minimum distance from source to each vertex.
                Unreachable vertices retain math.inf.

            parent (dict[int, int]):
                Predecessor of each reachable non-source vertex
                along a shortest path.

    Time Complexity:
        O((V + E) log V) for the heap-based implementation.

    Space Complexity:
        O(V + E) in the worst case, including heap entries.
    """

    # Initially, every vertex is unreachable from the source.
    distance = dict.fromkeys(adjacency, math.inf)

    # The distance from a vertex to itself is always zero.
    distance[source] = 0

    # Store predecessor relationships for path reconstruction.
    parent = {}

    # Priority queue entries are (current_distance, vertex).
    queue = [(0, source)]

    while queue:
        # Select the vertex with the smallest queued distance.
        cost, u = heapq.heappop(queue)

        # Ignore outdated entries created by earlier relaxations.
        if cost != distance[u]:
            continue

        # Examine all outgoing edges from the current vertex.
        for v, weight in adjacency[u]:
            candidate = cost + weight

            # Relax the edge if a shorter route has been discovered.
            if candidate < distance[v]:
                distance[v] = candidate

                # Remember the predecessor for reconstructing paths.
                parent[v] = u

                # Add the improved distance to the priority queue.
                heapq.heappush(queue, (candidate, v))

    return distance, parent


def johnson(nodes, edges):
    """
    Compute all-pairs shortest paths using Johnson's Algorithm.

    The algorithm consists of four main stages:

    1. Compute vertex potentials with Bellman-Ford.
    2. Reweight the original edges to eliminate negative weights.
    3. Run Dijkstra's algorithm once from every source vertex.
    4. Convert the resulting distances back to original weights.

    For an original edge (u, v) with weight w, reweighting uses:

        w'(u, v) = w(u, v) + h(u) - h(v)

    The transformed weight is guaranteed to be non-negative
    when no negative-weight cycle exists.

    After Dijkstra calculates d'(u, v), restore the original
    shortest-path distance using:

        d(u, v) = d'(u, v) - h(u) + h(v)

    Args:
        nodes (iterable[int]):
            Graph vertices.

        edges (iterable[tuple[int, int, int]]):
            Directed weighted edges.

    Returns:
        tuple:
            distances (dict[int, dict[int, int | float]]):
                All-pairs shortest-path distance matrix stored
                as nested dictionaries.

                Example:
                    distances[source][goal]

            parents (dict[int, dict[int, int]]):
                Predecessor dictionaries for every source.

                Example:
                    parents[source][vertex]

    Raises:
        NegativeCycleError:
            If Bellman-Ford detects a negative-weight cycle.

    Time Complexity:
        O(VE + V(E + V) log V)

    Space Complexity:
        O(V^2 + V + E)
    """

    # Materialize vertices and edges to support repeated iterations.
    nodes, edges = tuple(nodes), list(edges)

    # STEP 1: Compute vertex potentials using Bellman-Ford.
    # This also verifies that no negative-weight cycle exists.
    h = bellman_ford(nodes, edges)

    # STEP 2: Construct a reweighted adjacency list.
    adjacency = {u: [] for u in nodes}

    for u, v, weight in edges:
        # Reweight the edge while preserving shortest-path ordering.
        new_weight = weight + h[u] - h[v]
        adjacency[u].append((v, new_weight))

    # Store shortest distances and predecessor information.
    distances, parents = {}, {}

    # STEP 3: Run Dijkstra independently from every source vertex.
    for source in nodes:
        reweighted, parents[source] = dijkstra(adjacency, source)

        # STEP 4: Restore shortest-path costs to original weights.
        distances[source] = {
            v: cost - h[source] + h[v]
            for v, cost in reweighted.items()
        }

    return distances, parents


def reconstruct_path(parents, source, goal):
    """
    Reconstruct a shortest path from source to goal.

    The function follows predecessor references generated by
    Dijkstra's algorithm, beginning at the destination and moving
    backward until it reaches the source.

    The collected route is then reversed to obtain the correct
    source-to-destination order.

    Args:
        parents (dict[int, dict[int, int]]):
            Predecessor dictionaries produced by Johnson's Algorithm.

        source (int):
            Starting vertex.

        goal (int):
            Destination vertex.

    Returns:
        list[int]:
            Ordered list of vertices along the shortest path.

            Returns an empty list if the destination is unreachable.
            Returns [source] when source and goal are identical.

    Time Complexity:
        O(V) in the worst case.

    Space Complexity:
        O(V) for the reconstructed route.
    """

    # Check whether a predecessor chain exists for the destination.
    if source != goal and goal not in parents[source]:
        return []

    # Reconstruct the path backward from the destination.
    route = [goal]

    while route[-1] != source:
        # Move to the predecessor of the current vertex.
        route.append(parents[source][route[-1]])

    # Reverse the collected vertices to obtain source -> goal.
    return route[::-1]


def choose_csv():
    """
    Allow the user to select a CSV graph dataset interactively.

    Searches for CSV files inside the 'Datasets' directory located
    alongside the Python script.

    The user may select a listed file by its number or manually
    enter a CSV file path.

    Returns:
        pathlib.Path:
            Path to the selected CSV dataset.

    Notes:
        The returned path is not checked for existence here.
        File-access errors are handled when the dataset is read.
    """

    # Locate the Datasets directory relative to this script.
    folder = Path(__file__).resolve().parent / "Datasets"

    # Retrieve available CSV files in alphabetical order.
    files = sorted(folder.glob("*.csv"))

    # Display numbered dataset options.
    for index, path in enumerate(files, 1):
        print(f"{index}. {path.name}")

    # Accept either a selection number or a file path.
    choice = input(
        "Select a file number or enter a CSV path: "
    ).strip()

    # Return the corresponding dataset when a valid number is entered.
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return files[int(choice) - 1]

    # Otherwise, interpret the input as a file path.
    return Path(choice).expanduser()


def main():
    """
    Run the Johnson's Algorithm command-line application.

    The function controls the complete execution workflow:

    1. Parse command-line arguments.
    2. Load graph data from a CSV dataset.
    3. Calculate all-pairs shortest paths.
    4. Handle negative-weight cycle detection.
    5. Display the distance matrix for small graphs.
    6. Obtain source and destination vertices.
    7. Reconstruct and display the shortest path.
    8. Validate calculated distances against CSV references.

    Command-Line Arguments:
        csv:
            Optional path to a graph dataset.

        --source:
            Optional integer source vertex.

        --goal:
            Optional integer destination vertex.

    Returns:
        int:
            0 if execution completes without verification mismatches.
            1 if the calculated results disagree with expected
            values in the CSV dataset.

    Notes:
        Negative-cycle detection returns 0 when no expected results
        are supplied, or when the CSV includes a negative-infinity
        value indicating an expected negative cycle.
    """

    # Configure the command-line argument parser.
    parser = argparse.ArgumentParser(description=__doc__)

    # Optional positional argument specifying the dataset path.
    parser.add_argument("csv", nargs="?", type=Path)

    # Optional source and destination vertex identifiers.
    parser.add_argument("--source", type=int)
    parser.add_argument("--goal", type=int)

    # Parse arguments provided by the user.
    args = parser.parse_args()

    # Load the selected dataset.
    # If no path was provided, prompt the user interactively.
    nodes, edges, expected = read_dataset(args.csv or choose_csv())

    # Run Johnson's Algorithm and handle negative-weight cycles.
    try:
        distances, parents = johnson(nodes, edges)

    except NegativeCycleError as error:
        print(error)

        # Check whether the CSV expected a negative-weight cycle.
        if expected:
            matched = -math.inf in expected.values()

            print(
                "CSV cycle check:",
                "PASS" if matched else "FAIL"
            )

            return 0 if matched else 1

        # No expected distances are available for cycle verification.
        return 0

    # Display the full distance matrix only for small graphs.
    # Larger graphs are omitted to avoid excessive console output.
    if len(nodes) <= 20:
        print("Distance matrix (row and column order):", nodes)

        for source in nodes:
            print([
                distances[source][goal]
                for goal in nodes
            ])

    # Obtain the source vertex from arguments or interactive input.
    source = (
        args.source
        if args.source is not None
        else int(input("Source node: "))
    )

    # Obtain the destination vertex.
    goal = (
        args.goal
        if args.goal is not None
        else int(input("Goal node: "))
    )

    # Ensure the selected vertices exist in the dataset.
    if source not in distances or goal not in distances:
        raise ValueError(
            "Choose source and goal IDs present in the CSV."
        )

    # Reconstruct the shortest route between the selected vertices.
    route = reconstruct_path(parents, source, goal)

    # Display the route or indicate that the goal is unreachable.
    print(
        "Path:",
        " -> ".join(map(str, route)) if route else "unreachable"
    )

    # Display the original shortest-path distance.
    print("Cost:", distances[source][goal])

    # Compare calculated distances against expected CSV values.
    mismatches = 0

    for (u, v), reference in expected.items():

        # Count missing or incorrect results.
        if (
            u not in distances
            or v not in distances[u]
            or distances[u][v] != reference
        ):
            mismatches += 1

            # Limit detailed mismatch reporting to the first five.
            if mismatches <= 5:
                print(
                    f"Mismatch {u} -> {v}: "
                    f"expected {reference}, "
                    f"got {distances.get(u, {}).get(v)}"
                )

    # Display the overall CSV verification summary.
    if expected:
        print(
            f"CSV verification: "
            f"{len(expected) - mismatches}/{len(expected)} "
            f"distances matched"
        )
    else:
        print("No expected distances supplied.")

    # Return failure status if any reference distances differ.
    return int(bool(mismatches))


# Execute the application only when this file is run directly.
# Importing this module does not automatically run the program.
if __name__ == "__main__":
    try:
        # Convert the main function's return value to an exit status.
        raise SystemExit(main())

    except (ValueError, OSError, KeyError) as error:
        # Present common input and file errors without a traceback.
        raise SystemExit(str(error))
