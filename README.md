## Johnson's Algorithm – All-Pairs Shortest Paths

## Project Overview

This project implements **Johnson's Algorithm** to find the shortest paths between all pairs of vertices in a directed, weighted graph. It supports negative edge weights and detects negative-weight cycles using Bellman-Ford.

The project also includes a CSV graph generator, Floyd–Warshall verification, shortest-path visualization, and performance benchmarking.

## Features

- Computes an N × N shortest-path distance matrix.
- Handles positive and negative edge weights.
- Detects negative-weight cycles.
- Generates CSV graph datasets with varying sizes and densities.
- Uses Floyd–Warshall to generate expected results for correctness verification.
- Visualizes shortest paths from a selected source node.
- Measures execution time across different graph densities.

## Implementation

Johnson's Algorithm combines three main steps:

1. **Bellman-Ford:** Detects negative cycles and calculates vertex potentials.
2. **Dijkstra:** Computes shortest paths from every vertex using re-weighted edges.
3. **Distance Correction:** Converts the calculated distances back to their original values.

The CSV generator creates graph datasets containing nodes, edges, weights, and X-Y coordinates. The same generator is shared with the A* project, although Johnson's Algorithm does not use coordinates.

## Testing and Benchmarking

The algorithm is verified by comparing its output against the expected distance matrices calculated using Floyd–Warshall.

Testing covers negative edge weights, negative cycles, and unreachable vertices.

**Sample benchmark results (80 vertices):**

| Graph Density | Median Execution Time |
|---|---|
| 5% | 6.1 ms |
| 10% | 10.3 ms |

The results show that execution time increases with graph density for the tested configurations.

## Technologies

- Python
- CSV
- Jupyter Notebook
- Johnson's Algorithm
- Floyd–Warshall Algorithm

## How to Run

1. Run the CSV graph generator to create test datasets.
2. Run the Johnson's Algorithm source code using the generated datasets.
3. View the shortest-path distance matrix.
4. Compare the output against the Floyd–Warshall reference matrix.
5. Run the visualization or benchmarking code to view the results.

## Project Contributions

The CSV dataset generator was developed by me and shared with my teammate for A* testing. The Johnson's Algorithm implementation, shortest-path visualization, correctness verification, and benchmarking were completed independently.
