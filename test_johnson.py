
from johnson import johnson_algorithm

graph = {
    0: [(1, -2), (3, 3)],
    1: [(2, -1)],
    2: [(0, 4), (3, 1)],
    3: []
}

result = johnson_algorithm(graph)

for row in result:
    print(row)
