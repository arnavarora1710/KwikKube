from matrix_repr import *
from collections import deque
import numpy as np

def solve(cube_obj, algorithm, move_matr_cache):
    if cube_obj.isSolved():
        return []

    visited= {tuple(map(tuple, cube_obj.cube)): True}
    q = deque([(algorithm, [])])
    while q:
        alg, moves = q.popleft()

        alg_tuple = tuple(map(tuple, alg))
        if alg_tuple in move_inv_mapping_cache:
            moves.append(move_inv_mapping_cache[alg_tuple])
            return moves

        for move in ["U", "R", "F", "D", "L", "Ui", "Ri", "Fi", "Di", "Li"]:
            move_mat = move_matr_cache[move]
            new_alg = np.linalg.inv(move_mat).astype(bool) @ alg
            if tuple(map(tuple, new_alg)) not in visited:
                visited[tuple(map(tuple, new_alg))] = True
                q.append((new_alg, moves + [move]))

    return None

cube_obj = Cube()
cube_obj.move("URFDFR")
algorithm = np.linalg.inv(cube_obj.cube).astype(bool)

move_matr_cache = {}
for move in ["U", "R", "F", "D", "L", "Ui", "Ri", "Fi", "Di", "Li"]:
    move_matr_cache[move] = moveMatrix(move)

move_inv_mapping_cache = {}
for move in ["U", "R", "F", "D", "L", "Ui", "Ri", "Fi", "Di", "Li"]:
    move_tuple = tuple(map(tuple, moveMatrix(move)))
    move_inv_mapping_cache[move_tuple] = move

solution = solve(cube_obj, algorithm, move_matr_cache)
print(solution)
