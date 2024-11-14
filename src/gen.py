from utils import *
import numpy as np

def makeData(num_samples = 1, config = "random", train = True):
    data = []
    move_data = []
    for _ in range(num_samples):
        cube = getToDesired(config = config)
        tmp_cube = cube.copy()
        solve_moves = solveNow(tmp_cube)
        data.append(cube)
        move_data.append(solve_moves)
    dir = "train" if train else "test"
    np.savez(f'../data/{dir}/cube', *data)
    np.savez(f'../data/{dir}/cube_moves', *move_data)

def loadData(train = True):
    dir = "train" if train else "test"
    ret1 = []
    ret2 = []
    cube = np.load(f"../data/{dir}/cube.npz")
    cube_moves = np.load(f"../data/{dir}/cube_moves.npz")
    for k, v in cube.items():
        ret1.append(v.tolist())
    for k, v in cube_moves.items():
        ret2.append(v.tolist())
    return (ret1, ret2)
