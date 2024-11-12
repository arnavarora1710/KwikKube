from utils import *
import numpy as np

def makeData(num_samples = 1, config = "random", train = True):
    data = []
    move_data = []
    for _ in range(num_samples):
        cube = getToDesired(config = config)
        tmp_cube = cube.copy()
        cube = np.array(cube)
        solve_moves = solveNow(tmp_cube)
        data.append(cube)
        move_data.append(solve_moves)
    data = np.array(data)
    move_data = np.array(move_data)
    dir = "train" if train else "test"
    np.save(f'../data/{dir}/cube.npy', data)
    np.save(f'../data/{dir}/cube_moves.npy', move_data)

def loadData(train = True):
    dir = "train" if train else "test"
    return (np.load(f'../data/{dir}/cube.npy'), np.load(f'../data/{dir}/cube_moves.npy'))
