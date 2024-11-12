from utils import *
import numpy as np

def makeData(num_samples = 1, config = "random", train = True):
    data = []
    for _ in range(num_samples):
        cube = getToDesired(config = config)
        cube = np.array(cube)
        data.append(cube)
    data = np.array(data)
    dir = "train" if train else "test"
    np.save(f'../data/{dir}/cube.npy', data)

def loadData(train = True):
    dir = "train" if train else "test"
    return np.load(f'../data/{dir}/cube.npy')

