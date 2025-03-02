import numpy as np

def invMove(move):
    if move[-1] == "i":
        return move[0]
    else:
        return move + "i"

def moveOne(matr, move):
    # figure out what sticker moved where
    # example: if U is the move, then
    # then the stickers on the top face move counterclockwise
    # so look at the first 8 columns of the matrix
    # find the 1s in these columns
    # cycle them clockwise
    cube = matr.copy()
    if move == "U":
        cube[:, :8] = np.roll(cube[:, :8], -2, axis=1)
    elif move == "Ui":
        cube[:, :8] = np.roll(cube[:, :8], 2, axis=1)
    elif move == "R":
        # want to roll (2, 3, 4, 12, 20, 19, 18, 10)
        cube[:, [2, 3, 4, 12, 20, 19, 18, 10]] = np.roll(cube[:, [2, 3, 4, 12, 20, 19, 18, 10]], 2, axis=1)
    elif move == "Ri":
        cube[:, [2, 3, 4, 12, 20, 19, 18, 10]] = np.roll(cube[:, [2, 3, 4, 12, 20, 19, 18, 10]], -2, axis=1)
    elif move == "F":
        # want to roll (0, 1, 2, 10, 18, 17, 16, 8)
        cube[:, [0, 1, 2, 10, 18, 17, 16, 8]] = np.roll(cube[:, [0, 1, 2, 10, 18, 17, 16, 8]], 2, axis=1)
    elif move == "Fi":
        cube[:, [0, 1, 2, 10, 18, 17, 16, 8]] = np.roll(cube[:, [0, 1, 2, 10, 18, 17, 16, 8]], -2, axis=1)
    elif move == "D":
        cube[:, 16:] = np.roll(cube[:, 16:], 2, axis=1)
    elif move == "Di":
        cube[:, 16:] = np.roll(cube[:, 16:], -2, axis=1)
    elif move == "L":
        # want to roll (0, 7, 6, 14, 22, 21, 20, 8)
        cube[:, [0, 7, 6, 14, 22, 21, 20, 8]] = np.roll(cube[:, [0, 7, 6, 14, 22, 21, 20, 8]], -2, axis=1)
    elif move == "Li":
        cube[:, [0, 7, 6, 14, 22, 21, 20, 8]] = np.roll(cube[:, [0, 7, 6, 14, 22, 21, 20, 8]], 2, axis=1)
    elif move == "B":
        # want to roll (4, 5, 6, 14, 22, 21, 20, 12)
        cube[:, [4, 5, 6, 14, 22, 21, 20, 12]] = np.roll(cube[:, [4, 5, 6, 14, 22, 21, 20, 12]], 2, axis=1)
    elif move == "Bi":
        cube[:, [4, 5, 6, 14, 22, 21, 20, 12]] = np.roll(cube[:, [4, 5, 6, 14, 22, 21, 20, 12]], -2, axis=1)
    else:
        raise ValueError(f"Invalid move: {move}")
    return cube

def moveMultiple(matr, moves):
    cube = matr.copy()
    i = 0
    while i < len(moves):
        if i+1 < len(moves) and moves[i+1] == "i":
            cube = moveOne(cube, moves[i] + "i")
            i += 1
        else:
            cube = moveOne(cube, moves[i])
        i += 1
    return cube

def moveMatrix(move):
    return moveOne(np.eye(24, dtype=bool), move)

class Cube:
    def __init__(self):
        self.cube = np.eye(24, dtype=bool)
        self.ref = np.eye(24, dtype=bool)
        self.cache = {}

    def __str__(self):
        return str(self.cube.astype(int))

    def isSolved(self):
        return np.array_equal(self.cube, self.ref)

    def move(self, moves):
        i = 0
        while i < len(moves):
            if i+1 < len(moves) and moves[i+1] == "i":
                self.cube = moveOne(self.cube, moves[i] + "i")
                i += 1
            else:
                self.cube = moveOne(self.cube, moves[i])
            i += 1
