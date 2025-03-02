import numpy as np

class Cube:
    def __init__(self):
        self.cube = np.eye(24, dtype=bool)
        self.ref = np.eye(24, dtype=bool)
        # self.mapping = {}
        # # mapping[i] stores j such that cube[j][i] = 1 for cubelet i
        # # meaning that (original) cubelet i is at cubelet j
        # for i in range(24):
        #     self.mapping[i] = i

    def isValidMove(self):
        # determinant of cube must not be 0
        return np.linalg.det(self.cube) != 0

    def isValid(self):
        pass

    def isSolved(self):
        return self.cube == self.ref

    def move(self, move):
        # figure out what sticker moved where
        # example: if U is the move, then
        # then the stickers on the top face move counterclockwise
        # so look at the first 8 columns of the matrix
        # find the 1s in these columns
        # cycle them clockwise
        if move == "U":
            self.cube[:, :8] = np.roll(self.cube[:, :8], -2, axis=1)
        elif move == "Ui":
            self.cube[:, :8] = np.roll(self.cube[:, :8], 2, axis=1)
        elif move == "R":
            # want to roll (2, 3, 4, 12, 20, 19, 18, 10)
            self.cube[:, [2, 3, 4, 12, 20, 19, 18, 10]] = np.roll(self.cube[:, [2, 3, 4, 12, 20, 19, 18, 10]], 2, axis=1)
        elif move == "Ri":
            self.cube[:, [2, 3, 4, 12, 20, 19, 18, 10]] = np.roll(self.cube[:, [2, 3, 4, 12, 20, 19, 18, 10]], -2, axis=1)
        elif move == "F":
            # want to roll (0, 1, 2, 10, 18, 17, 16, 8)
            self.cube[:, [0, 1, 2, 10, 18, 17, 16, 8]] = np.roll(self.cube[:, [0, 1, 2, 10, 18, 17, 16, 8]], 2, axis=1)
        elif move == "Fi":
            self.cube[:, [0, 1, 2, 10, 18, 17, 16, 8]] = np.roll(self.cube[:, [0, 1, 2, 10, 18, 17, 16, 8]], -2, axis=1)
        elif move == "D":
            self.cube[:, 16:] = np.roll(self.cube[:, 16:], 2, axis=1)
        elif move == "Di":
            self.cube[:, 16:] = np.roll(self.cube[:, 16:], -2, axis=1)
        elif move == "L":
            # want to roll (0, 7, 6, 14, 22, 21, 20, 8)
            self.cube[:, [0, 7, 6, 14, 22, 21, 20, 8]] = np.roll(self.cube[:, [0, 7, 6, 14, 22, 21, 20, 8]], -2, axis=1)
        elif move == "Li":
            self.cube[:, [0, 7, 6, 14, 22, 21, 20, 8]] = np.roll(self.cube[:, [0, 7, 6, 14, 22, 21, 20, 8]], 2, axis=1)
        elif move == "B":
            # want to roll (4, 5, 6, 14, 22, 21, 20, 12)
            self.cube[:, [4, 5, 6, 14, 22, 21, 20, 12]] = np.roll(self.cube[:, [4, 5, 6, 14, 22, 21, 20, 12]], 2, axis=1)
        elif move == "Bi":
            self.cube[:, [4, 5, 6, 14, 22, 21, 20, 12]] = np.roll(self.cube[:, [4, 5, 6, 14, 22, 21, 20, 12]], -2, axis=1)
        else:
            raise ValueError(f"Invalid move: {move}")

    def move(self, moves):
        for move in moves:
            self.move(move)
