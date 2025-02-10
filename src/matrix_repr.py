import numpy as np

class Cube:
    def __init__(self):
        self.cube = np.eye(24, dtype=bool)
        self.ref = np.eye(24, dtype=bool)

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
            self.cube[:, :8] = np.roll(self.cube[:, :8], -1, axis=1)
        elif move == "Ui":
            self.cube[:, :8] = np.roll(self.cube[:, :8], 1, axis=1)
        elif move == "R":
            pass
        elif move == "Ri":
            pass
        elif move == "F":
            pass
        elif move == "Fi":
            pass
        elif move == "D":
            pass
        elif move == "Di":
            pass
        elif move == "L":
            pass
        elif move == "Li":
            pass
        elif move == "B":
            pass
        elif move == "Bi":
            pass
        else:
            raise ValueError(f"Invalid move: {move}")

    def move(self, moves):
        for move in moves:
            self.move(move)
