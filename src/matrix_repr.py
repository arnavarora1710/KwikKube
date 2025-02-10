import numpy as np

class Cube:
    def __init__(self):
        self.cube = np.eye(24, dtype=bool)

    def move(self, move):
        if move == "U":
            pass
        elif move == "Ui":
            pass
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
