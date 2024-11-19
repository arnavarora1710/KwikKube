import sys, os
sys.path.append(os.path.abspath("../lib"))
from cfop import *

# WARNING: DON'T USE 'a' AS A VARIABLE (USED BY CUBE LIB FOR THE CUBE 😭)

color_map = {
    'W': 0b000,  # White
    'G': 0b001,  # Green
    'R': 0b010,  # Red
    'O': 0b011,  # Orange
    'Y': 0b100,  # Yellow
    'B': 0b101   # Blue
}

code_map = {
    0b000 : 'W',
    0b001 : 'G',
    0b010 : 'R',
    0b011 : 'O',
    0b100 : 'Y',
    0b101 : 'B'
}

move_map = {
    'F': 0, 'F\'': 1, 'F2': 2,
    'R': 3, 'R\'': 4, 'R2': 5,
    'U': 6, 'U\'': 7, 'U2': 8,
    'L': 9, 'L\'': 10, 'L2': 11,
    'B': 12, 'B\'': 13, 'B2': 14,
    'D': 15, 'D\'': 16, 'D2': 17,
    "BOT": 18
}

reverse_move_map = {
    0: 'F', 1: 'F\'', 2: 'F2',
    3: 'R', 4: 'R\'', 5: 'R2',
    6: 'U', 7: 'U\'', 8: 'U2',
    9: 'L', 10: 'L\'', 11: 'L2',
    12: 'B', 13: 'B\'', 14: 'B2',
    15: 'D', 16: 'D\'', 17: 'D2',
    18: "BOT"
}

def getRandomScramble(moves):
    scramble(moves)

def encodeCube(cube):
    global color_map
    encoded_cube = []
    for face in cube:
        encoded_face = []
        for row in face:
            encoded_row = [color_map[color] for color in row]
            encoded_face.append(encoded_row)
        encoded_cube.append(encoded_face)
    return encoded_cube

def decodeCube(cube):
    global code_map
    decoded_cube = []
    for face in cube:
        decoded_face = []
        for row in face:
            decoded_row = [code_map[color] for color in row]
            decoded_face.append(decoded_row)
        decoded_cube.append(decoded_face)
    return decoded_cube

def encodeMoves(moves):
    global move_map
    move_list = moves.split()
    encoded_moves = [move_map[move] for move in move_list]
    return encoded_moves

def decodeMoves(encoded_moves):
    global reverse_move_map
    decoded_moves = [reverse_move_map[code] for code in encoded_moves]
    while "BOT" in decoded_moves:
        decoded_moves.remove("BOT")
    return " ".join(decoded_moves)

def getToDesired(config="random"):
    make_cube()
    getRandomScramble(30)
    if config == "cross":
        cross()
    elif config == "f2l":
        cross()
        f2l()
    elif config == "solved":
        solve()
    else:
        pass
    return encodeCube(get_cube())

def solveNow(cube):
    a = decodeCube(cube)
    solve()
    return encodeMoves(get_moves())

def doMoves(cube, moves):
    a = decodeCube(cube)
    m(decodeMoves(moves))
    return encodeCube(get_cube())
