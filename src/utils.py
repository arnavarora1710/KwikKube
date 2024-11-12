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
    return get_moves()
