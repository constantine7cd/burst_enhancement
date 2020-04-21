import numpy as np


Fa_11 = np.array(
    [[1, -2, 1], 
     [2, -4, 2],
     [1, -2, 1]], dtype=np.float32
).reshape(-1) / 4


Fa_22 = np.array(
    [[ 1,  2,  1], 
     [-2, -4, -2],
     [ 1,  2,  1]], dtype=np.float32
).reshape(-1) / 4


Fa_12 = np.array(
    [[ 1, 0, -1], 
     [ 0, 0,  0],
     [-1, 0,  1]], dtype=np.float32
).reshape(-1) / 4


Fb_1 = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float32 
).reshape(-1) / 8


Fb_2 = np.array(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]], dtype=np.float32
).reshape(-1) / 8