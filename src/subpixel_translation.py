import numpy as np

from .constants import Fa_11
from .constants import Fa_22
from .constants import Fa_12
from .constants import Fb_1
from .constants import Fb_2


def subpix_translation(displacement):
    d_h, d_w = displacement.shape

    argmin = np.argmin(displacement)

    min_row = (argmin // d_w).astype(np.int16)
    min_column = (argmin % d_w).astype(np.int16)

    cond = np.array(
        [min_row == 0, 
         min_column == 0, 
         min_row == d_h - 1, 
         min_column == d_w - 1])

    if cond.any():
        return min_row, min_column

    disp_sub = displacement[min_row - 1: min_row + 2, min_column - 1: min_column + 2]
    disp_sub = np.reshape(disp_sub, -1)

    A_11 = np.dot(Fa_11, disp_sub)
    A_22 = np.dot(Fa_22, disp_sub)
    A_12 = np.dot(Fa_12, disp_sub)

    b_1 = np.dot(Fb_1, disp_sub)
    b_2 = np.dot(Fb_2, disp_sub)

    A_11 = max(0, A_11)
    A_22 = max(0, A_22)

    detA = A_11 * A_22 - np.square(A_12)

    if abs(detA) < 1e-6:
        return min_row, min_column

    ds_w = -(A_22 * b_1 - A_12 * b_2) / detA
    ds_w = np.round(ds_w).astype(np.int16)

    if abs(ds_w) > 1:
        return min_row, min_column

    ds_h = -(A_11 * b_2 - A_12 * b_1) / detA
    ds_h = np.round(ds_h).astype(np.int16)

    if abs(ds_h) > 1:
        return min_row, min_column

    ds_h += min_row
    ds_w += min_column

    return ds_h, ds_w