import cv2
import numpy as np

from numpy.fft import fft2
from numpy.fft import ifft2

from .subpixel_translation import subpix_translation


def __l2_norm_squared(matrix):
    return np.sum(np.square(matrix))


def __l1_norm(matrix):
    return np.sum(np.abs(matrix))


def l1_displacement(ref_tile, alternate_tile):
    return __compute_displacement(ref_tile, alternate_tile, __l1_norm)


def l2_displacement(ref_tile, alternate_tile):
    return __compute_displacement(ref_tile, alternate_tile, __l2_norm_squared)

    
def __compute_displacement(ref_tile, alternate_tile, norm_callback):
    ref_h, ref_w = ref_tile.shape
    alt_h, alt_w = alternate_tile.shape

    disp_h = alt_h - ref_h + 1
    disp_w = alt_w - ref_w + 1

    displacement = np.zeros((disp_h, disp_w))

    #TODO: can we perform this calculations in parallel?
    for h in range(disp_h):
        for w in range(disp_w):
            diff = ref_tile - alternate_tile[h:h + ref_h, w:w + ref_w]
            disp = norm_callback(diff)
            displacement[h, w] = disp

    return displacement


def l2_displacement_fast(ref_tile, alternate_tile):
    pass


def __alternate_image_slice(start, end, margin, right_limit, left_limit=0):
    start = start - margin
    end = end + margin

    if start < left_limit:
        end -= start
        start = 0
    elif end > right_limit:
        start += (right_limit - end)
        end = right_limit

    return start, end


def alignment(ref_frame, alternate_frame, tile_size, search_level, disp_callback):
    height_ref, width_ref = ref_frame.shape

    width_steps = width_ref // tile_size
    height_steps = height_ref // tile_size

    margin = search_level // 2

    alignments = np.zeros((height_steps, width_steps, 2))

    height_slice = lambda start, end: __alternate_image_slice(start, end, margin, height_ref)
    width_slice = lambda start, end: __alternate_image_slice(start, end, margin, width_ref)

    print("Horizontal steps: {}, vertical steps: {}".format(height_steps, width_steps))

    # Could it be parallelized?
    for ws in range(width_steps):
        for hs in range(height_steps):
            start_h = hs * tile_size
            end_h = start_h + tile_size

            start_w = ws * tile_size
            end_w = start_w + tile_size

            ref_slice = ref_frame[start_h : end_h, start_w : end_w]

            start_h_alt, end_h_alt = height_slice(start_h, end_h)
            start_w_alt, end_w_alt = width_slice(start_w, end_w)
            
            alt_slice = alternate_frame[start_h_alt : end_h_alt, start_w_alt : end_w_alt]

            displacement = disp_callback(ref_slice, alt_slice)

            alignment = subpix_translation(displacement)
            alignments[hs, ws] = alignment

    return alignments


def l2_alignment(ref_frame, alternate_frame, tile_size, search_level):
    return alignment(ref_frame, alternate_frame, tile_size, search_level, l2_displacement)


def l1_alignment(ref_frame, alternate_frame, tile_size, search_level):
    return alignment(ref_frame, alternate_frame, tile_size, search_level, l1_displacement)

