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


def __alternate_image_slice_coarser(start, end, margin, right_limit, left_limit=0):
    start = start - margin
    end = end + margin

    clip = None

    if start < left_limit:
        clip = left_limit - start

    start = max(left_limit, start)
    end = min(right_limit, end)

    return start, end, clip


def __alternate_image_slice(start, end, coarser_alignment, margin, right_limit, left_limit=0):
    if coarser_alignment is not None:
        start += coarser_alignment
        end += coarser_alignment

    return __alternate_image_slice_coarser(start, end, margin, right_limit, left_limit)


def alignment(
    ref_frame, 
    alternate_frame, 
    tile_size, 
    search_level, 
    coarser_level_alignment, 
    disp_callback):

    height_ref, width_ref = ref_frame.shape

    width_steps = width_ref // tile_size
    height_steps = height_ref // tile_size

    margin = search_level

    print("Search level: {}".format(search_level))

    alignments = np.zeros((height_steps, width_steps, 2), dtype=np.int16)

    height_slice = lambda start, end, coarser_alignment: __alternate_image_slice(
        start, end, coarser_alignment, margin, height_ref)
    width_slice = lambda start, end, coarser_alignment: __alternate_image_slice(
        start, end, coarser_alignment, margin, width_ref)

    print("Alignment Horizontal steps: {}, vertical steps: {}".format(width_steps, height_steps))

    width_shift, height_shift = None, None

    # Could it be parallelized?
    for ws in range(width_steps):
        for hs in range(height_steps):
            start_h = hs * tile_size
            end_h = start_h + tile_size

            start_w = ws * tile_size
            end_w = start_w + tile_size

            ref_slice = ref_frame[start_h : end_h, start_w : end_w]

            if coarser_level_alignment is not None:
                height_shift, width_shift = coarser_level_alignment[hs, ws]

            start_h_alt, end_h_alt, clip_h = height_slice(start_h, end_h, height_shift)
            start_w_alt, end_w_alt, clip_w = width_slice(start_w, end_w, width_shift)
            
            alt_slice = alternate_frame[start_h_alt : end_h_alt, start_w_alt : end_w_alt]

            displacement = disp_callback(ref_slice, alt_slice)

            alignment_h, alignment_w = subpix_translation(displacement)

            # the previous calculations of displacement and subpixel translation
            # were based on absolute coordinates, where (0,0) is upper left corner
            # of alternate image. Final alignments should be centered with respect
            # to margin and clip values.
            assert alignment_h >= 0, "Alignment after subpix translation should be >=0"
            assert alignment_w >= 0, "Alignment after subpix translation should be >=0"

            alignment_h -= margin
            alignment_w -= margin

            if height_shift is not None:
                alignment_h += height_shift
                alignment_w += width_shift
    
            if clip_h is not None:
                alignment_h += clip_h
            if clip_w is not None:
                alignment_w += clip_w

            alignments[hs, ws] = [alignment_h, alignment_w]

    return alignments


def l2_alignment(ref_frame, alternate_frame, tile_size, search_level, \
    coarser_alignment):

    return alignment(ref_frame, alternate_frame, tile_size, search_level, \
        coarser_alignment, l2_displacement)


def l1_alignment(ref_frame, alternate_frame, tile_size, search_level, \
    coarser_alignment):

    return alignment(ref_frame, alternate_frame, tile_size, search_level, \
        coarser_alignment, l1_displacement)


def get_coarser_neighbor_alignments(
    coarser_alignments, height_step, width_step, downsample_factor, tile_ratio):

    #print("coarser alignments dtype: {}".format(coarser_alignments.dtype))

    alignment_height, alignment_width, _ = coarser_alignments.shape

    tile_scale = downsample_factor // tile_ratio

    coarser_hs = np.floor(height_step / tile_scale).astype(np.int16)
    coarser_ws = np.floor(width_step / tile_scale).astype(np.int16)

    pos_h = height_step % tile_scale
    pos_w = width_step % tile_scale

    half_factor = tile_scale // 2

    coarser_hs_2 = coarser_hs + 1 if pos_h >= half_factor else coarser_hs - 1
    coarser_ws_2 = coarser_ws + 1 if pos_w >= half_factor else coarser_ws - 1

    coarser_hs_2 = min(max(0, coarser_hs_2), alignment_height - 1)
    coarser_ws_2 = min(max(0, coarser_ws_2), alignment_width - 1)

    coarser_indices = set()

    for coarser_h in [coarser_hs, coarser_hs_2]:
        for coarser_w in [coarser_ws, coarser_ws_2]:
            coarser_indices.add((coarser_h, coarser_w))

    init_alignments = []

    for indices in coarser_indices:
        hs, ws = indices
        alignment = coarser_alignments[hs, ws] * downsample_factor

        init_alignments.append(alignment)

    return init_alignments


def __l1_residual(first, second):
    return np.sum(np.abs(first - second))


def compute_coarser_init_alignment(
    ref_frame, 
    tile_size, 
    alternate_frame,
    coarser_alignments, 
    coarser_tile_size, 
    downsample_factor):

    height_curr, width_curr = ref_frame.shape

    height_steps = height_curr // tile_size
    width_steps = width_curr // tile_size

    init_alignments = np.zeros((height_steps, width_steps, 2), dtype=np.int16)

    tile_ratio = tile_size // coarser_tile_size

    print("Total tiles: {}".format(height_steps * width_steps))
    print("Init Horizontal steps: {}, vertical steps: {}".format(width_steps, height_steps))

    for hs in range(height_steps):
        for ws in range(width_steps):
            start_h = hs * tile_size
            end_h = start_h + tile_size

            start_w = ws * tile_size
            end_w = start_w + tile_size

            ref_slice = ref_frame[start_h : end_h, start_w : end_w]

            alignments = get_coarser_neighbor_alignments(
                coarser_alignments, hs, ws, downsample_factor, tile_ratio)

            l1_residuals = []

            for alignment in alignments:
                # computing l1 distance between ref and alternate frames 
                # according to initial alignment
                init_h, init_w = alignment

                start_h_alt = start_h + init_h
                end_h_alt = start_h_alt + tile_size

                start_w_alt = start_w + init_w
                end_w_alt = start_w_alt + tile_size

                alternate_slice = alternate_frame[start_h_alt:end_h_alt, start_w_alt:end_w_alt]

                l1_residual = __l1_residual(ref_slice, alternate_slice)
                l1_residuals.append(l1_residual)

            # Adding to init alignments the alignment with min l1 residual
            l1_residuals = np.array(l1_residuals)
            argmin = np.argmin(l1_residuals)
            min_alignment = alignments[argmin]

            init_alignments[hs, ws] = min_alignment

    print("Init alignment shape: {}".format(init_alignments.shape))

    return init_alignments


def average_bayer_raw(frame):
    """
    Averages each 4 pixels of Bayer raw image into one
    """

    height, width = frame.shape

    out_height = height // 2
    out_width = width // 2
    
    out = np.reshape(frame, (out_height, 2, out_width, 2)).astype(np.float32)
    out = np.sum(np.sum(out, axis=-1), -2) / 4

    return out


def __downsample(image, ds_factor):
    for _ in range(ds_factor // 2):
        image = cv2.pyrDown(image)

    return image

    
def hierarchical_image_alignment(ref_frame, alt_frame):
    ref_frame = average_bayer_raw(ref_frame)
    alt_frame = average_bayer_raw(alt_frame)

    assert ref_frame.shape == alt_frame.shape

    print("After averaging shape: {}".format(ref_frame.shape))

    ref_pyramid = [ref_frame]
    alt_pyramid = [alt_frame]

    for idx, factor in enumerate([2, 4, 4]):
        ref_upper_lvl = __downsample(ref_pyramid[-1], factor)
        alt_upper_lvl = __downsample(alt_pyramid[-1], factor)

        print("Level: {}, shape: {}".format(idx + 1, ref_upper_lvl.shape))

        ref_pyramid.append(ref_upper_lvl)
        alt_pyramid.append(alt_upper_lvl)

    tiles = [16, 16, 16, 8]
    search_lvls = [1, 4, 4, 4]
    alignment_funcs = [l1_alignment, l2_alignment, l2_alignment, l2_alignment]
    coarser_alignment = None

    ds_factors = [2, 4, 4]

    for i in range(3, 0, -1):
        alignments = alignment_funcs[i](
            ref_pyramid[i], alt_pyramid[i], tiles[i], search_lvls[i], coarser_alignment)

        coarser_alignment = compute_coarser_init_alignment(
            ref_pyramid[i-1], tiles[i-1], alt_pyramid[i-1], alignments, tiles[i], ds_factors[i-1])

    final_alignment = alignment_funcs[0](
        ref_pyramid[0], alt_pyramid[0], tiles[0], search_lvls[0], coarser_alignment)

    return final_alignment



    
            




    

