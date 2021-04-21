import numpy as np


def __merge_tiles_v1(ref_tile, alt_tiles, noise_variance, scale_factor):
    assert ref_tile.dtype == np.float32, \
        "Reference frame type should have np.float32 dtype"

    wiener_factors = []
    merged_tile = np.zeros_like(ref_tile)

    for alt_tile in alt_tiles:
        mse = np.square(ref_tile - alt_tile).mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
        noise_variance = noise_variance.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)

        assert mse.shape == (1, 1, 4), "Mae shape is incorrect"
        assert noise_variance.shape == (1, 1, 4), "Noise var shape is incorrect"

        wiener_factor = mse / (mse + scale_factor * noise_variance)
        
        merged_tile = merged_tile + \
            alt_tile + wiener_factor * (ref_tile - alt_tile)

    return merged_tile


def merge(
    ref_frame, 
    alt_frames, 
    alignments, 
    tile_size, 
    noise_variance, 
    scale_factor=8,):

    merged_image = np.zeros_like(ref_frame)

    _, height_steps, width_steps, _ = alignments.shape

    for h_step in range(height_steps):
        for w_step in range(width_steps):
            h_start = h_step * tile_size
            h_end = h_start + tile_size

            w_start = w_step * tile_size
            w_end = w_start + tile_size

            ref_slice = ref_frame[h_start:h_end, w_start:w_end,:]
            ref_slice = ref_slice.astype(np.float32)
            noise_slice = noise_variance[h_start:h_end, w_start:w_end,:]
            noise_slice = noise_slice.astype(np.float32)

            alt_slices = []
            for i, alt_frame in enumerate(alt_frames):
                h_shift, w_shift = alignments[i, h_step, w_step]

                h_alt_start = h_start + h_shift
                h_alt_end = h_end + h_shift

                w_alt_start = w_start + w_shift
                w_alt_end = w_end + w_shift

                alt_slice = alt_frame[h_alt_start:h_alt_end, w_alt_start:w_alt_end]
                alt_slice = alt_slice.astype(np.float32)
                alt_slices.append(alt_slice)

            merged_tile = __merge_tiles_v1(
                ref_slice, alt_slices, noise_slice, scale_factor)

            merged_image[h_start:h_end, w_start:w_end] = merged_tile

    return merged_image

    
    


