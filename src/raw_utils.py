import numpy as np


def rggb_to_planes(image):    
    height, width = image.shape

    pl_height = height // 2
    pl_width = width // 2
    
    hw = pl_height * width
    
    rg_rows = np.arange(0, height, 2)
    gb_rows = rg_rows + 1
    
    rg_image = image[rg_rows]
    gb_image = image[gb_rows]
    
    rg_ravel = rg_image.ravel()
    gb_ravel = gb_image.ravel()
    
    indices = np.arange(0, hw, 2)
    
    red = rg_ravel[indices].reshape(pl_height, pl_width)
    green_red = rg_ravel[indices + 1].reshape(pl_height, pl_width)
    green_blue = gb_ravel[indices].reshape(pl_height, pl_width)
    blue = gb_ravel[indices + 1].reshape(pl_height, pl_width)
    
    result = np.zeros((pl_height, pl_width, 4), dtype=np.uint16)
    result[:,:,0] = red
    result[:,:,1] = green_red
    result[:,:,2] = green_blue
    result[:,:,3] = blue
    
    return result


def planes_to_rggb(image):
    """
    image: ndarray of shape [H, W, 4]
    """
    
    pl_height, pl_width, _ = image.shape
    height, width = pl_height * 2, pl_width * 2
    
    red = image[:,:,0].ravel()
    green_r = image[:,:,1].ravel()
    green_b = image[:,:,2].ravel()
    blue = image[:,:,3].ravel()
    
    half_hw = pl_height * width
    
    red_green = np.zeros(half_hw)
    green_blue = np.zeros(half_hw)
    
    indices = np.arange(0, half_hw, 2)
    red_green[indices] = red
    red_green[indices + 1] = green_r
    green_blue[indices] = green_b
    green_blue[indices + 1] = blue
    
    red_green = red_green.reshape(pl_height, width)
    green_blue = green_blue.reshape(pl_height, width)
    
    rggb_image = np.zeros((height, width), dtype=np.uint16)
    
    indices = np.arange(0, height, 2)
    rggb_image[indices] = red_green
    rggb_image[indices + 1] = green_blue
    
    return rggb_image