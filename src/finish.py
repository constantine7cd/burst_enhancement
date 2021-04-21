import cv2
import rawpy
import numpy as np

from math import pi
from math import sin

from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007


# TODO: Fix white balance, demosaic, srgb, add crhoma denoise
# def black_white_level(input_, black_point, white_point):
#     white_factor = 65535. / (white_point - black_point)
#     res = (input_.astype(np.int32) - black_point) * white_factor
#     res = res.astype(np.uint16)

#     return res


# def white_balance(input_, white_balance):
#     r, g0, g1, b = white_balance

#     output = input_.copy()
#     output[:,:,0] = output[:,:,0] * r
#     output[:,:,1] = output[:,:,1] * g0
#     output[:,:,2] = output[:,:,2] * g1
#     output[:,:,3] = output[:,:,3] * b

#     return output


# def demosaic(input_, pattern):
#     return demosaicing_CFA_Bayer_Menon2007(input_, pattern)


# def bilateral_filter(input_, width, height):
#     pass


# def chroma_denoise(input_):
#     pass


# def srgb(input_, srgb_mtx):
#     red_column = srgb_mtx[:,0][None, None,:]
#     green_column = srgb_mtx[:,1][None, None,:]
#     blue_column = srgb_mtx[:,2][None, None,:]

#     print("red column: ", red_column)
#     print("green column: ", green_column)
#     print("blue column: ", blue_column)

#     print(red_column.shape)
#     print(green_column.shape)
#     print(blue_column.shape)

#     red_channel = np.sum(input_ * red_column, axis=-1)
#     green_channel = np.sum(input_ * green_column, axis=-1)
#     blue_channel = np.sum(input_ * blue_column, axis=-1)

#     print("red channel dtype: ", red_channel.dtype)

#     output = np.zeros_like(input_, dtype=np.float32)
#     output[:,:,0] = red_channel
#     output[:,:,1] = green_channel
#     output[:,:,2] = blue_channel

#     return output.astype(np.uint16)


# TODO: remove all constants from functions below
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def __brighten(input_, gain):
    output = gain * input_.astype(np.uint32)
    output[output > 65535] = 65535

    return output.astype(np.uint16)


def __gamma_correct(input_):
    input_ = input_.astype(np.int32)

    cutoff = 200
    gamma_toe = 12.92
    gamma_pow = 0.416667
    gamma_fac = 680.552897
    gamma_con = -3604.425

    inp_ravel = input_.ravel()
    mask = inp_ravel >= cutoff

    inp_ravel[mask == 1] = np.power(
        inp_ravel[mask == 1], gamma_pow) * gamma_fac + gamma_con
    inp_ravel[mask == 0] = inp_ravel[mask == 0] * gamma_toe

    return np.reshape(inp_ravel, input_.shape)


def __gamma_inverse(input_):
    input_ = input_.astype(np.int32)

    cutoff = 2575
    gamma_toe = 0.0774
    gamma_pow = 2.4
    gamma_fac = 57632.49226
    gamma_con = 0.055

    inp_ravel = input_.ravel()
    mask = inp_ravel >= cutoff
    inp_ravel[mask == 1] = np.power(
        inp_ravel[mask == 1] / 65535. + gamma_con, gamma_pow) * gamma_fac 
    inp_ravel[mask == 0] = inp_ravel[mask == 0] * gamma_toe

    return np.reshape(inp_ravel, input_.shape)


def __normal_dist(x):
    return np.exp(-12.5 * np.power(x / 65535. - .5, 2.))


def __combine(image1, image2, num_layers=2):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    unblurred1 = image1
    unblurred2 = image2

    blurred1 = cv2.GaussianBlur(image1, (7, 7), 0)
    blurred2 = cv2.GaussianBlur(image2, (7, 7), 0)

    weight1 = __normal_dist(image1)
    weight2 = __normal_dist(image2)

    init_mask1 = weight1 / (weight1 + weight2)
    init_mask2 = 1. - init_mask1

    mask1 = init_mask1
    mask2 = init_mask2

    accumulator = np.zeros_like(image1)

    for layer in range(1, num_layers):
        laplace1 = unblurred1 - blurred1
        laplace2 = unblurred2 - blurred2

        accumulator = accumulator + laplace1 * mask1 + \
            laplace2 * mask2

        unblurred1 = blurred1
        unblurred2 = blurred2

        blurred1 = cv2.GaussianBlur(blurred1, (7, 7), 0)
        blurred2 = cv2.GaussianBlur(blurred2, (7, 7), 0)

        mask1 = cv2.GaussianBlur(mask1, (7, 7), 0)
        mask2 = cv2.GaussianBlur(mask2, (7, 7), 0)

    accumulator = accumulator + blurred1 * mask1 + \
        blurred2 * mask2

    output = accumulator#.astype(np.uint16)

    return output


def tone_map(
    input_, 
    compression=3.8, 
    gain=1.1, 
    num_passes=3):

    #grayscale = np.mean(input_.astype(np.uint32), axis=-1)

    grayscale = rgb2gray(input_.astype(np.float32))
    #grayscale = grayscale.astype(np.uint16)

    dark = grayscale.copy()

    comp_const = 1. + compression / num_passes
    gain_const = 1. + gain / num_passes

    comp_slope = (compression - comp_const) / (num_passes - 1)
    gain_slope = (gain - gain_const) / (num_passes - 1)

    for pass_ in range(num_passes):
        norm_comp = pass_ * comp_slope + comp_const
        norm_gain = pass_ * gain_slope + gain_const

        bright = __brighten(dark, norm_comp)

        dark_gamma = __gamma_correct(dark)
        bright_gamma = __gamma_correct(bright)

        dark_gamma = __combine(dark_gamma, bright_gamma)

        dark = __brighten(__gamma_inverse(dark_gamma), norm_gain)

    grayscale[grayscale < 1] = 1

    output = input_.astype(np.uint32) * dark.astype(np.uint32)[:,:,None] / grayscale[:,:,None]
    output[output > 65535] = 65535

    return output


def contrast(input_, strength=5., black_level=2000):
    scale = 0.8 + 0.3 / min(1., strength)

    inner_constant = pi / (2. * scale)
    sin_constant = sin(inner_constant)

    slope = 65535. / (2. * sin_constant)
    constant = slope * sin_constant

    factor = pi / (scale * 65535.)

    val = factor * input_

    output = slope * np.sin(val - inner_constant) + constant
    #output = output.astype(np.uint16)

    white_scale = 65535. / (65535. - black_level)
    output = (output - black_level) * white_scale
    #output = output.astype(np.uint16)
    output[output < 0] = 0

    return output


def __rgb_to_yuv(input_):
    return cv2.cvtColor(input_, cv2.COLOR_RGB2YUV)


def __yuv_to_rgb(input_):
    return cv2.cvtColor(input_, cv2.COLOR_YUV2RGB)


#TODO: check Gauss blur
def sharpen(input_, strength):
    yuv_input = __rgb_to_yuv(input_.astype(np.float32))

    small_blurred = cv2.GaussianBlur(yuv_input, (7, 7), 0)
    large_blurred = cv2.GaussianBlur(small_blurred, (7, 7), 0)

    difference_of_gaussian = small_blurred - large_blurred

    output_yuv = yuv_input.copy()
    output_yuv[:,:,0] = yuv_input[:,:,0] + strength * difference_of_gaussian[:,:,0]

    output = __yuv_to_rgb(output_yuv)

    output[output < 0] = 0
    output[output > 65535] = 65535

    return output


def compress_to_8bit(image):
    return (image / 256).astype(np.uint8)


