import time

import cv2
import numpy as np

from numba import njit
from scipy.ndimage import correlate
from sklearn.linear_model import Ridge


def compute_image_grads(image): 
    kernel_hor = np.array([-1, 0, 1], dtype=np.float32).reshape(1, 3)    
    kernel_ver = kernel_hor.T

    grad_hor = correlate(image.astype(np.float32), kernel_hor)
    grad_ver = correlate(image.astype(np.float32), kernel_ver)
    
    grads = np.maximum(grad_hor, grad_ver)
    
    return grads


def compute_gradient_sensivity(image):
    height, width = image.shape

    lapl_diff = np.array([
        [ 1, -2,  1],
        [-2,  4, -2],
        [ 1, -2,  1]
    ], dtype=np.float32)

    convolved = correlate(image.astype(np.float32), lapl_diff)

    factor = np.sqrt(np.pi / 2) / (6 * (height - 2) * (width - 2))
    gradient_sens = np.abs(convolved.ravel()).sum()
    gradient_sens = gradient_sens * factor

    return gradient_sens


def get_centroids_intensities(image, num_centroids=15):
    counts = np.bincount(image.ravel())
    intensities = np.argpartition(counts, -num_centroids).astype(np.float32)
    
    return intensities[-num_centroids:]


@njit
def __compute_inten_weights(image, grad_weights, centroids, sensivity):
    weights = np.exp(-np.square(centroids - image) / sensivity) 
    weights = weights * grad_weights

    return weights


def compute_weights(image, grads_square, sensivity=80, num_centroids=15):
    """
    returns: ndarray of shape [Nc, H, W], where Nc is number of centroids
    """

    centroids = get_centroids_intensities(image, num_centroids)

    gradient_sens = compute_gradient_sensivity(image)
    grads_weights = np.exp(-grads_square / gradient_sens)

    image = image[None,:,:].copy()
    centroids = centroids[:, None, None].copy()
    grads_weights = grads_weights[None,:,:].copy()

    weights = __compute_inten_weights(
        image, grads_weights, centroids, sensivity)

    return weights


def __sum_spatial_axes(image):
    return image.sum(axis=-1).sum(axis=-1)


def __compute_est_variance(weights, grads, grads_square, denominator):
    est_variance = __sum_spatial_axes(weights * grads_square) / denominator 
    est_variance = est_variance - np.square(__sum_spatial_axes(weights * grads) / denominator)
    est_variance = est_variance * 0.5

    return est_variance


def compute_estimates(image):
    grads = compute_image_grads(image)
    grads_square = np.square(grads)

    start = time.time()
    weights = compute_weights(image, grads_square)
    end = time.time()

    print("Weights computing took: {}".format(end - start))

    start = time.time()
    denominator = __sum_spatial_axes(weights)
    end = time.time()

    print("Denominator took: {}".format(end - start))

    start = time.time()
    est_variance = __compute_est_variance(
        weights, grads, grads_square, denominator)
    end = time.time()

    print("Est variance took: {}".format(end - start))

    print("Est variance shape: {}".format(est_variance.shape))

    start = time.time()
    est_intensity = __sum_spatial_axes(weights * image) / denominator
    end = time.time()

    print("Intensity took: {}".format(end - start))

    print("Est image shape: {}".format(est_intensity.shape))

    return est_variance, est_intensity


def noise_estimation(image):
    est_variance, est_intensity = compute_estimates(image)

    reg = Ridge()
    reg.fit(est_intensity[:, None], est_variance)

    variance = reg.predict(image.ravel()[:, None])
    variance = np.reshape(variance, image.shape)

    return variance







