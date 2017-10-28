#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Author: Q.Liu
# Date: 23.10.2017
# Code for mandatory project 2
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
wrapped up functions for glcm feature extraction
Import this lib into the main project

"""

# Standard library imports
import numpy as np

# Third party imports
import cv2
# using numba to optimize code
from numba import jit
from skimage.feature import greycomatrix, greycoprops

from datasets import *


@jit
def quadrants(glcm=None, num=4):
    sumall=np.sum(glcm)
    if sumall == 0:
        print("glcm sum is zero --------")
        sumall += 0.00000001
    M, N = glcm.shape
    # feature = np.zeros(num)
    quad_features = {}
    step = int(M/np.sqrt(num))
    k = 0
    for i in range(0, M, step):
        for j in range(i, N, step):
            k += 1
            key = "q{0}".format(k)
            quad_features[key] = np.sum(glcm[i:i+step, j:j+step])/sumall

    return quad_features


@jit
def get_quad_features(glcm=None, quadnum=4):
    sumall = np.sum(glcm)

    if sumall == 0:
        print("glcm sum is zero --------")
        sumall += 0.00000001

    M, N = glcm.shape
    # feature = np.zeros(num)
    quad_features = {}
    step = int(M / np.sqrt(quadnum))
    k = 0
    for i in range(0, M, step):
        for j in range(i, N, step):
            k += 1
            key = "q{0}".format(k)
            quad_features[key] = np.sum(glcm[i:i + step, j:j + step]) / sumall

    return quad_features


def construct_quad_images(gray_img, win_order=3, features=None, offsets=None, angles=None,
                            fill_type='mirror', norm=True, symm=True, levels=256,
                            isotropic=False, weight=None,rescale=False):
    # THIS  FUNCTION  is implemented by skimage's functions
    # please first re-scale input image to 0~levels-1 gray-level,

    scl_img = requantize(gray_img, level_num=levels-1)
    if rescale:
        scl_img = scale_image(scl_img, 0, levels-1)

    feature_imgs = {}
    M, N = gray_img.shape
    if len(features) < 4:
        quad_num = 4
    else:
        quad_num = 16

    for idx in range(len(features)):
        feature_imgs[features[idx]] = np.zeros([M, N])

    if offsets is None:
        offsets = [1]
    if angles is None:
        angles = [0]
    if isotropic :
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for i in range(M):
        for j in range(N):
            win_img = slide_window(scl_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)

            glcm = get_glcm(win_image=win_img,
                            offsets=offsets,
                            angles=angles,
                            levels=levels,
                            symm=symm, norm=norm,
                            isotropic=isotropic, weights=weight)

            quad_features = get_quad_features(glcm=glcm, quadnum=quad_num)

            for idx in range(len(features)):
                feature_imgs[features[idx]][i, j] = quad_features[features[idx]]

    for idx in range(len(features)):
        feature_imgs[features[idx]] = scale_image(feature_imgs[features[idx]], 0, 255)

    return feature_imgs

def requantize(image, level_num=8):
    """
    Perform requantization on input gray image
    :param img: Gray image or 2-D array
    :param level_num:
    :return: 2-D image
    """
    M, N = image.shape
    level_space = np.linspace(0, 255, level_num)
    out_img = np.zeros([M, N], dtype='uint8')
    for i in range(M):
        for j in range(N):
            out_img[i, j] = min(level_space, key=lambda x: abs(x - image[i, j]))

    return out_img.astype('uint8')


@jit
def slide_window(gray_image, neighbor=2, current_row=0, current_col=0, fill='constant'):
    """
    Get an sliding window image with defined window size on the original image with
    specific pixel location index

    :param gray_image: input gray level image
    :param neighbor: window order - size defined by neighborhood
    :param current_row: give current location of pixel - row index
    :param current_col: give current location of pixel - column index
    :param fill: boundary filling flag, now only have 2 types, constant or mirror
    :return: window image
    """
    max_row, max_col = gray_image.shape
    win_shape = (2 * neighbor + 1, 2 * neighbor + 1)  # window shape - 5 x 5
    win_img = np.zeros(win_shape)

    for row_offset in range(-1 * neighbor, neighbor + 1):
        for col_offset in range(-1 * neighbor, neighbor + 1):

            cp_i = current_row + row_offset  # pixel row index on the image with offsets
            cp_j = current_col + col_offset  # pixel col index on the image with offsets

            if 0 <= cp_i < max_row and 0 <= cp_j < max_col:
                win_img[neighbor + row_offset, neighbor + col_offset] = gray_image[cp_i, cp_j]
            else:
                if fill is 'constant':
                    win_img[neighbor + row_offset, neighbor + col_offset] = 0
                elif fill is 'mirror':
                    if cp_i >= max_row:
                        cp_i = max_row - row_offset
                    if cp_j >= max_col:
                        cp_j = max_col - col_offset
                    if cp_i < 0:
                        cp_i = -1 - cp_i
                    if cp_j < 0:
                        cp_j = -1 - cp_j

                    win_img[neighbor + row_offset, neighbor + col_offset] = gray_image[cp_i, cp_j]

    return win_img


def get_glcm(win_image, offsets=[1],
             angles=[0, np.pi/2], levels=256,
             symm=True, norm=True, isotropic=False, weights=None,):

    if isotropic :
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm_array = greycomatrix(win_image.astype('uint8'), np.array(offsets), np.array(angles),
                 levels=levels, symmetric=symm, normed=norm)

    xi,yj,M,N = glcm_array.shape
    glcm = np.zeros([xi,yj])

    for i in range(M):
        for j in range(N):
            if isotropic:
                if weights is None:
                    weights = [0.25, 0.25, 0.25, 0.25]
                glcm += glcm_array[:,:,i,j]* weights[j] / M
            else:
                glcm += glcm_array[:, :, i, j] / (M*N)

    return glcm


@jit
def symmetrise(glcm):
    return glcm + glcm.T


@jit
def normalize(glcm):
    return glcm / float(sum(glcm.flatten()))


@jit
def get_feature(glcm, featurelist=['contrast']):
    """
    Perform computing all kinds of measures/features based on a given glcm
    :param glcm: input glcm matrix
    :param featurelist: measures name list
    :return: measure_list
    """
    measure_list = dict(max_prob=0, contrast=0, dissimilarity=0, homogeneity=0, ASM=0, energy=0, entropy=0,
                        correlation=0, cluster_shade=0, variance_i=0, variance_j=0, mean_i=0, mean_j=0)

    M, N = glcm.shape

    np.seterr(divide='ignore', invalid='ignore')

    flat_glcm = glcm.flatten()
    index_i = np.arange(0, M)  # row index
    index_j = np.arange(0, N)  # column index = row

    sum_v = np.sum(glcm, axis=0)  # sum column[] , vertical
    sum_h = np.sum(glcm, axis=1)  # sum row[] , horizontal

    max_prob = np.max(flat_glcm)
    mean_i = np.dot(index_i, sum_h.flatten())
    mean_j = np.dot(index_j, sum_v.flatten())
    var_i = np.dot((index_i - mean_i) ** 2, sum_h.flatten())
    var_j = np.dot((index_j - mean_j) ** 2, sum_v.flatten())

    measure_list['max_prob'] = max_prob
    measure_list['variance_i'] = var_i
    measure_list['variance_j'] = var_j
    measure_list['mean_i'] = mean_i
    measure_list['mean_j'] = mean_j

    for name in featurelist:
        if name in measure_list.keys():
            if name is 'max_prob':
                measure_list[name] = np.max(flat_glcm)
            elif name is 'ASM':
                measure_list[name] = np.dot(flat_glcm, flat_glcm)
            elif name is 'energy':
                ASM = np.dot(flat_glcm, flat_glcm)
                measure_list[name] = np.sqrt(ASM)
            elif name is 'cluster_shade':
                cluster_weights = np.zeros([M, N])
                for i in range(M):
                    for j in range(N):
                        cluster_weights[i, j] = (i + j - mean_i - mean_j) ** 3
                measure_list[name] = np.dot(flat_glcm, cluster_weights.flatten())
            elif name is 'correlation':
                stdev_i = np.sqrt(var_i)
                stdev_j = np.sqrt(var_j)
                correl_weights = np.outer((index_i - mean_i), (index_j - mean_j)) / (stdev_i * stdev_j)
                measure_list[name] = np.dot(flat_glcm, correl_weights.flatten())
            elif name is 'contrast':
                contrast_weights = np.zeros([M, N])
                for i in range(M):
                    for j in range(N):
                        contrast_weights[i, j] = (i - j) ** 2
                measure_list[name] = np.dot(flat_glcm, contrast_weights.flatten())
            elif name is 'entropy':
                # ln = np.log(flat_glcm) here, log(0) = -inf, will have some problem, using np.ma.log instead
                # np.ma.log(0) = -- : not -inf. ? can pass
                ln = np.ma.log(flat_glcm)
                measure_list[name] = -np.dot(flat_glcm, ln)
            elif name is 'dissimilarity':
                dissi_weights = np.zeros([M, N])
                for i in range(M):
                    for j in range(N):
                        dissi_weights[i, j] = abs(i - j)
                measure_list[name] = np.dot(flat_glcm, dissi_weights.flatten())
            elif name is 'homogeneity':
                homo_weights = np.zeros([M, N])
                for i in range(M):
                    for j in range(N):
                        homo_weights[i, j] = 1 / (1 + (i - j) ** 2)
                measure_list[name] = np.dot(flat_glcm, homo_weights.flatten())

    return measure_list


# @jit
def construct_texture_image(gray_img, win_order=3, feature='contrast', offsets=None, angles=None,
                            fill_type='mirror', norm=True, symm=True, levels=256,
                            isotropic=False, weight=None,rescale=False):
    # THIS  FUNCTION  is implemented by skimage's functions
    # please first re-scale input image to 0~levels-1 gray-level,

    scl_img = requantize(gray_img, level_num=levels-1)
    if rescale:
        scl_img = scale_image(scl_img, 0, levels-1)

    M, N = gray_img.shape
    feature_img = np.zeros([M, N])

    if offsets is None:
        offsets = [1]
    if angles is None:
        angles = [0]
    if isotropic :
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for i in range(M):
        for j in range(N):
            win_img = slide_window(scl_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)

            glcm = get_glcm(win_image=win_img,
                            offsets=offsets,
                            angles=angles,
                            levels=levels,
                            symm=symm, norm=norm,
                            isotropic=isotropic, weights=weight)

            feature_img[i, j] = get_feature(glcm, [feature])[feature]

            #
            # The blow code is another way by using skimage-greycoprops() to calculate, faster than my function
            #
            # glcm = greycomatrix(win_img, np.array(offsets), np.array(angles),
            #                     levels=levels, symmetric=symm, normed=norm)
            #
            # if feature is 'cluster_shade' or 'entropy':
            #     _, _, Ki, Kj = glcm.shape
            #     val = 0
            #     for k_i in range(Ki):
            #         for k_j in range(Kj):
            #             val += get_feature(glcm[:, :, k_i, k_j], [feature])[feature]
            #
            #     feature_img[i, j] = val
            # else:
            #     feature_img[i, j] = greycoprops(glcm, feature).sum()

    return scale_image(feature_img, 0, 255)

# @jit(nopython=True)
@jit
def scale_image(image, min_val=0, max_val=255):
    image = image.astype(float)
    im_max = np.nanmax(image) # if using np.max, sometimes will return Nan values
    im_min = np.nanmin(image)

    if im_max == im_min:
        print("scale image error min==max: ", im_max, im_min)
        scale_img = image - image
    else:
        print("scaling images: ", im_min, im_max)
        scale_img = min_val + (1 - (im_max - image) / (im_max - im_min)) * max_val

    return scale_img.astype('uint8')

@jit
def normalize2(image):
    im_max = np.nanmax(image)  # if using np.max, sometimes will return Nan values
    im_min = np.nanmin(image)
    norm_img = (image - im_min) / (im_max - im_min) # make value between 0-1
    return norm_img


def mask_featured_image(image, feature_img, threshold=40, above=True):
    mask = (feature_img < threshold) * (1 - above * 1) + (feature_img >= threshold) * (above * 1)
    return image * mask

def mask_feature(feature_img, th_low=0, th_high=40):
    mask = (feature_img <= th_high) * (feature_img >= th_low)
    return mask

def subimage(path_image, height=None, width=None, stepsize=1):
    """
    Perform image patch extraction with specific height-width sliding window by stepsize
    if no specific window size, the input image will be split into 4 patches equally as default
    :param path_image: full path and file name
    :param height:  patch height cropped
    :param width:   patch width cropped
    :param stepsize:  sliding step size
    :return: patches extracted
    """
    img = cv2.imread(path_image)
    h, w = img.shape[:2]

    if height is None or width is None:
        height = int(h / 2)
        width = int(w / 2)
        stepsize = width

    for x in range(0, h, stepsize):
        px = x
        end_x = x + height
        if end_x > h:
            end_x = h
            px = max(end_x - height, 0)

        for y in range(0, w, stepsize):
            py = y
            end_y = y + width
            if end_y > w:
                end_y = w
                py = max(end_y - width, 0)

            yield img[px:end_x, py:end_y]


@jit
def get_basic_feature(image, featurelist=['entropy']):
    """
    Perform some basic (1-order statistic) texture features for gray-level images.
    :param image: 2-D gray-level image
    :return: info_list, some basic features
    """
    features = {'min': image.min(), 'max': image.max(), 'variance': 0, 'mean': 0, 'std_dev': 0, 'skewness': 0,
                 'kurtosis': 0, 'entropy': 0, 'energy': 0, 'smoothness': 0, 'coefficient': 0}
    # hist, - = np.histogram(image.flatten(), bins=256, range=[0, 255], density=False)

    # hist = cv2.calcHist(image,[0],None,[256],[0,256])
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 255], density=False)
    hx = hist.ravel() / hist.sum()
    # mean = np.mean(image.flatten())
    x = np.arange(256)
    mean = hx.dot(x)
    variance = ((x - mean) ** 2).dot(hx)
    std = np.sqrt(variance)

    features['mean'] = mean
    features['variance'] = variance
    features['std_dev'] = std

    for name in featurelist:
        if name in features.keys():
            if name is 'skewness':
                features[name] = ((x - mean) ** 3).dot(hx) / std ** 3 # different with lecture notes
            elif name is 'kurtosis':
                features[name] = (((x - mean) ** 4) * hx).sum() / std ** 4 - 3  # different with lecture notes
            elif name is 'energy':
                features[name] = (hx * hx).sum()
            elif name is 'smoothness':
                features[name] = 1 - 1 / (1 + variance)
            elif name is 'coefficient':
                features[name] = float(std) / mean
            elif name is 'entropy':
                # ref: https://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
                log_h = np.log2(hx + 0.00001)
                features[name] = -1 * (log_h * hx).sum()

    return features

def filter_image(gray_img, win_order=3, feature='mean', fill_type='mirror'):
    M, N = gray_img.shape
    filter_img = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            win_img = slide_window(gray_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)
            filter_img[i, j] = get_basic_feature(win_img, [feature])[feature]

    return scale_image(filter_img, 0, 255)

def noise_reduction(gray_img, win_order=2, feature='smoothness',fill_type='mirror'):
    M, N = gray_img.shape
    filter_img = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            win_img = slide_window(gray_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)
            dct_img = np.float32(win_img[1:,1:]) / 255.0
            dst_img = cv2.dct(dct_img)
            filter_img[i, j] = get_basic_feature(dst_img, [feature])[feature]

    return scale_image(filter_img)


@jit
def glcm_weights(glcm, name=None, normed=False, symmetric=False):
    """
    Perform computing feature weights for glcm measures
    :param glcm: input glcm metrix
    :param name: weights name (contrast, dissimilarity, homogeneity,correlation,cluster_shade)

    :param normed: normalize glcm if want
    :param symmetric: symmetrise glcm if want
    :return: weight matrix
    """

    M, N = glcm.shape

    np.seterr(divide='ignore', invalid='ignore')

    if symmetric:
        # symmetrisation
        glcm = symmetrise(glcm)
        if normed:
            glcm = normalize(glcm)
    else:
        if normed:
            glcm = normalize(glcm)

    index_i = np.arange(0, M)  # row index
    index_j = np.arange(0, N)  # column index = row

    sum_v = np.sum(glcm, axis=0)  # sum column[] , vertical
    sum_h = np.sum(glcm, axis=1)  # sum row[] , horizontal

    mean_i = np.dot(index_i, sum_h.flatten())
    mean_j = np.dot(index_j, sum_v.flatten())

    var_i = np.dot((index_i - mean_i) ** 2, sum_h.flatten())
    var_j = np.dot((index_j - mean_j) ** 2, sum_v.flatten())

    stdev_i = np.sqrt(var_i)
    stdev_j = np.sqrt(var_j)

    weights = np.zeros([M, N])

    if name is 'correlation':
        weights = np.outer((index_i - mean_i), (index_j - mean_j)) / (stdev_i * stdev_j)
    else:
        for i in range(M):
            for j in range(N):
                if name is 'contrast':
                    weights[i, j] = (i - j) ** 2
                elif name is 'dissimilarity':
                    weights[i, j] = abs(i - j)
                elif name is 'homogeneity':
                    weights[i, j] = 1 / (1 + (i - j) ** 2)
                elif name is 'cluster_shade':
                    weights[i, j] = (i + j - mean_i - mean_j) ** 3

    return weights
