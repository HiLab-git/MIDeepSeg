import os
import random
from os.path import join as opj

import GeodisTK
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torchvision.transforms as ts
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import color, measure
import cv2

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    volume = (volume - volume.min()) / (volume.max() - volume.min())
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    out = out.astype(np.float32)
    return out


def extreme_points(mask, pert=0):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)),  # left
                     find_point(inds_x, inds_y, np.where(
                         inds_x >= np.max(inds_x)-pert)),  # right
                     find_point(inds_x, inds_y, np.where(
                         inds_y <= np.min(inds_y)+pert)),  # top
                     find_point(inds_x, inds_y, np.where(
                         inds_y >= np.max(inds_y)-pert))  # bottom
                     ])


def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max


def cropped_image(image, bbox, pixel=0):
    random_bbox = [bbox[0] - pixel, bbox[1] -
                   pixel, bbox[2] + pixel, bbox[3] + pixel]
    cropped = image[random_bbox[0]:random_bbox[2],
                    random_bbox[1]:random_bbox[3]]
    return cropped


def zoom_image(data):
    """
    reshape image to 64*64 pixels
    """
    x, y = data.shape
    zoomed_image = zoom(data, (96 / x, 96 / y))
    # zoomed_image = zoom(data, (128 / x, 128 / y))
    return zoomed_image


def extends_points(seed):
    if(seed.sum() > 0):
        points = ndimage.distance_transform_edt(seed == 0)
        points[points > 2] = 0
        points[points > 0] = 1
    else:
        points = seed
    return points.astype(np.uint8)


def gaussian_kernel(d, bias=0, sigma=10):
    """
    this a gaussian kernel
    input:
        d: distance between each extreme point to every point in volume
        bias:
        sigma: is full-width-half-maximum, which can be thought of as an effective radius.
    """
    gaus_dis = (1 / (sigma * np.sqrt(2 * np.pi))) * \
        np.exp(- ((d - bias)**2 / (2 * sigma**2)))
    return gaus_dis


def interaction_euclidean_distance(img, seed):
    if seed.sum() > 0:
        euc_dis = ndimage.distance_transform_edt(seed == 0)
    else:
        euc_dis = np.ones_like(seed, dtype=np.float32)
    euc_dis = cstm_normalize(euc_dis)
    return euc_dis


def interaction_gaussian_distance(img, seed, sigma=10, bias=0):
    if seed.sum() > 0:
        euc_dis = ndimage.distance_transform_edt(seed == 0)
        gaus_dis = gaussian_kernel(euc_dis, bias, sigma)
    else:
        gaus_dis = np.zeros_like(seed, dtype=np.float32)
    gaus_dis = cstm_normalize(gaus_dis)
    return gaus_dis


def interaction_geodesic_distance(img, seed, threshold=0):
    if seed.sum() > 0:
        # I = itensity_normalize_one_volume(img)
        I = np.asanyarray(img, np.float32)
        S = seed
        geo_dis = GeodisTK.geodesic2d_fast_marching(I, S)
        # geo_dis = GeodisTK.geodesic2d_raster_scan(I, S, 1.0, 2.0)
        if threshold > 0:
            geo_dis[geo_dis > threshold] = threshold
            geo_dis = geo_dis / threshold
        else:
            geo_dis = np.exp(-geo_dis)
    else:
        geo_dis = np.zeros_like(img, dtype=np.float32)
    return cstm_normalize(geo_dis)


def interaction_refined_geodesic_distance(img, seed, threshold=0):
    if seed.sum() > 0:
        # I = itensity_normalize_one_volume(img)
        I = np.asanyarray(img, np.float32)
        S = seed
        geo_dis = GeodisTK.geodesic2d_fast_marching(I, S)
        if threshold > 0:
            geo_dis[geo_dis > threshold] = threshold
            geo_dis = geo_dis / threshold
        else:
            geo_dis = np.exp(-geo_dis**2)
    else:
        geo_dis = np.zeros_like(img, dtype=np.float32)
    return geo_dis


def cstm_normalize(im, max_value=1.0):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def get_start_end_points(scribbles):
    points = np.where(scribbles != 0)
    minZidx = int(np.min(points[0]))
    maxZidx = int(np.max(points[0]))
    minXidx = int(np.min(points[1]))
    maxXidx = int(np.max(points[1]))
    start_end_points = [minZidx - 5, minXidx - 5, maxZidx + 5, maxXidx + 5]
    return start_end_points


def add_countor(In, Seg, Color=(0, 255, 0)):
    Out = In.copy()
    [H, W] = In.size
    for i in range(H):
        for j in range(W):
            if(i == 0 or i == H-1 or j == 0 or j == W-1):
                if(Seg.getpixel((i, j)) != 0):
                    Out.putpixel((i, j), Color)
            elif(Seg.getpixel((i, j)) != 0 and
                 not(Seg.getpixel((i-1, j)) != 0 and
                     Seg.getpixel((i+1, j)) != 0 and
                     Seg.getpixel((i, j-1)) != 0 and
                     Seg.getpixel((i, j+1)) != 0)):
                Out.putpixel((i, j), Color)
    return Out


def add_overlay(image, seg_name, Color=(0, 255, 0)):
    seg = Image.open(seg_name).convert('L')
    seg = np.asarray(seg)
    if(image.size[1] != seg.shape[0] or image.size[0] != seg.shape[1]):
        print('segmentation has been resized')
    seg = scipy.misc.imresize(
        seg, (image.size[1], image.size[0]), interp='nearest')
    strt = ndimage.generate_binary_structure(2, 1)
    seg = np.asarray(ndimage.morphology.binary_opening(seg, strt), np.uint8)
    seg = np.asarray(ndimage.morphology.binary_closing(seg, strt), np.uint8)

    img_show = add_countor(image, Image.fromarray(seg), Color)
    strt = ndimage.generate_binary_structure(2, 1)
    seg = np.asarray(ndimage.morphology.binary_dilation(seg, strt), np.uint8)
    img_show = add_countor(img_show, Image.fromarray(seg), Color)
    return img_show


def get_largest_two_component(img, prt=False, threshold=None):
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(prt):
        print("component size", sizes_list)
    if(len(sizes) == 1):
        return img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(prt):
                print(max_size2, max_size1, max_size2/max_size1)
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0

            return component1


def softmax_seg(seg):
    m, n = seg.shape
    prob_feature = np.zeros((2, m, n), dtype=np.float32)
    prob_feature[0] = seg
    prob_feature[1] = 1.0 - seg
    softmax_feture = np.exp(prob_feature)/np.sum(np.exp(prob_feature), axis=0)
    fg_prob = softmax_feture[0].astype(np.float32)
    return fg_prob


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0).astype(np.float32)


def itensity_standardization(image):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = image[image > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (image - mean)/std
    out = out.astype(np.float32)
    return out


def itensity_normalization(image):
    out = (image - image.min()) / (image.max() - image.min())
    out = out.astype(np.float32)
    return out
