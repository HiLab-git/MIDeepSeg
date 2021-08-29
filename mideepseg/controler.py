import os
from collections import OrderedDict
from os.path import join as opj

import cv2
import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import color, measure
from network import UNet
from utils import (add_countor, add_overlay, cropped_image, extends_points,
                   extreme_points, get_bbox, get_largest_two_component,
                   get_start_end_points, interaction_euclidean_distance,
                   interaction_gaussian_distance,
                   interaction_geodesic_distance,
                   interaction_refined_geodesic_distance,
                   itensity_normalization, itensity_normalize_one_volume,
                   itensity_standardization, softmax, softmax_seg, zoom_image)

rootPATH = os.path.abspath(".")


class Controler(object):
    seeds = 0
    extreme_points = 5
    foreground = 2
    background = 3
    imageName = "../mideepseg/logo.png"
    model_path = "../mideepseg/iter_15000.pth"

    def __init__(self):
        self.img = None
        self.step = 0
        self.image = None
        self.mask = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.extreme_point_seed = []
        self.background_seeds = []
        self.foreground_seeds = []
        self.current_overlay = self.seeds
        self.load_image(self.imageName)

        self.initial_seg = None
        self.initial_extreme_seed = None

    def initial_param(self):
        self.step = 0
        self.img = None
        self.image = None
        self.mask = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.extreme_point_seed = []
        self.background_seeds = []
        self.foreground_seeds = []
        self.current_overlay = self.seeds
        self.initial_seg = None
        self.initial_extreme_seed = None

    def load_image(self, filename):
        self.filename = filename
        self.initial_param()
        self.init_image = cv2.imread(filename)
        self.image = cv2.imread(filename)
        self.img = np.array(Image.open(filename).convert('L'))
        self.images = cv2.imread(filename)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None
        self.refined_clicks = 0
        self.refined_iterations = 0

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1),
                              (x + 1, y + 1), (255, 0, 255), 2)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                if self.step == 0:
                    self.extreme_point_seed.append((x, y))
                    cv2.rectangle(self.seed_overlay, (x - 1, y - 1),
                                  (x + 1, y + 1), (255, 255, 0), 2)
                if self.step == 1:
                    self.foreground_seeds.append((x, y))
                    cv2.rectangle(self.seed_overlay, (x - 1, y - 1),
                                  (x + 1, y + 1), (0, 0, 255), 2)
                if len(self.extreme_point_seed) == 1:
                    import time
                    self.stage1_begin = time.time()
        if len(self.background_seeds) > 0 or len(self.foreground_seeds) > 0:
            self.refined_clicks += 1

        if self.refined_clicks == 1:
            import time
            self.stage2_begin = time.time()
        if self.refined_clicks == 0:
            import time
            self.stage2_begin = None

    def clear_seeds(self):
        self.step = 0
        self.background_seeds = []
        self.foreground_seeds = []
        self.extreme_point_seed = []
        self.background_superseeds = []
        self.foreground_superseeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)
        self.image = self.init_image

    def get_image_with_overlay(self, overlayNumber):
        return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.7, 0.7)

    def segment_show(self):
        pass

    def save_image(self, filename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        self.mask = self.mask * 255
        cv2.imwrite(str(filename), self.mask.astype(int))

    def extreme_segmentation(self):
        if self.step == 0:
            seed = np.zeros_like(self.img)
            for i in self.extreme_point_seed:
                seed[i[1], i[0]] = 1
            if seed.sum() == 0:
                print('Please provide initial seeds for segmentation.')
                return
            seed = extends_points(seed)
            self.initial_extreme_seed = seed
            bbox = get_start_end_points(seed)
            cropped_img = cropped_image(self.img, bbox)
            x, y = cropped_img.shape
            normal_img = itensity_normalization(cropped_img)

            cropped_seed = cropped_image(seed, bbox)
            cropped_geos = interaction_geodesic_distance(
                normal_img, cropped_seed)
            # cropped_geos = itensity_normalization(cropped_geos)

            zoomed_img = zoom_image(normal_img)
            zoomed_geos = zoom_image(cropped_geos)

            inputs = np.asarray([[zoomed_img, zoomed_geos]])
            if torch.cuda.is_available():
                inputs = torch.from_numpy(inputs).float().cuda()
            else:
                inputs = torch.from_numpy(inputs).float().cpu()
            net = self.initial_model()
            net.eval()
            output = net(inputs)
            output = torch.softmax(output, dim=1)
            output = output.squeeze(0)
            predict = output.cpu().detach().numpy()
            fg_prob = predict[1]
            bg_prob = predict[0]

            crf_param = (5.0, 0.1)
            Prob = np.asarray([bg_prob, fg_prob])
            Prob = np.transpose(Prob, [1, 2, 0])
            fix_predict = maxflow.maxflow2d(zoomed_img.astype(
                np.float32), Prob, crf_param)

            fixed_predict = zoom(fix_predict, (x/96, y/96), output=None,
                                 order=0, mode='constant', cval=0.0, prefilter=True)
            # fixed_predict = zoom(fg_prob, (x/96, y/96), output=None,
            #                      order=0, mode='constant', cval=0.0, prefilter=True)

            pred = np.zeros_like(self.img, dtype=np.float)

            pred[bbox[0]:bbox[2], bbox[1]:bbox[3]] = fixed_predict
            self.initial_seg = pred

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            strt = ndimage.generate_binary_structure(2, 1)
            seg = np.asarray(
                ndimage.morphology.binary_opening(pred, strt), np.uint8)
            seg = np.asarray(
                ndimage.morphology.binary_closing(pred, strt), np.uint8)
            seg = self.largestConnectComponent(seg)
            seg = ndimage.binary_fill_holes(seg)

            seg = np.clip(seg, 0, 255)
            seg = np.array(seg, np.uint8)

            contours, hierarchy = cv2.findContours(
                seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if len(contours) != 0:
                image_data = cv2.drawContours(
                    self.image, contours, -1, (0, 255, 0), 2)

            self.image = image_data
            self.mask = seg
            self.step = 1

    def largestConnectComponent(self, img):
        binaryimg = img

        label_image, num = measure.label(
            binaryimg, background=0, return_num=True)
        areas = [r.area for r in measure.regionprops(label_image)]
        areas.sort()
        if len(areas) > 1:
            for region in measure.regionprops(label_image):
                if region.area < areas[-1]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        label_image = label_image.astype(np.int8)
        label_image[np.where(label_image > 0)] = 1
        return label_image

    def initial_model(self):
        model = UNet(2, 2, 16)
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        model.load_state_dict(torch.load(self.model_path))
        return model

    def refined_seg(self):
        fore_seeds = np.zeros_like(self.img)
        for i in self.foreground_seeds:
            fore_seeds[i[1], i[0]] = 1
        back_seeds = np.zeros_like(self.img)
        for i1 in self.background_seeds:
            back_seeds[i1[1], i1[0]] = 1

        fore_seeds = extends_points(fore_seeds)
        back_seeds = extends_points(back_seeds)

        all_refined_seeds = np.maximum(fore_seeds, back_seeds)
        all_seeds = np.maximum(all_refined_seeds, self.initial_extreme_seed)

        bbox = get_start_end_points(all_seeds)
        cropped_img = cropped_image(self.img, bbox)

        normal_img = itensity_standardization(cropped_img)
        init_seg = [self.initial_seg, 1.0-self.initial_seg]
        fg_prob = init_seg[0]
        bg_prob = init_seg[1]

        cropped_initial_seg = cropped_image(fg_prob, bbox)
        cropped_fore_seeds = cropped_image(fore_seeds, bbox)

        cropped_fore_geos = interaction_refined_geodesic_distance(
            normal_img, cropped_fore_seeds)
        cropped_back_seeds = cropped_image(back_seeds, bbox)
        cropped_back_geos = interaction_refined_geodesic_distance(
            normal_img, cropped_back_seeds)

        fore_prob = np.maximum(cropped_fore_geos, cropped_initial_seg)

        cropped_back_seg = cropped_image(bg_prob, bbox)
        back_prob = np.maximum(cropped_back_geos, cropped_back_seg)

        crf_seeds = np.zeros_like(cropped_fore_seeds, np.uint8)
        crf_seeds[cropped_fore_seeds > 0] = 170
        crf_seeds[cropped_back_seeds > 0] = 255
        crf_param = (5.0, 0.1)

        crf_seeds = np.asarray([crf_seeds == 255, crf_seeds == 170], np.uint8)
        crf_seeds = np.transpose(crf_seeds, [1, 2, 0])

        x, y = fore_prob.shape
        prob_feature = np.zeros((2, x, y), dtype=np.float32)
        prob_feature[0] = fore_prob
        prob_feature[1] = back_prob
        softmax_feture = np.exp(prob_feature) / \
            np.sum(np.exp(prob_feature), axis=0)
        softmax_feture = np.exp(softmax_feture) / \
            np.sum(np.exp(softmax_feture), axis=0)
        fg_prob = softmax_feture[0].astype(np.float32)
        bg_prob = softmax_feture[1].astype(np.float32)

        Prob = np.asarray([bg_prob, fg_prob])
        Prob = np.transpose(Prob, [1, 2, 0])

        refined_pred = maxflow.interactive_maxflow2d(
            normal_img, Prob, crf_seeds, crf_param)

        pred = np.zeros_like(self.img, dtype=np.float)
        pred[bbox[0]:bbox[2], bbox[1]:bbox[3]] = refined_pred

        pred = self.largestConnectComponent(pred)
        strt = ndimage.generate_binary_structure(2, 1)
        seg = np.asarray(
            ndimage.morphology.binary_opening(pred, strt), np.uint8)
        seg = np.asarray(
            ndimage.morphology.binary_closing(pred, strt), np.uint8)
        contours, hierarchy = cv2.findContours(
            seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        img = self.images.copy()
        image_data = cv2.drawContours(
            self.images, contours, -1, (0, 255, 0), 2)
        self.images = img

        self.image = image_data
        self.mask = seg
