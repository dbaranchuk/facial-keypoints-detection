# -*- coding: utf-8 -*-
from math import ceil, sqrt

from skimage.transform import resize

import numpy as np
import time
from sys import argv, stdout, exit
from skimage.io import imread
from detection import train_detector, detect
from PIL import Image, ImageDraw

from skimage.transform import resize
import matplotlib.pyplot as plt

TEST_SIZE = 1000
FACEPOINTS_COUNT = 14
VISUALIZE = True

def visualize_grid(Xs, ubound=255.0, padding=1):
    
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

def load_data(path, return_paths=False):
    fi = open(path + '/gt.txt')
    lines = [line if line[-1] != '\n' else line[:-1]
             for line in fi.readlines()]
    fi.close()
    i = 0
    data = []
    gt = []
    if return_paths:
        impaths = []
    while i < len(lines) - FACEPOINTS_COUNT:
        imgdata = imread(path + '/images/' + lines[i], plugin='matplotlib')
        
        if return_paths:
            impaths.append(lines[i])
        if len(imgdata.shape) < 3:
            imgdata = np.array(
                [imgdata, imgdata, imgdata]).transpose((1, 2, 0))
        i += 1
        imggt = np.zeros((FACEPOINTS_COUNT, 2))
        for j in range(FACEPOINTS_COUNT):
            str_text = lines[i + j].split(';')
            nums = [int(s) for s in str_text]
            imggt[nums[0], :] = nums[1:]
        i += FACEPOINTS_COUNT

        data.append(imgdata)
        gt.append(imggt)

    data = np.array(data)
    gt = np.array(gt)
    if return_paths:
        return (data, gt, impaths)
    else:
        return (data, gt)


def compute_metrics(imgs, detected, gt):
    if len(detected) != len(gt):
        raise "Sizes don't match"
    diff = np.array(detected, dtype=np.float64) - np.array(gt)
    for i in range(len(imgs)):
        diff[i, :, 1] /= imgs[i].shape[0]
        diff[i, :, 0] /= imgs[i].shape[1]
    return np.sqrt(np.sum(diff ** 2) / (len(imgs) * 2 * FACEPOINTS_COUNT))


def visualise(imgs, detection_points, gt_points, res_dir, relative_radius=0.02, detection_color=(255, 0, 0), gt_color = (0, 255, 0)):
    for i in range(len(imgs)):
        pil_img = Image.fromarray(imgs[i])
        pil_draw = ImageDraw.Draw(pil_img)
        radius = relative_radius * min(pil_img.height, pil_img.width)
        for j in range(FACEPOINTS_COUNT):
            pt1 = detection_points[i, j, :]
            pt2 = gt_points[i, j, :]
            pil_draw.ellipse(
                (pt1[0] - radius, pt1[1] - radius, pt1[0] + radius, pt1[1] + radius), fill=detection_color)
            pil_draw.ellipse(
                (pt2[0] - radius, pt2[1] - radius, pt2[0] + radius, pt2[1] + radius), fill=gt_color)

        pil_img.save(res_dir + '/out' + str(i) + '.jpg')#impaths[i]

start_time = time.time()
train_dir = '/Users/dmitrybaranchuk/cvintro2016/hw-06/10-facial-keypoints-pub'
model_dir = '/Users/dmitrybaranchuk/cvintro2016/hw-06/my_model.h5'
res_dir = '/Users/dmitrybaranchuk/cvintro2016/hw-06/results'

train_imgs, train_gt = load_data(train_dir)
print("Data has been loaded")

train_num = len(train_imgs)
test_num = TEST_SIZE

mask = range(test_num)
test_imgs = train_imgs[mask]
test_gt = train_gt[mask]

mask = range(test_num, 6 * test_num)
train_imgs = train_imgs[mask]
train_gt = train_gt[mask]

try:
    barnet = train_detector(train_imgs, train_gt)
    barnet.model.save_weights(model_dir, overwrite=True)
#    barnet.model.to_json()

    layer = barnet.model.layers[0]
    weights = np.array(layer.get_weights()[0])
    grid = visualize_grid(weights.transpose(0, 2, 3, 1))
    grid = resize(grid, (512, 512, 3))
    grid_img = Image.fromarray(grid.astype('uint8'))
    grid_img.save(res_dir + '/layer_weights.jpg')

finally:
    del train_imgs, train_gt

print("CNN has been trained")

detection_results = np.array(detect(barnet, test_imgs))
print("Result: %.4f" % compute_metrics(test_imgs, detection_results, test_gt))

if VISUALIZE:
    visualise(test_imgs, detection_results, test_gt, res_dir)

end_time = time.time()
print("Running time:", round(end_time - start_time, 2),
      's (' + str(round((end_time - start_time) / 60, 2)) + " minutes)")


