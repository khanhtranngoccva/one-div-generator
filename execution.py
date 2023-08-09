#!usr/bin/env python3

"""
File name: execution.py
Author: Peter Maldonado
Date created: 3/05/2019
Date last modified: 3/11/2019
Python Version: 3.7

This module contains the methods need to triangulate an image using Delaunay
Triangulation as the underlying algorithm.
"""

import argparse
import os
import sys

import numpy as np
import tqdm
from numpy.random import randint, uniform, choice
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import KDTree, Delaunay
from scipy.signal import convolve2d

import skimage.restoration
from skimage.color import rgb2gray, gray2rgb, rgb2lab
from skimage.draw import polygon, polygon_perimeter, circle_perimeter
from skimage.feature import canny
from skimage.filters import gaussian, scharr
from skimage.filters.rank import entropy
from skimage.io import imread, imsave, imshow
from skimage.morphology import disk, dilation
from skimage.restoration import denoise_bilateral
from skimage.transform import pyramid_reduce
from skimage.util import img_as_ubyte, invert, img_as_float64

import time

import triangle_util


def visualize_sample(img, weights, sample_points):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')

    ax2.imshow(weights, cmap='gray')
    ax2.axis('off')

    heatmap = gray2rgb(img_as_float64(weights))
    for point in sample_points:
        rr, cc = circle_perimeter(point[0], point[1], 2, shape=weights.shape)
        heatmap[rr, cc, 0] = 1
    ax3.imshow(heatmap)
    ax3.axis('off')

    fig.tight_layout()

    plt.show()


def generate_sample_points(img, max_points):
    '''
    Generates samples points for triangulation of a given image.

    Parameters
    ----------
    img : np.array
        The image to sample.

    Returns
    -------
    list :
        The list of points to triangulate.
    '''
    width = img.shape[0]
    height = img.shape[1]
    n = min(round(height * width * args.rate), max_points)

    print("Preprocessing...")
    t0 = time.perf_counter()
    if args.process == 'approx-canny':
        weights = approx_canny(img, args.blur)
    elif args.process == 'edge-entropy':
        weights = edge_entropy(img)
    t1 = time.perf_counter()
    if args.time:
        print(f"Preprocess timer: {round(t1 - t0, 3)} seconds.")

    print("Sampling...")
    t0 = time.perf_counter()
    if args.sample == 'threshold':
        threshold = args.threshold
        sample_points = threshold_sample(n, weights, threshold)
    elif args.sample == 'disk':
        sample_points = poisson_disk_sample(n, weights)
    t1 = time.perf_counter()
    if args.time:
        print(f"Sample timer: {round(t1 - t0, 3)} seconds.")

    if args.debug:
        visualize_sample(img, weights, sample_points)
    corners = np.array([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
    return np.append(sample_points, corners, axis=0)


def approx_canny(img, blur):
    '''
    Weights pixels based on an approximate canny edge-detection algorithm.

    Parameters
    ----------
    img : ndarray
        Image to weight.
    blur : int
        Blur radius for pre-processing.

    Returns
    -------
    ndarray : 
        Noramlized weight matrix for pixel sampling.
    '''
    edge_threshold = 3 / 256

    gray_img = rgb2gray(img)
    blur_filt = np.ones(shape=(2 * blur + 1, 2 * blur + 1)) / ((2 * blur + 1) ** 2)
    blurred = convolve2d(gray_img, blur_filt, mode='same', boundary='symm')
    edge_filt = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    edge = convolve2d(blurred, edge_filt, mode='same', boundary='symm')
    for idx, val in np.ndenumerate(edge):
        if val < edge_threshold:
            edge[idx] = 0
    dense_filt = np.ones((3, 3))
    dense = convolve2d(edge, dense_filt, mode='same', boundary='symm')
    dense /= np.amax(dense)

    if args.debug:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                            sharex=True, sharey=True)
        ax1.imshow(blurred)
        ax1.axis('off')

        ax2.imshow(edge)
        ax2.axis('off')

        ax3.imshow(dense)
        ax3.axis('off')

        fig.tight_layout()
        plt.show()
    return dense


def edge_entropy(img, bal=0.1):
    '''
    Weights pixels based on a weighted edge-detection and entropy balance.

    Parameters
    ----------
    img : ndarray
        Image to weight.
    bal : float (optional)
        How much to value entropy (bal) versus edge-detection (1 - bal)

    Returns
    -------
    ndarray : 
        Noramlized weight matrix for pixel sampling.
    '''
    dn_img = skimage.restoration.denoise_tv_bregman(img, 0.1)
    img_gray = rgb2gray(dn_img)
    img_lab = rgb2lab(dn_img)

    entropy_img = gaussian(img_as_float64(dilation(entropy(img_as_ubyte(img_gray), disk(5)), disk(5))))
    edges_img = dilation(np.mean(np.array([scharr(img_lab[:, :, channel]) for channel in range(3)]), axis=0), disk(3))

    weight = (bal * entropy_img) + ((1 - bal) * edges_img)
    weight /= np.mean(weight)
    weight /= np.amax(weight)

    if args.debug:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                            sharex=True, sharey=True)
        ax1.imshow(entropy_img)
        ax1.axis('off')

        ax2.imshow(edges_img)
        ax2.axis('off')

        ax3.imshow(weight)
        ax3.axis('off')

        fig.tight_layout()
        plt.show()

    return weight


def poisson_disk_sample(n, weights, k=16):
    '''
    Performs weighted poisson disk sampling over a region.

    Algorithm based on 
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

    Weighted approach inspired by
    https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map

    Parameters
    ----------
    n : int
        The number of points to sample.
    weights : np.array
        Weights of grid to sample over. Assumes weights are normalized.
    k : int (optional)
        The number of attempts to sample an annulus before removing center.
    
    Returns
    -------
    ist :
        List of sampled points
    '''
    width = weights.shape[0]
    height = weights.shape[1]

    c = np.log10(width * height) / 2
    max_rad = min(width, height) / 4
    avg_rad = np.sqrt((height * width) / ((1 / c) * n * np.pi))
    min_rad = avg_rad / 4

    weights /= np.mean(weights)
    rads = np.clip(avg_rad / (weights + 0.01), min_rad, max_rad)
    if args.debug:
        print(f"Weights: [{np.min(weights)}, {np.max(weights)}]" \
              f" and Radii: [{rads.min()}, {rads.max()}]")

    first = (randint(width), randint(height))
    queue = [first]
    sample_points = [first]
    tree = KDTree(sample_points)

    def in_bounds(point):
        return 0 <= point[0] < width and 0 <= point[1] < height

    def has_neighbor(new_point, rads, tree):
        return len(tree.query_ball_point(new_point, rads[new_point])) > 0

    while queue and len(sample_points) < n:
        idx = randint(len(queue))
        point = queue[idx]

        success = False
        for it in range(k):
            new_point = get_point_near(point, rads, max_rad)

            if (in_bounds(new_point) and not
            has_neighbor(new_point, rads, tree)):
                queue.append(new_point)
                sample_points.append(new_point)
                tree = KDTree(sample_points)
                success = True
                break

        if not success:
            queue.pop(idx)

    print(f"Goal points: {n}")
    print(f"Generated {len(sample_points)} sample points with disk sampling.")
    print(f"{len(set(sample_points))} unique points.")
    return np.array(list(sample_points))


def get_point_near(point, rads, max_rad):
    '''
    Randomly samples an annulus near a given point using a uniform 
    distribution.

    Parameters
    ----------
    point : (int, int)
        The point to sample nearby.
    rads : np.array
        The lower bound for the random search.
    max_rad : int
        The upper bound for the random search.
    
    Returns
    -------
    (int, int) :
        The nearby point.
    '''
    rad = uniform(rads[point], max_rad)
    theta = uniform(0, 2 * np.pi)
    new_point = (point[0] + rad * np.cos(theta),
                 point[1] + rad * np.sin(theta))
    return (int(new_point[0]), int(new_point[1]))


def threshold_sample(n, weights, threshold):
    '''
    Sample the weighted points uniformly above a certain threshold.

    Parameters
    ----------
    n : int
        The number of points to sample.
    weights : np.array
        Weights of grid to sample over. Assumes weights are normalized.
    threshold : float
        The threshold to ignore points

    Returns
    -------
    list :
        The list of points to triangulate.
    '''
    candidates = np.array([idx for idx, weight in np.ndenumerate(weights) if weight >= threshold])
    if candidates.shape[0] < n:
        raise ValueError(f"Not enough candidate points for threshold {threshold}. "
                         f"Only {candidates.shape[0]} available.")

    print(f"Generated {n} sample points with threshold sampling.")
    return candidates[choice(candidates.shape[0], size=n, replace=False)]


def render_gradients(triangles, img, *, edge_angle=None):
    layers = []
    for triangle in triangles:
        for i in range(len(triangle)):
            triangle[i] = triangle[i][::-1]
        centroid_x, centroid_y = triangle_util.centroid(*triangle)
        color = img[centroid_y][centroid_x]
        converted = triangle_util.convert_triangle(triangle, color, edge_angle=edge_angle)
        for layer in converted:
            layers.append(layer)
    layers.reverse()
    return layers


def execute(app_args):
    print(f"Running {__name__} with arguments: {app_args}")

    if app_args.seed is not None:
        np.random.seed(app_args.seed)
    print(f"Using seed {np.random.get_state()[1][0]}.")

    img = imread(app_args.img)[:, :, :3]
    sample_points = generate_sample_points(img, app_args.max_points)
    triangulation = Delaunay(sample_points)
    triangles = sample_points[triangulation.simplices]

    layers = render_gradients(triangles, img, edge_angle=app_args.edge_angle)
    layer_string = ',\n'.join(layers)

    rule = f"""html, body {{
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0;
}}

.frame {{
    width: {len(img[0])}px;
    height: {len(img)}px;
    background: {layer_string};
    background-repeat: no-repeat;
}}"""

    return rule


def execute_many(app_args):
    print(f"Running {__name__} in directory mode with arguments: {app_args}")

    dir_path = app_args.img
    listing = list(os.scandir(dir_path))
    total_frames = len(listing)

    frames = []
    initial_frame = "none"

    for idx, path in enumerate(tqdm.tqdm(listing)):
        img_path = os.path.join(path)

        try:
            img = imread(img_path)[:, :, :3]
            sample_points = generate_sample_points(img, app_args.max_points)
            triangulation = Delaunay(sample_points)
            triangles = sample_points[triangulation.simplices]

            layers = render_gradients(triangles, img, edge_angle=app_args.edge_angle)
            layer_string = ',\n'.join(layers)
        except Exception as e:
            print(e, file=sys.stderr)
            layer_string = "none"

        if idx == app_args.thumbnail:
            initial_frame = layer_string

        frame = f"""
    {round(idx / total_frames * 100, 3)}% {{
        background: {layer_string};
        background-repeat: no-repeat;
    }}
        """

        frames.append(frame)

    frame_string = '\n'.join(frames)

    rule = f"""html, body {{
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0;
}}

.frame {{
    width: 100%;
    height: 100%;
    background: {initial_frame};
    background-repeat: no-repeat;
}}

.executor {{
    animation: play {total_frames / args.fps}s step-end forwards;
}}

@keyframes play {{
    {frame_string}
}}"""

    return rule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Perform delaunay triangulation 
        on a given image to create a low-polygon approximation.''')
    parser.add_argument('img', help="The image to triangulate.",
                        type=str)
    parser.add_argument('-directory', help="Use @keyframe animation.",
                        action='store_true', default=False)
    parser.add_argument('-fps', help="Frames per second.",
                        type=int, default=24)
    parser.add_argument('-sample', help="Sampling method for candidate points.",
                        type=str, default="threshold", choices=['disk', 'threshold'])
    parser.add_argument('-process', help="Pre-processing method to use.",
                        type=str, default='approx-canny', choices=['approx-canny',
                                                                   'edge-entropy'])
    parser.add_argument('-rate', help="Desired ratio of sample points to pixels.",
                        type=float, default=0.03)
    parser.add_argument('-blur', help="Blur radius for approximate canny.",
                        type=int, default=2)
    parser.add_argument('-threshold', help='Threshold for threshold sampling.',
                        type=float, default=0.02)
    parser.add_argument('-max-points', help="Max number of sample points.",
                        type=int, default=5000)
    parser.add_argument('-seed', help="Seed for random number generation.",
                        type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--time', help="Display timer for each section.",
                        action='store_true')
    parser.add_argument('-thumbnail', help="Frame number to pick thumbnail.",
                        type=int, default=None)
    parser.add_argument('-edge-angle', help="Extra angle around the cone to fill up gaps in the output.", type=float, default=0.025)
    parser.add_argument('-rounding', help="Round CSS gradients to max decimal places.", type=int, default=3)

    args = parser.parse_args()

    if args.directory:
        rule = execute_many(args)
    else:
        rule = execute(args)

    with open("styles.css", "w") as f:
        f.write(rule)
