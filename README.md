# One-div CSS Generator

This script aims to convert an image or a set of frames into a CSS file, packed in only a single div element, without using clip-path or canvas elements.

![image](https://github.com/khanhtranngoccva/one-div-generator/assets/61155608/9b4c0a72-a215-410e-97ba-fb5c8fbaf0e5)

## Usage

### Quick start
For images:

```shell
python execution.py IMAGE_PATH
```

For directories of frames (use ffmpeg to convert a video into frames first):

```shell
python execution.py -directory DIRECTORY_PATH [-fps FPS] [-thumbnail THUMBNAIL] 
```

- -fps: number of frames per second
- -thumbnail: the frame index that will be shown as thumbnail.

### Advanced

- -sample: {disk,threshold} Sampling method for candidate points. Default: threshold
- -process: {approx-canny,edge-entropy} Pre-processing method to use. Default: approx-canny
- -rate: Desired ratio of sample points to pixels. Default: 0.03
- -blur: Blur radius for approximate canny. Default: 2
- -threshold: Threshold for threshold sampling. Default: 0.02
- -max-points: Max number of sample points. Default: 5000
- -seed: Seed for random number generation.
- --debug: Enable debugging.
- --time: Display timer for each section.
- -thumbnail: Frame number to pick thumbnail.
- -edge-angle: Extra angle around the cone to fill up gaps in the output. Default: 0.025
- -rounding: Round CSS gradients to max decimal places. Default: 3

## Abstract steps

1. Triangulate the image with help of PyTri. Each triangle's color is based on the color of its centroid.
2. For each triangle, split it into 2 halves with a horizontal cut. This creates 2 triangles, both of which have exactly
   2 points having the same y-coordinate, which is necessary for conic-gradients to work.
3. Convert mini-triangles created from step 4 into conic gradients.

## Optimizations

1. Added extra angle parameters and enlarged bounding box to minimize gaps between triangles due to browser rounding
   problems.

## Limitations

1. Triangulation of polygons is incredibly space-intensive. A simple image as in the demo can take up thousands of
   triangles, corresponding to megabytes of data. (however, it will not crash the browser if the maximum number of
   sample points remains under 15000)
2. Details are not as well-preserved as directly using polygons.
3. Videos have very low FPS even running on as few as 300 sample points, and only work on Firefox. Google Chrome and
   Chromium rendering engines are extremely horrible at tackling one-div animations.
4. There are gaps forming between the triangles that need to be eliminated by using edge-angle, which makes the rendering of triangles less accurate.

## Credits

- This script is inspired by Junferno's CSS generator. <a href="https://github.com/kevinjycui/css-video">Visit
  kevinjycui/css-video</a>
- Triangulation of images is done using the PyTri library, which is custom-retrofitted for conversion into
  CSS. <a href="https://github.com/pmaldonado/PyTri">Visit PyTri</a>
