# Morphology project - Vein segmentation

## Prerequisites

This program has been written in python3 which should be installed on your computer.

### Dependencies

Moreover, the following packages are also necessary:

- numpy
- matplotlib
- pillow
- scikit-image
- scipy

### Images

Please add a images_IOSTAR folder at the root of the folder with the images number 1, 2, 3, 8, 21, 26, 28, 32, 37, 48 and their annotations.

## The contents of this folder

This folder contains two utility files, [utils](utils.py) and [morphology_utils](morphology_utils.py). The main module is [segmentation](segmentation.py). Old optimization functions are in the [optimizers](optimizers.py) module.

### [utils](utils.py)

[utils](utils.py) contains 2 functions and a class:

- load_image: load an image with its id
- evaluate: evaluate the quality of the segmentation
- lineArrayGenerator: Objects which creates linear structures to cross the plane

### [morphology_utils](morphology_utils.py)

[morphology_utils](morphology_utils.py) contains 3 functions:

- algebraic_opening: performs an algebraic opening of an image
- conjugate_tophats: performs a sum of conjugate top-hats of an image
- erosion_reconstruction: performs an algebraic erosion of an image followed by a reconstruction

### [segmentation](segmentation.py)

The main script, [segmentation](segmentation.py), contains the segmentation function and the optimization function as well as variables for user interactions.

## How to use this script

To use this script, use the boolean and float variables of script_hysteresis.py, then launch this script **from the inside of the folder** (since it uses relative paths). They start on line 190.

if `do_optimize` is True, you will try to find the best parameters

if `mean` is True, you will compute the mean metrics for the defined set of parameters

if `exemple` is True, you will compute the segmentation for the image img_nb

if `summary` is True, you will see the segmentation and the ground truth only if exemple is True

if `watch` is True, you will see the different steps of the segmentation only if exemple is True

if `do_map` is True, you will see the false positives and negatives maps only if exemple is True

Constraints:

- 0 < img_nb < 11
