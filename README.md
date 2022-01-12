# opencv-single-camera-calibration
Single camera calibration using opencv library - Python code

This repository contains a simple python code for geometric calibration of single camera. Camera calibration is the process of estimating the transformation matrices from the world frame to the image plane. The calibration is performed by oserving a planar object printed with known pattern. You could use a circle board or a checkerboard. Beform using the code, be sure that your camera's focal lenght is not set to automatic otherwise the estimated parameters are wrong.
This calibration is based on Pinhole camera model, you can check the mathematical details here (https://en.wikipedia.org/wiki/Pinhole_camera_model)

# About
This repository contains the code for single camera calibration. The code have been tested on UBUNTU 20.04 (works also on UBUNTU 16.04 18.04) with Pycharm 2021.3 (Community Edition)

# Requirements
Make sure to connect you webcam (or to activate it)
Libraries to be installed on the Pycharm terminal
```
pip install opencv-python 
pip install libopencv-dev 
pip install numpy
```
# HOW TO
To perform calibration, a minimum of 3 images are needed. To obtain a good estimation of the parameters, at least 20 images must be taken, in each image the pose of the calibration grid must change
The angle variation of the calibration grid is also important to improve the estimation of focal length

One the program is executed, you can use the following keys to snap images, calibrate, reset or quit.
[space]     : take picture
[c]         : compute calibration
[r]         : reset program
[ESC]    : quit

# Additional parameters
The code provides the camera intrinsic matrice ```mtx``` and individual extrinsic matrices ```T``` of the images. Individual Reprojection and the mean reprojection errors are also provided
