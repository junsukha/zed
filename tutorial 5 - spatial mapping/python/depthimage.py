import open3d.geometry
import pyzed.sl as sl
import numpy as np
import cv2
from open3d import *
import open3d as o3d
import math

# Set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)

image = sl.Mat()
depth_map = sl.Mat()
runtime_parameters = sl.RuntimeParameters()


# Create a ZED camera
zed = sl.Camera()

# Create configuration parameters
# init_params = sl.InitParameters()
init_params.sdk_verbose = True # Enable the verbose mode
# init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)


# Open the camera
err = zed.open(init_params)
if (err!=sl.ERROR_CODE.SUCCESS):
  exit(-1)


if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
  # A new image and depth is available if grab() returns SUCCESS
  zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve left image
  zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth


# Capture 50 images and depth, then stop
i = 0
image = sl.Mat()
depth = sl.Mat()
while (i < 50) :
    # Grab an image
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Retrieve depth Mat. Depth is aligned on the left image
        i = i + 1


# Get and print distance value in mm at the center of the image
# We measure the distance camera - object using Euclidean distance
x = image.get_width() / 2
y = image.get_height() / 2
point_cloud_value = point_cloud.get_value(x, y)

distance = math.sqrt(point_cloud_value[0]*point_cloud_value[0] + point_cloud_value[1]*point_cloud_value[1] + point_cloud_value[2]*point_cloud_value[2])
print("Distance to Camera at (", x, y, "): ", distance, "mm")

# Close the camera
zed.close()
