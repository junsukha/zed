import pyzed.sl as sl
import numpy as np
import cv2
from open3d import *

path_l = 'images/rect_003_3_r5000.png'


image = cv2.imread(path_l)
print(image.shape)

image2 = cv2.imread('images/depth_map_0000.pfm')
print(image2.shape)