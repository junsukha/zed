import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pyzed.sl as sl
import cv2
from read_pfm import read_pfm
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
# Use a right-handed Y-up coordinate system
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
init_params.coordinate_units = sl.UNIT.METER  # Set units in meters
init_params.depth_mode = sl.DEPTH_MODE.ULTRA

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Enable positional tracking with default parameters
py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
err = zed.enable_positional_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Track the camera position during 1000 frames
zed_pose = sl.Pose()  # the pose containing the position of the camera and other information (timestamp, confidence)

zed_sensors = sl.SensorsData()
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

# images will be saved here
image_l = sl.Mat(zed.get_camera_information().camera_resolution.width,
                 zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
image_r = sl.Mat(zed.get_camera_information().camera_resolution.width,
                 zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

depth_l = sl.Mat()



if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # Get the pose of the left eye of the camera with reference to the world frame
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)  # zed_pose에 정보를 저장한다. #  the returned pose relates to the initial
    # position of the camera
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)

    # A new image is available if grab() returns SUCCESS
    zed.retrieve_image(image_l, sl.VIEW.LEFT)
    zed.retrieve_image(image_r, sl.VIEW.RIGHT)

    # get depth map
    zed.retrieve_measure(depth_l, sl.MEASURE.DEPTH)
    # zed.retrieve_image(depth_l, sl.VIEW.DEPTH)

    # save image in current directory
    cv2.imwrite('image1.png', image_l.get_data())
    cv2.imwrite('image2.png', image_r.get_data())

    print(f'depth_ls.shape: {depth_l.get_data().shape}')

    cv2.imwrite('depth_map.pfm', depth_l.get_data())

zed.close()

# Read depth image:
# depth_image = iio.imread('depth_map.pfm')
depth_image, scale = read_pfm('depth_map.pfm')
print(f'Max value: {np.max(depth_image)}')

# print properties:
print(f"Image resolution: {depth_image.shape}")
print(f"Data type: {depth_image.dtype}")
print(f"Min value: {np.min(depth_image)}")
print(f"Max value: {np.max(depth_image)}")

depth_instensity = np.array(256 * depth_image / np.max(depth_image),
                            dtype=np.uint8)
iio.imwrite('grayscale.png', depth_image)



