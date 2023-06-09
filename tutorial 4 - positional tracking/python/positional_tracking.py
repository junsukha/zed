########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import numpy as np
import cv2
import math
def transform_pose(pose, tx) :
    transform_ = sl.Transform()
    transform_.set_identity()
    # Translate the tracking frame by tx along the X axis
    # print(f'transform_ shape: {transform_.m.shape}')
    transform_[0,3] = tx
    # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
    transform_inv = sl.Transform()
    transform_inv.init_matrix(transform_)
    transform_inv.inverse()
    new_pose = transform_inv * pose * transform_
    return new_pose


def main():
    # Create a Camera object
    zed = sl.Camera()



    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
    # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.coordinate_units = sl.UNIT.METER  # Set units in meters

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
    i = 0
    zed_pose = sl.Pose() # the pose containing the position of the camera and other information (timestamp, confidence)

    zed_sensors = sl.SensorsData()
    runtime_parameters = sl.RuntimeParameters()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Get the pose of the left eye of the camera with reference to the world frame
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) # zed_pose에 정보를 저장한다. #  the returned pose relates to the initial position of the camera
            zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
            zed_imu = zed_sensors.get_imu_data()

            # print(f'Translation left pose data: {zed_pose.pose_data()}')
            # Display the translation and timestamp
            py_translation = sl.Translation()
            tx = round(zed_pose.get_translation(py_translation).get()[0], 3) #  translation from the pose  # zed_pose 가 가지고 있는 translation 정보를 py_translation에 저장? # py_translation 없어도됨;
            ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
            tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
            # print("Translation left: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))
            # print(f'angles: {zed_pose.get_euler_angles(radian=False)}')
            pose_data = zed_pose.pose_data()
            print(f'position: {pose_data.m[:3, 3]}')
            phi = math.atan2(pose_data[1,0], pose_data[0,0])
            # print(f'rotation about z axis: {phi * 180 / np.pi}')

            # theta = atan2(-R31, sqrt(R32 ^ 2 + R33 ^ 2))
            theta = math.atan2(-pose_data[2,0], math.sqrt(pose_data[2,1]**2 + pose_data[2,2]**2))
            # print(f'rotation about y axis: {theta * 180 / np.pi}')
            ''' right poses '''

            # Get the distance between the right eye and the left eye
            translation_left_to_right = zed.get_camera_information().calibration_parameters.T[0]
            # Retrieve and transform the pose data into a new frame located at the right camera
            # tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            # print(f'translation_left_to_right: {translation_left_to_right}')

            new_pose = transform_pose(zed_pose.pose_data(), translation_left_to_right)
            # print(f'Translation right: {new_pose}')

            # world2cam_right[0,3] += right_cam_pos_world_space[0]
            # scene_info['exts'].append(world2cam_right)
            # tx = round(new_pose.get_translation().get()[0], 3)  # translation from the pose  # zed_pose 가 가지고 있는 translation 정보를 py_translation에 저장? # py_translation 없어도됨;
            # ty = round(new_pose.get_translation().get()[1], 3)
            # tz = round(new_pose.get_translation().get()[2], 3)
            # print("Translation right: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

            # R = zed_pose_right.get_rotation_matrix(sl.Rotation()).r
            # t = zed_pose_right.get_translation(sl.Translation()).get()
            # print("Translation right: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(t[0], t[1], t[2], zed_pose.timestamp.get_milliseconds()))

            ''' right poses ends'''

            py_rotation = sl.Rotation()
            r00 = round(zed_pose.get_rotation_matrix(py_rotation)[0, 0], 3) 
            r01 = round(zed_pose.get_rotation_matrix(py_rotation)[0, 1], 3)
            r02 = round(zed_pose.get_rotation_matrix(py_rotation)[0, 2], 3)
            r10 = round(zed_pose.get_rotation_matrix(py_rotation)[1, 0], 3) 
            r11 = round(zed_pose.get_rotation_matrix(py_rotation)[1, 1], 3)
            r12 = round(zed_pose.get_rotation_matrix(py_rotation)[1, 2], 3)
            r20 = round(zed_pose.get_rotation_matrix(py_rotation)[2, 0], 3) 
            r21 = round(zed_pose.get_rotation_matrix(py_rotation)[2, 1], 3)
            r22 = round(zed_pose.get_rotation_matrix(py_rotation)[2, 2], 3)
            # print("Rotation: {0}, {1}, {2}, Timestamp: {3}\n".format(r00, r01, r02, zed_pose.timestamp.get_milliseconds()))
            # print("          {0}, {1}, {2}, Timestamp: {3}\n".format(r00, r01, r02, zed_pose.timestamp.get_milliseconds()))
            # print("          {0}, {1}, {2}, Timestamp: {3}\n".format(r00, r01, r02, zed_pose.timestamp.get_milliseconds()))

            # get extrinsic matrix
            # print(py_rotation.get_infos())
            py_rotation = py_rotation.r
            py_translation = py_translation.get()
            py_extrinsic = np.c_[py_rotation, py_translation]
            # print(f'py_extrinsic.shape: {py_extrinsic.shape}')

            # Get the distance between the center of the camera and the left eye
            # translation_left_to_center = zed.get_camera_information().calibration_parameters.T[0]

            # Get the distance between the right eye and the left eye
            translation_left_to_right = zed.get_camera_information().calibration_parameters.T[0]
            # print(f'translation_left_to_right: {translation_left_to_right}')

            # Retrieve and transform the pose data into a new frame located at the center of the camera
            # tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            # transform_pose(zed_pose.pose_data(sl.Transform()), translation_left_to_right)


            # print("Translation of py_translation: Tx: {0}, Ty: {1}, Tz: {2}".format(py_translation.get()[0], py_translation.get()[1], py_translation.get()[2]))
            # Display the orientation quaternion
            py_orientation = sl.Orientation()
            ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
            oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
            oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
            ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
            # print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
            
            #Display the IMU acceleratoin
            acceleration = [0,0,0]
            zed_imu.get_linear_acceleration(acceleration)
            ax = round(acceleration[0], 3)
            ay = round(acceleration[1], 3)
            az = round(acceleration[2], 3)
            # print("IMU Acceleration: Ax: {0}, Ay: {1}, Az {2}\n".format(ax, ay, az))
            
            #Display the IMU angular velocity
            a_velocity = [0,0,0]
            zed_imu.get_angular_velocity(a_velocity)
            vx = round(a_velocity[0], 3)
            vy = round(a_velocity[1], 3)
            vz = round(a_velocity[2], 3)
            # print("IMU Angular Velocity: Vx: {0}, Vy: {1}, Vz {2}\n".format(vx, vy, vz))

            # Display the IMU orientation quaternion
            zed_imu_pose = sl.Transform()
            ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
            oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
            oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
            ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
            # print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

            i = i + 1
            input("Press Enter to continue...")
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
