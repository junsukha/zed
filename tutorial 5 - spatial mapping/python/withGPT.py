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
import open3d.geometry
import pyzed.sl as sl
import numpy as np
import cv2
from open3d import *
import open3d as o3d
import math

# from lib.utils import data_utils
workspace = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data'
take_photos = True


def transform_pose(pose, tx):
    transform_ = sl.Transform()
    transform_.set_identity()
    # Translate the tracking frame by tx along the X axis
    # print(f'transform_ shape: {transform_.m.shape}')
    transform_[0, 3] = tx
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
        init_params.coordinate_units = sl.UNIT.CENTIMETER  # Set units in meters
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
        i = 0
        zed_pose = sl.Pose()  # the pose containing the position of the camera and other information (timestamp, confidence)

        zed_sensors = sl.SensorsData()
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
        # runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD

        # images will be saved here
        image_l = sl.Mat(zed.get_camera_information().camera_resolution.width,
                         zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
        image_r = sl.Mat(zed.get_camera_information().camera_resolution.width,
                         zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

        depth_l = sl.Mat()

        scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []}

        while i<10:
        # while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get the pose of the left eye of the camera with reference to the world frame
                zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)  # zed_pose에 정보를 저장한다. #  the returned pose relates to the initial
                # position of the camera
                zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
                zed_imu = zed_sensors.get_imu_data()

                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image_l, sl.VIEW.LEFT)
                zed.retrieve_image(image_r, sl.VIEW.RIGHT)

                image_zed_l = image_l.get_data()
                image_zed_r = image_r.get_data()

                # get depth map
                zed.retrieve_measure(depth_l, sl.MEASURE.DEPTH)



                ''' left camera ext '''
                R = zed_pose.get_rotation_matrix(sl.Rotation()).r
                t = zed_pose.get_translation(sl.Translation()).get()
                # print(f'R: \n{R}')
                print(f't: \n{t}')

                world2cam_left = np.hstack((R.T, np.dot(-R.T, t).reshape(3, -1)))  # -t bc t they give me is not really
                # translation of ext matrix. also np.dot(-R.T, -t) is for inversing ext mat.
                world2cam_left = np.vstack((world2cam_left, np.array([0, 0, 0, 1])))

                scene_info['exts'].append(world2cam_left)

                ''' right camera ext 1'''
                # translation_left_to_right = zed.get_camera_information().calibration_parameters.T[0]
                # # Retrieve and transform the pose data into a new frame located at the right camera
                # # tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                # # print(f'translation_left_to_right: {translation_left_to_right}')
                #
                # new_pose = transform_pose(zed_pose.pose_data(), translation_left_to_right)
                #
                # R = new_pose.m[:3, :3]
                # t = new_pose.m[:3, 3]
                #
                # world2cam_right = np.hstack((R.T, np.dot(-R.T, t).reshape(3, -1)))  # -t bc t they give me is not really
                # # translation of ext matrix. also np.dot(-R.T, -t) is for inversing ext mat.
                # world2cam_right = np.vstack((world2cam_right, np.array([0, 0, 0, 1])))

                ''' right camera ext 2'''
                world2cam_right = np.identity(4)
                # world2cam_right[0, 3] = world2cam_right[0, 3] + translation_left_to_right_x
                # world2cam_right[:, 3] = world2cam_right[:, 3] - np.append(translation_left_to_right,[0])
                # world2cam_right[:, 3] = np.append(world2cam_left[:3,:3] @ world2cam_left[:3, 3] - translation_left_to_right, [0])
                # print(f'\n world2cam_right: {world2cam_right.view()}')

                cam2world_left = np.identity(4)
                cam2world_left[:3, :3] = world2cam_left[:3, :3].T
                cam2world_left[:3, 3] = -world2cam_left[:3, :3].T @ world2cam_left[:3, 3]

                translation_left_to_right = zed.get_camera_information().calibration_parameters.T
                right_cam_pos_world = cam2world_left @ np.append(translation_left_to_right, [1])
                right_cam_pos_world = right_cam_pos_world[:3]  # this is right camera position in world space

                world2cam_right = np.hstack((R.T, np.dot(-R.T, right_cam_pos_world).reshape(3, -1)))
                world2cam_right = np.vstack((world2cam_right, np.array([0, 0, 0, 1])))

                scene_info['exts'].append(world2cam_right)

                # get intrinsic
                left_cam_info = zed.get_camera_information().calibration_parameters.left_cam
                right_cam_info = zed.get_camera_information().calibration_parameters.right_cam

                l_fx = left_cam_info.fx
                l_fy = left_cam_info.fy
                l_cx = left_cam_info.cx
                l_cy = left_cam_info.cy

                r_fx = right_cam_info.fx
                r_fy = right_cam_info.fy
                r_cx = right_cam_info.cx
                r_cy = right_cam_info.cy

                l_ixts = np.array([[l_fx, .0, l_cx],
                                   [.0, l_fy, l_cy],
                                   [.0, .0, 1]])
                r_ixts = np.array([[r_fx, .0, r_cx],
                                   [.0, r_fy, r_cy],
                                   [.0, .0, 1]])


                scene_info['ixts'].append(l_ixts)
                scene_info['ixts'].append(r_ixts)

                # save image in current directory
                cv2.imwrite('closeimages/rect_{:03d}_3_r5000.png'.format(i + 1), image_zed_l)
                cv2.imwrite('closeimages/rect_{:03d}_3_r5000.png'.format(i + 2), image_zed_r)

                # print(f'depth_ls.shape: {depth_l.get_data().shape}')
                cv2.imwrite('closeimages/depth_map_{:04d}.pfm'.format(i), depth_l.get_data())
                cv2.imwrite('closeimages/depth_map_{:04d}.pfm'.format(i + 1), depth_l.get_data())

                i = i + 2

                # input("Press Enter to continue...")
                input("Enter..")
        write_to_file(scene_info)
        # Close the camera
        zed.close()

def write_to_file(scene_info):
    for i in range(len(scene_info['ixts'])):
        f = open('./cam/{:08d}_cam.txt'.format(i), 'w+')

        # write extrinsic
        f.write("extrinsic\n")
        mats = scene_info['exts']
        mat = np.matrix(mats[i])
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')

        # write intrinsic
        f.write("\nintrinsic\n")
        mats = scene_info['ixts']
        mat = np.matrix(mats[i])
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')

        depth = 0.0
        f.write('\n%d %d' % (depth, depth))

        f.close()


if __name__ == "__main__":
    main()
