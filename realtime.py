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
from read_pfm import read_pfm

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
    if take_photos:
        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
        # Use a right-handed Y-up coordinate system
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.coordinate_units = sl.UNIT.CENTIMETER  # Set units in meters

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

        # images will be saved here
        image_l = sl.Mat(zed.get_camera_information().camera_resolution.width,
                         zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
        image_r = sl.Mat(zed.get_camera_information().camera_resolution.width,
                         zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

        depth_l = sl.Mat()

        # create visualizer and window.
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=480*2, width=640*2)
        pcd_l = o3d.geometry.PointCloud()
        # points = np.random.rand(10, 3)
        # pcd_l.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd_l)
        i = 0
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get the pose of the left eye of the camera with reference to the world frame
                zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)  # zed_pose에 정보를 저장한다. #  the returned pose relates to the initial
                # position of the camera
                zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)

                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image_l, sl.VIEW.LEFT)
                zed.retrieve_image(image_r, sl.VIEW.RIGHT)

                image_zed_l = image_l.get_data()
                image_zed_r = image_r.get_data()
                # print(f'image_zed_l: {image_zed_l[:5, :5]}')
                # print(f'image_zed_l.shape: {image_zed_l.shape}')


                # get depth map
                zed.retrieve_measure(depth_l, sl.MEASURE.DEPTH)
                cv2.imwrite('depth_map.pfm', depth_l.get_data())

                # get rotation and translation relative to world space
                R = zed_pose.get_rotation_matrix(sl.Rotation()).r
                t = zed_pose.get_translation(sl.Translation()).get()

                world2cam_left = np.hstack((R.T, np.dot(-R.T, t).reshape(3, -1)))  # -t bc t they give me is not really
                                                                                 # translation of ext matrix. also np.dot(-R.T, -t) is for inversing ext mat.
                world2cam_left = np.vstack((world2cam_left, np.array([0, 0, 0, 1])))

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

                h = image_zed_l.shape[0]  # the size i cropped to. the size i want the output image to be
                w = image_zed_l.shape[1]

                # generate meshgrid
                xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
                xs = xs.reshape(-1, 1)
                ys = ys.reshape(-1, 1)
                uv2 = np.column_stack((xs, ys)).reshape(h, w, 2)

                # make 3d coordinate homogeneous
                uv3 = np.concatenate((uv2, np.ones(shape=(uv2.shape[0], uv2.shape[1], 1))), axis=-1)
                uv3 = uv3.reshape(h * w, -1)  # (h*w, 3)

                # change d shape
                d_l, scale = read_pfm('depth_map.pfm')
                d_l = d_l.reshape(-1, 1)
                # d_l = depth_l.get_data().reshape(-1,1)
                d_r = d_l.reshape(-1, 1)
                # d_r = depth_l.get_data().reshape(-1,1)


                # multiply depth
                uv3_l = uv3  # * d_l
                uv3_r = uv3  # * d_r

                # pixel space to image to camera space
                i2c_l = np.sum(uv3_l[..., None, :] * np.linalg.inv(l_ixts)[None, ...], axis=-1)
                i2c_r = np.sum(uv3_r[..., None, :] * np.linalg.inv(r_ixts)[None, ...], axis=-1)

                # maybe my depth wrong?
                i2c_l *= d_l
                i2c_r *= d_r

                # make homogeneous coordinate
                i2c_l = np.hstack((i2c_l, np.ones((i2c_l.shape[0], 1))))  # (h*w, 4)
                i2c_r = np.hstack((i2c_r, np.ones((i2c_r.shape[0], 1))))  # (h*w, 4)

                # convert camera space into world space
                cam2world_left = np.identity(4)
                cam2world_left[:3, :3] = (world2cam_left[:3, :3]).T
                cam2world_left[:3, 3] = -world2cam_left[:3, :3] @ world2cam_left[:3, 3]
                print(f'cam2world_left: {cam2world_left}')

                c2w_l = np.sum(i2c_l[..., None, :] * cam2world_left[None, ...], axis=-1)
                c2w_r = np.sum(i2c_r[..., None, :] * cam2world_left[None, ...], axis=-1)

                # homogenous to normal coordinate
                c2w_l = c2w_l[:, :3]  # .reshape(h,w,3).astype(float)
                c2w_r = c2w_r[:, :3]

                # make point cloud instances
                # pcd_l = o3d.geometry.PointCloud()
                pcd_r = o3d.geometry.PointCloud()

                # convert np array to vector of 3d vectors
                pcd_l.points = o3d.utility.Vector3dVector(c2w_l)
                # pcd_l.points.extend(np.random.rand(10, 3))
                # pcd_r.points = o3d.utility.Vector3dVector(c2w_r)



                # change image shape

                l = image_zed_l.reshape(-1, 4)[:, :3]  # (327680, 3)
                r = image_zed_r.reshape(-1, 4)[:, :3]

                # add color
                pcd_l.colors = o3d.utility.Vector3dVector(l.astype(np.float64) / 255.0)
                # pcd_r.colors = o3d.utility.Vector3dVector(r.astype(np.float64) / 255.0)




                # transform point clouds. otherwise generate wrong point clouds
                # pcd_l.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                # pcd_r.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                print(f'Generating point cloud...')

                # if i == 0:
                #     vis.add_geometry(pcd_l)
                # else:

                vis.update_geometry(pcd_l)
                vis.poll_events()
                vis.update_renderer()

                # o3d.visualization.draw_geometries([pcd_l])

                input("Enter please")
                # vis.remove_geometry(pcd_l)
                i += 1
        # write_to_file(scene_info)
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
