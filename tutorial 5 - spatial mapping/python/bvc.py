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
# from lib.utils import data_utils
workspace = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data'
take_photos = True
def transform_pose(pose, tx):
    transform_ = sl.Transform()
    transform_.set_identity()
    # Translate the tracking frame by tx along the X axis
    print(f'transform_ shape: {transform_.m.shape}')
    transform_[0, 3] = tx
    # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
    transform_inv = sl.Transform()
    transform_inv.init_matrix(transform_)
    transform_inv.inverse()
    pose = transform_inv * pose * transform_


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

        scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []}

        while i < 10:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get the pose of the left eye of the camera with reference to the world frame
                zed.get_position(zed_pose,
                                 sl.REFERENCE_FRAME.WORLD)  # zed_pose에 정보를 저장한다. #  the returned pose relates to the initial
                # position of the camera
                zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
                zed_imu = zed_sensors.get_imu_data()

                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image_l, sl.VIEW.LEFT)
                zed.retrieve_image(image_r, sl.VIEW.RIGHT)

                image_zed_l = image_l.get_data()
                image_zed_r = image_r.get_data()

                zed.retrieve_measure(depth_l, sl.MEASURE.DEPTH)

                # save image in current directory
                cv2.imwrite('images2/rect_{:03d}_3_r5000.png'.format(i+1), image_zed_l)
                cv2.imwrite('images2/rect_{:03d}_3_r5000.png'.format(i+2), image_zed_r)

                # print(f'depth_ls.shape: {depth_l.get_data().shape}')
                cv2.imwrite('images2/depth_map_{:04d}.pfm'.format(i), depth_l.get_data())
                cv2.imwrite('images2/depth_map_{:04d}.pfm'.format(i+1), depth_l.get_data())

                # Display the translation and timestamp
                py_translation = sl.Translation()
                tx = round(zed_pose.get_translation(py_translation).get()[0],
                           3)  # translation from the pose  # zed_pose 가 가지고 있는 translation 정보를 py_translation에 저장? # py_translation 없어도됨;
                ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
                tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
                print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))
                print(f'py_translation: {zed_pose.get_translation().get()}')

                py_rotation = sl.Rotation()
                zed_pose.get_rotation_matrix(py_rotation)
                r00 = round(zed_pose.get_rotation_matrix(py_rotation)[0, 0], 3)
                r01 = round(zed_pose.get_rotation_matrix(py_rotation)[0, 1], 3)
                r02 = round(zed_pose.get_rotation_matrix(py_rotation)[0, 2], 3)
                r10 = round(zed_pose.get_rotation_matrix(py_rotation)[1, 0], 3)
                r11 = round(zed_pose.get_rotation_matrix(py_rotation)[1, 1], 3)
                r12 = round(zed_pose.get_rotation_matrix(py_rotation)[1, 2], 3)
                r20 = round(zed_pose.get_rotation_matrix(py_rotation)[2, 0], 3)
                r21 = round(zed_pose.get_rotation_matrix(py_rotation)[2, 1], 3)
                r22 = round(zed_pose.get_rotation_matrix(py_rotation)[2, 2], 3)

                # print(f'py_rotation: {zed_pose.get_rotation_matrix().r}')
                # print("Rotation: {0}, {1}, {2}, Timestamp: {3}\n".format(r00, r01, r02,
                #                                                          zed_pose.timestamp.get_milliseconds()))
                # print("          {0}, {1}, {2}, Timestamp: {3}\n".format(r00, r01, r02,
                #                                                          zed_pose.timestamp.get_milliseconds()))
                # print("          {0}, {1}, {2}, Timestamp: {3}\n".format(r00, r01, r02,
                #                                                          zed_pose.timestamp.get_milliseconds()))



                '''get extrinsic matrix old version'''
                '''# print(py_rotation.get_infos())
                # py_rotation = py_rotation.r.T
                # py_translation = py_translation.get()
                # py_extrinsic = np.c_[py_rotation, py_translation]
                py_extrinsic = np.c_[zed_pose.get_rotation_matrix().r, zed_pose.get_translation().get().reshape(3,1)]
                print(f'py_extrinsic: {py_extrinsic.view()}')
    
                # add row to extrinsic to make it 4*4
                world2cam_left = np.vstack((py_extrinsic, np.array([0,0,0,1])))
                print(f'after row: {world2cam_left}')'''

                '''get extrinsic matrix new version'''
                pose_data = zed_pose.pose_data().m
                print(f'pose_data: {pose_data}')  # m is used to get numpy array :
                # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Transform.html
                print(f'rotation: {py_rotation.r.T}')
                print(f'translation: {py_translation.get()}')
                world2cam_left = np.copy(pose_data)
                # rotation part
                world2cam_left[:3, :3] = pose_data[:3, :3].T
                # translation part
                world2cam_left[:3, 3] = -pose_data[:3, 3]
                # world2cam_left[:3, 3] = - world2cam_left[:3, :3] @ pose_data[:3, 3]

                ''' new one '''
                R = zed_pose.get_rotation_matrix(sl.Rotation()).r
                t = zed_pose.get_translation(sl.Translation()).get()
                world2cam_left = np.hstack((R.T, np.dot(-R.T, t).reshape(3, -1))) # -t bc t they give me is not really
                # translation of ext matrix. also np.dot(-R.T, -t) is for inversing ext mat.
                world2cam_left = np.vstack((world2cam_left, np.array([0, 0, 0, 1])))
                print(f'new: {world2cam_left}')

                ''' 
                # https://github.com/stereolabs/zed-examples/issues/226
                R = zed_pose.get_rotation_matrix(sl.Rotation()).r.T
                t = zed_pose.get_translation(sl.Translation()).get()
                world2cam_left = np.hstack((R, np.dot(-R, t).reshape(3, -1))) # this is extrinsic
                world2cam_left = np.vstack((world2cam_left, np.array([0, 0, 0, 1])))
                # print(f'\n world2cam_left: {world2cam_left.view()}') # this should be left sensor's. Need to calculate right
                # sensor's
                '''

                ''' add left image extrinsic '''
                # scene_info['exts'].append(world2cam_left)
                scene_info['exts'].append(world2cam_left)

                print(f'world2cam_left: {world2cam_left}')

                ''' Get the distance between the right eye and the left eye '''
                translation_left_to_right_x = zed.get_camera_information().calibration_parameters.T[0] # just x coord?
                translation_left_to_right = zed.get_camera_information().calibration_parameters.T #[6.30032063 0.         0.        ]
                print(f'\n translation_left_to_right: {translation_left_to_right.view()}')


                # Get right sensor's extrinsic
                # m3_3 = np.hstack((np.zeros(shape=(3, 3)), translation_left_to_right.reshape(3,1)))
                # m4_4 = np.vstack((m3_3, np.array([0,0,0,1])))
                # world2cam_right = world2cam_left + m4_4 # wrong. this makes ext[3,3] = 2. should be 1

                '''add right image extrinsic'''
                world2cam_right = np.copy(world2cam_left)
                # world2cam_right[0, 3] = world2cam_right[0, 3] + translation_left_to_right_x
                # world2cam_right[:, 3] = world2cam_right[:, 3] - np.append(translation_left_to_right,[0])
                # world2cam_right[:, 3] = np.append(world2cam_left[:3,:3] @ world2cam_left[:3, 3] - translation_left_to_right, [0])
                # print(f'\n world2cam_right: {world2cam_right.view()}')

                cam2world_left = np.copy(world2cam_left)
                cam2world_left[:3, :3] = world2cam_left[:3, :3].T
                cam2world_left[:3, 3] =  -world2cam_left[:3, :3].T @ world2cam_left[:3, 3]
                # cam2world_left[:3, 3] = world2cam_left[:3, :3].T @ world2cam_left[:3, 3]
                # world2cam_right[:3, 3] = -( cam2world_left @ np.append(pose_data[:3, 3] + translation_left_to_right,
                #                                                        [0]) )[:3]
                hom = (cam2world_left @ np.append(translation_left_to_right, [1]))

                hom = hom[:3] # 이전 hom 이 이론상으로는 right cam pos 인데, 여기서도 rotation이 적용된값이면 그거 원상복귀
                print(f'hom: {hom}')

                right_cam_pos_world_space = hom[:3] # this is right camera position in world space
                print(right_cam_pos_world_space)

                # world2cam_right[:3, 3] = - world2cam_right[:3, :3]@right_cam_pos_world_space
                world2cam_right = np.hstack((R.T,np.dot(-R.T, right_cam_pos_world_space).reshape(3,-1)))
                world2cam_right = np.vstack((world2cam_right, np.array([0, 0, 0, 1])))

                # world2cam_right[0,3] += right_cam_pos_world_space[0]
                scene_info['exts'].append(world2cam_right)

                # left camera space 에서 +x 로 6.30이니까.   left camera space 에서 left camera pose 에다 6.40 을 한값을 다시 world space로
                # 옮긴다음에 다시 right camera space 로.


                # world2cam_right = world2cam_left
                print(f'world2cam_right: {world2cam_right}')

                # get intrinsic
                left_cam_info = zed.get_camera_information().calibration_parameters.left_cam
                right_cam_info = zed.get_camera_information().calibration_parameters.right_cam
                # stereo_transform = zed.get_camera_information().calibration_parameters.stereo_transform
                # print(f'stereo: {stereo_transform}')


                print(f'left_cam_info shape: {left_cam_info}')

                l_fx = left_cam_info.fx
                l_fy = left_cam_info.fy
                l_cx = left_cam_info.cx
                l_cy = left_cam_info.cy

                r_fx = right_cam_info.fx
                r_fy = right_cam_info.fy
                r_cx = right_cam_info.cx
                r_cy = right_cam_info.cy

                l_ixts = np.array([[l_fx,   .0, l_cx],
                                  [   .0, l_fy, l_cy],
                                  [   .0,    .0,   1]])
                r_ixts = np.array([[r_fx,   .0, r_cx],
                                  [   .0, r_fy, r_cy],
                                  [   .0,    .0,   1]])

                # print(f'\n l_ixts: \n {l_ixts.view()}')
                # print(f'\n r_ixts: \n {r_ixts.view()}')
                scene_info['ixts'].append(l_ixts)
                scene_info['ixts'].append(r_ixts)

                # Retrieve and transform the pose data into a new frame located at the center of the camera
                tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                transform_pose(zed_pose.pose_data(sl.Transform()), translation_left_to_right_x)

                # print("Translation of py_translation: Tx: {0}, Ty: {1}, Tz: {2}".format(py_translation.get()[0], py_translation.get()[1], py_translation.get()[2]))
                # Display the orientation quaternion
                # py_orientation = sl.Orientation()
                # ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
                # oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
                # oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
                # ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
                # print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

                # Display the IMU acceleratoin
                # acceleration = [0, 0, 0]
                # zed_imu.get_linear_acceleration(acceleration)
                # ax = round(acceleration[0], 3)
                # ay = round(acceleration[1], 3)
                # az = round(acceleration[2], 3)
                # print("IMU Acceleration: Ax: {0}, Ay: {1}, Az {2}\n".format(ax, ay, az))

                # Display the IMU angular velocity
                # a_velocity = [0, 0, 0]
                # zed_imu.get_angular_velocity(a_velocity)
                # vx = round(a_velocity[0], 3)
                # vy = round(a_velocity[1], 3)
                # vz = round(a_velocity[2], 3)
                # print("IMU Angular Velocity: Vx: {0}, Vy: {1}, Vz {2}\n".format(vx, vy, vz))

                # Display the IMU orientation quaternion
                # zed_imu_pose = sl.Transform()
                # ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
                # oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
                # oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
                # ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
                # print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

                i = i + 2

        write_to_file(scene_info)
        # Close the camera
        zed.close()

        exit()
        l_img_num = 1
        r_img_num = 3
        l_dpt_num = l_img_num - 1
        r_dpt_num = r_img_num - 1
        l_cam_num = l_dpt_num
        r_cam_num = r_dpt_num
        # path_l = 'images2/rect_001_3_r5000.png'
        path_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
                 r'\tutorial 5 - spatial mapping\python\images2\rect_{:03d}_3_r5000.png'.format(l_img_num)
        # path_r = 'images2/rect_002_3_r5000.png'
        path_r = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
                 r'\tutorial 5 - spatial mapping\python\images2\rect_{:03d}_3_r5000.png'.format(r_img_num)
        # depth_l = 'images2/depth_map_0000.pfm' # depth0 is depth of image0 and image1. Likewise depth2 for image2 and image3
        depth_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
                  r'\tutorial 5 - spatial mapping\python\images2\depth_map_{:04d}.pfm'.format(l_dpt_num)


        l = cv2.imread(path_l)
        r = cv2.imread(path_r)
        d = cv2.imread(depth_l)


        height =    720
        width =    1280

        # create open3d intrinsic object
        l_fx = scene_info['ixts'][0][0,0]
        l_fy = scene_info['ixts'][0][1,1]
        l_cx = scene_info['ixts'][0][0,2]
        l_cy = scene_info['ixts'][0][1,2]

        r_fx = scene_info['ixts'][1][0,0]
        r_fy = scene_info['ixts'][1][1,1]
        r_cx = scene_info['ixts'][1][0,2]
        r_cy = scene_info['ixts'][1][1,2]

        l_intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height, l_fx, l_fy, l_cx, l_cy)
        r_intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height, r_fx, r_fy, r_cx, r_cy)

        rgb =  open3d.geometry.Image(l)
        depth = open3d.geometry.Image(d)


        # rgbd = open3d.geometry.Image(rgb, depth_l)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd_l = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, l_intrinsic, np.array(scene_info['exts'][l_cam_num]))


        rgb =  open3d.geometry.Image(r)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd_r = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, r_intrinsic, np.array(scene_info['exts'][r_cam_num]))

        print(f'pcd_l.shape: {pcd_l}')
        pcd_l.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # open3d.visualization.draw_geometries([pcd_l])

        pcd_r.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        vis = open3d.visualization.Visualizer()
        # vis.create_window()
        #
        #
        # vis.add_geometry(pcd_l)
        # vis.add_geometry(pcd_r)

        # for i in range(icp_iteration):
        #     # now modify the points of your geometry
        #     # you can use whatever method suits you best, this is just an example
        #     geometry.points = pcd_list[i].points
        #     vis.update_geometry(geometry)
        #     vis.poll_events()
        #     vis.update_renderer()



        pcd_combined = np.concatenate((np.asarray(pcd_l.points),np.asarray(pcd_r.points)), axis=0)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_combined)
        open3d.visualization.draw_geometries([pcd_l, pcd_r])

        '''# print(f'pcd_combined.length: {len(pcd_combined)}')
        pcds = open3d.geometry.PointCloud()
    
        #
        # pcds.points = open3d.utility.Vector3dVector(pcd_combined)
        print(f'what is this: {len(pcd_l.points)}')
    
        # pcd_combined = []
        for point in range(len(pcd_l.points)):
            # pcd_combined.append(pcd_l.points[point])
            pcds.points.extend(pcd_l.points[point])
        for point in range(len(pcd_r.points)):
            # pcd_combined.append(pcd_r.points[point])
            pcds.points.extend(pcd_r.points[point])
        # pcds = pcd_r+pcd_l
    
        # pcds.points.extend(pcd_r.points)
        # pcds.points.extend(pcd_l.points)
        # pcds.points.extend(pcd_l)
        # pcds.points.extend(pcd_r)
    
        # pcds.points = open3d.utility.Vector3dVector(pcd_combined)
        print(f'pcd_combined.length: {len(pcd_l.points.shape)}')
        pcd_l.points.extend(pcd_r.points)
        print(f'pcd_combined.length: {len(pcd_l.points)}')
        open3d.visualization.draw_geometries([pcd_l])
        # open3d.visualization.draw_geometries(pcd_r)'''


        # write_to_file(scene_info)
        # # Close the camera
        # zed.close()

def write_to_file(scene_info):
    for i in range(len(scene_info['ixts'])):
        f = open( './cam/{:08d}_cam.txt'.format(i), 'w+')

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
