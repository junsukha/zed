import open3d.geometry
import pyzed.sl as sl
import numpy as np
import cv2
from open3d import *


# from lib.utils import data_utils
# workspace = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data'

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
    # # Create a Camera object
    # zed = sl.Camera()
    #
    # # Create a InitParameters object and set configuration parameters
    # init_params = sl.InitParameters()
    # init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
    # # Use a right-handed Y-up coordinate system
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # init_params.coordinate_units = sl.UNIT.CENTIMETER  # Set units in meters
    #
    # # Open the camera
    # err = zed.open(init_params)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     exit(1)
    #
    # # Enable positional tracking with default parameters
    # py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    # tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    # err = zed.enable_positional_tracking(tracking_parameters)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     exit(1)
    #
    # # Track the camera position during 1000 frames
    # i = 0
    # zed_pose = sl.Pose()  # the pose containing the position of the camera and other information (timestamp, confidence)
    #
    # zed_sensors = sl.SensorsData()
    # runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
    #
    # # images will be saved here
    # image_l = sl.Mat(zed.get_camera_information().camera_resolution.width,
    #                  zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    # image_r = sl.Mat(zed.get_camera_information().camera_resolution.width,
    #                  zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    #
    # depth_l = sl.Mat()

    path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
             r'\rect_001_3_r5000.png'
    # path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
    #          r'\rect_001_3_r5000.png'

    path_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
             r'/rect_002_3_r5000.png'
    # path_r = path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
    #          r'\rect_002_3_r5000.png'

    depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Depths\scan114\depth_map_0000.pfm'
    # depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Depths\scan114\depth_map_0000.pfm'

    l = cv2.imread(path_l)
    r = cv2.imread(path_r)
    d = cv2.imread(depth_l)

    print(f'l.shape: {l.shape}')
    print(f'd.shape: {d.shape}')

    height =   720 # 512
    width =  1280 # 640

    l_fx = 670.2325
    l_fy = 670.2325
    l_cx = 645.2366
    l_cy = 350.6550

    r_fx = 670.2325
    r_fy = 670.2325
    r_cx = 645.2366
    r_cy = 350.6550

    l_intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height, l_fx, l_fy, l_cx, l_cy)
    r_intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height, r_fx, r_fy, r_cx, r_cy)

    '''  l_ext = np.array([[1.0000, 0.0000, 0.0000, 0.0000],
                      [0.0000, 1.0000, 0.0000, 0.0000],
                      [0.0000, 0.0000, 1.0000, 0.0000],
                      [0.0000, 0.0000, 0.0000, 1.0000]])

    r_ext = np.array([[1.0000, 0.0000, 0.0000, 0.0006],#[1.0000, 0.0000, 0.0000, 6.3003],
                      [0.0000, 1.0000, 0.0000, 0.0000],
                      [0.0000, 0.0000, 1.0000, 0.0000],
                      [0.0000, 0.0000, 0.0000, 1.0000]])'''

    l_ext = np.array([[0.9997 ,0.0123,-0.0223, -0.6748],
                      [-0.0119, 0.9998, 0.0163, -0.2631],
                      [0.0225, -0.0160, 0.9996, 0.0991],
                      [0.0000, 0.0000, 0.0000, 1.0000]])

    r_ext = np.array([[0.9997 ,0.0123,-0.0223, -0.6748+0.06], #5.6255
                      [-0.0119, 0.9998, 0.0163, -0.2631],
                      [0.0225, -0.0160, 0.9996, 0.0991],
                      [0.0000, 0.0000, 0.0000, 1.0000]])

    rgb = open3d.geometry.Image(l)
    depth = open3d.geometry.Image(d)

    # rgbd = open3d.geometry.Image(rgb, depth_l)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    pcd_l = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, l_intrinsic, l_ext) # l_ext parameter
    # pcd_l.points



    rgb = open3d.geometry.Image(r)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    pcd_r = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, r_intrinsic)  # r_ext parameter # returns
    # open3d.geometry.PointCloud


    print(f'pcd_l.shape: {pcd_l}') # PointCloud with 327680 points.
    print(f'pcd_l.points.shhape: {pcd_l.points}') #  std::vector<Eigen::Vector3d> with 327680 elements.
    # Flip it, otherwise the pointcloud will be upside down
    pcd_l.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # open3d.visualization.draw_geometries([pcd_l])

    print(f'pcd_r.shape: {pcd_r}')
    # Flip it, otherwise the pointcloud will be upside down
    pcd_r.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #----------------------------
    # pcd_l.points.extend(pcd_r.points)
    # print(f'extended pcd_l.shape: {pcd_l}')
    # ----------------------------------------------
    '''pcd_l = open3d.geometry.PointCloud()
    # pcd =  open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, l_intrinsic, l_ext)

    pcd_l_new = pcd_l.create_from_rgbd_image(rgbd, l_intrinsic, l_ext)

    print(f'pcd_l length: {len(pcd_l.points)}')
    pcd_r = open3d.geometry.PointCloud()
    pcd_r.create_from_rgbd_image(rgbd, r_intrinsic, r_ext)'''


    '''
    This gives rainblow output
    
    pcd_total = open3d.geometry.PointCloud()
    pcd_total.points = pcd_r.points
    open3d.visualization.draw_geometries([pcd_total])
    '''

    '''pcd_total = open3d.geometry.PointCloud()
    print(type(pcd_r.points)) # open3d.cpu.pybind.utility.Vector3dVector
    # points = []
    points = np.asarray(pcd_r.points)
    total = np.append(points, np.asarray(pcd_l.points), axis=0)
    print(type(total))
    print(total.shape)

    list = []
    points2 = np.asarray(pcd_l.points)
    for i in range(len(points)):
        list.extend(points[i])
        list.extend(points2[i])
    print(len(list))
    list_to_array = np.array(list)
    pcd_total.points = open3d.utility.Vector3dVector(list_to_array)'''

    # pcd_total.points = open3d.utility.Vector3dVector(total)

    # pcd_l_pointcloud = open3d.geometry.PointCloud()
    # pcd_r_pointcloud = open3d.geometry.PointCloud()
    # pcd_l_pointcloud.points = open3d.utility.Vector3dVector(np.array(pcd_l.points))
    # pcd_r_pointcloud.points = open3d.utility.Vector3dVector(np.array(pcd_r.points))

    # pcd_concat_cloud = open3d.geometry.PointCloud()
    # pcd_concat_cloud.points = np.concatenate((np.asarray(pcd_l.points), np.asarray(pcd_r.points)), axis=0)

    # open3d.visualization.draw_geometries([pcd_r, pcd_l])
    # open3d.visualization.draw_geometries([pcd_l, pcd_r])
    open3d.visualization.draw_geometries([pcd_l])


    '''pcd_combined = np.concatenate((np.asarray(pcd_l.points), np.asarray(pcd_r.points)), axis=0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_combined)

    print(f'pcd_combined.shape: {pcd_combined.shape}')
    # open3d.visualization.draw_geometries([pcd_l])
    open3d.visualization.draw_geometries([pcd])
    '''
    # pcds = []
    # pcds.append(pcd_l)
    # pcds.append(pcd_r)
    # open3d.visualization.draw_geometries([pcds])

    # pcd_combined = open3d.geometry.PointCloud()
    # pcd_combined += pcd_l.points
    # pcd_combined += pcd_r.points
    # open3d.visualization.draw_geometries([pcd_combined])

def write_to_file(scene_info):
    for i in range(len(scene_info['ixts'])):
        f = open('{:08d}_cam.txt'.format(i), 'w+')

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
