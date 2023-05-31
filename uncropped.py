import open3d as o3d
import open3d.geometry
import pyzed.sl as sl
import numpy as np
import cv2
from open3d import *
from utils import *
import os
# from lib.utils import data_utils
# workspace = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data'

def main():
    '''
    Insights: Use original image (not cropped one). But used cropped shape for the output width and height. Also
    update ixt matrices accordingly. When using color for point cloud variables, use cropped images.
    즉 input 과 output 의 shape 을 내가 직접 고정할수 있어야함. input 은 original image, output은 cropped size로. 그래야 ixt matrices 를
    바꿨을때 meshgrid 를 output shape에 맞게한 경우, crop 효과가 생김.
    '''
    l_img_num = 5
    r_img_num = 9
    l_dpt_num = l_img_num - 1
    r_dpt_num = r_img_num - 1
    l_cam_num = l_dpt_num
    r_cam_num = r_dpt_num
    '''
    Cropped images
    '''
    # path_l_crop = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
    #          r'\rect_003_3_r5000.png'
    # path_r_crop = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
    #          r'/rect_004_3_r5000.png'
    # depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Depths\scan114\depth_map_0002.pfm'

    '''
    original images
    path start from 0
    depth, cam starts from 1
    '''
    # path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
    #          r'\rect_{:03d}_3_r5000.png'.format(l_img_num)
    # path_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
    #          r'/rect_{:03d}_3_r5000.png'.format(r_img_num)
    # depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Depths\scan114\depth_map_{' \
    #           r':04d}.pfm'.format(l_dpt_num)
    # depth_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Depths\scan114\depth_map_{' \
    #           r':04d}.pfm'.format(r_dpt_num)
    #
    # cam_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Cameras\train_temp\{' \
    #         r':08d}_cam.txt'.format(l_cam_num)
    # cam_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Cameras\train_temp\{' \
    #         r':08d}_cam.txt'.format(r_cam_num)

    '''
    new sets
    '''


    # path_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
    #          r'\tutorial 5 - spatial mapping\python\images2\rect_{:03d}_3_r5000.png'.format(l_img_num)
    # path_r = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
    #          r'\tutorial 5 - spatial mapping\python\images2\rect_{:03d}_3_r5000.png'.format(r_img_num)
    #
    # depth_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
    #           r'\tutorial 5 - spatial mapping\python\images2\depth_map_{:04d}.pfm'.format(l_dpt_num)
    #
    # depth_r = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
    #           r'\tutorial 5 - spatial mapping\python\images2\depth_map_{:04d}.pfm'.format(r_dpt_num)

    path_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
             r'\tutorial 5 - spatial mapping\python\closeimages\rect_{:03d}_3_r5000.png'.format(l_img_num)
    path_r = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
             r'\tutorial 5 - spatial mapping\python\closeimages\rect_{:03d}_3_r5000.png'.format(r_img_num)

    depth_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
              r'\tutorial 5 - spatial mapping\python\closeimages\depth_map_{:04d}.pfm'.format(l_dpt_num)

    depth_r = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
              r'\tutorial 5 - spatial mapping\python\closeimages\depth_map_{:04d}.pfm'.format(r_dpt_num)


    cam_l = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
            r'\tutorial 5 - spatial mapping\python\cam\{:08d}_cam.txt'.format(l_cam_num)
    cam_r = r'C:\Users\junsu\Documents\Brown\2023Spring\BVC\zed-examples-master\zed-examples-master\tutorials' \
            r'\tutorial 5 - spatial mapping\python\cam\{:08d}_cam.txt'.format(r_cam_num)


    l = cv2.imread(path_l)
    l_crop = l[104:616, 320:960]   # 104:616, 320:960
    l = l[:, 22:]

    r = cv2.imread(path_r)
    r_crop = r[104:616, 320:960]
    r = r[:, :-22]
    # d_l = cv2.imread(depth_l)

    d_l, scale_l = read_pfm(depth_l) # depth_l can't be used for imshow. check bvc.py.
    d_r, scale_r = read_pfm(depth_r)
    d_l = d_l[:, 22:] # aligend with left image
    d_r = d_r[:, 22:]
    # d_l[:, :100] = 0
    print(f'scale: {scale_l}')

    print(f'l.shape: {l.shape}')
    print(f'd.shape: {d_l.shape}')

    # import imageio
    # import matplotlib.pyplot as plt
    #
    # pfm_data = imageio.imread(depth_l)
    # plt.imshow(pfm_data, cmap='gray')
    # plt.colorbar()  # Add a colorbar if needed
    # plt.show()
    #
    # cv2.waitKey(0)


    # d_l = cv2.normalize(d_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # print(f'd_l min and max: {d_l.min()}  {d_l.max()}')
    # cv2.imshow('depth', d_l)
    # cv2.waitKey(0)
    # exit(0)

    # breakpoint()
    # d_l = depth_l
    # d_r = depth_r

    # d_l[d_l == np.inf] = 0
    # d_l = cv2.normalize(d_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # d_r[d_r == np.inf] = 0
    # d_r = cv2.normalize(d_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # crop depth image
    # d_l = d_l[104:616, 320:960]
    # d_r = d_r[104:616, 320:960]

    print(f' d_l[:5, :5]: {d_l[:5, :5]}')
    print(f' d_l max: {np.max(d_l)}')
    print(f' d_r[:5, :5]: {d_r[:5, :5]}')

    print(f'l.shape: {l.shape}') #  (720, 1280, 3)
    print(f'd.shape: {d_l.shape}')

    ixt_l, ext_l, _ = read_cam_file(cam_l)
    ixt_r, ext_r, _ = read_cam_file(cam_r)

    print(f'ext_r before: \n{ext_r}')
    # ext_r[0, 3] = ext_r[0, 3]
    # ext_r[1, 3] = ext_r[1, 3] - 6.3003 * 2
    # ext_l[:3, 3] = 0
    print(f'ext_r after: \n{ext_r}')

    print(f'ext_l: \n{ext_l}')

    '''
    adjust focal length
    '''
    # img_path =  r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data/Rectified/scan114_train/rect_{' \
    #             r':03d}_3_r5000.png'.format(1)  # (720, 1280, 3)
    img = cv2.imread(path_l)
    factor_y = 512 / img.shape[0]  # 512
    factor_x = 640 / img.shape[1]  # 640

    print(f'factor_y : {factor_y}')

    ixt_l_crop = np.copy(ixt_l)
    ixt_l_crop[0,0] = ixt_l[0,0] / factor_x
    ixt_l_crop[1,1] = ixt_l[1,1] / factor_y

    # ixt_l[0, 0] = ixt_l[0, 0] / factor_x
    # ixt_l[1, 1] = ixt_l[1, 1] / factor_y
    #
    # ixt_r[0, 0] = ixt_r[0, 0] / factor_x
    # ixt_r[1, 1] = ixt_r[1, 1] / factor_y

    # update cx and cy
    # ixt_l[0, 2] = ixt_l[0, 2] / factor_x
    # ixt_l[1, 2] = ixt_l[1, 2] / factor_y
    #
    # ixt_r[0, 2] = ixt_r[0, 2] / factor_x
    # ixt_r[1, 2] = ixt_r[1, 2] / factor_y

    print(f'ixt_l_crop: {ixt_l_crop}')
    h = l.shape[0] # the size i cropped to. the size i want the output image to be
    # w = l.shape[1]
    w = l.shape[1] - 22

    # h = l.shape[0]
    # w = l.shape[1]

    h_crop = 512
    w_crop = 640
    # l = l.reshape(h*w, -1)


    xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    uv2 = np.column_stack((xs, ys)).reshape(h, w, 2)
    print(f'uv2: {uv2[:5]}')

    xs_crop, ys_crop = np.meshgrid(np.arange(w_crop), np.arange(h_crop), indexing='xy')
    xs_crop = xs_crop.reshape(-1, 1)
    ys_crop = ys_crop.reshape(-1, 1)
    uv2_crop = np.column_stack((xs_crop, ys_crop)).reshape(h_crop, w_crop, 2)

    # make 3d coordinate homogeneous
    uv3 = np.concatenate((uv2, np.ones(shape=(uv2.shape[0], uv2.shape[1], 1))), axis=-1)
    uv3 = uv3.reshape(h*w, -1) # (h*w, 3)

    # change d shape
    d_l = d_l.reshape(-1, 1)
    d_r = d_r.reshape(-1, 1)

    # multiply depth
    uv3_l = np.copy(uv3) #* d_l
    uv3_r = np.copy(uv3) #* d_r

    print(f'uv3 last: {uv3[-5:]}')

    test_x = uv3_l[0][0] - ixt_l[0,2] * d_l[0] / ixt_l[0,0]
    test_y = uv3_l[0][1] - ixt_l[1,2] * d_l[0] / ixt_l[1,1]
    test_z = d_l[0]
    # pixel space to image to camera space
    i2c_l = np.sum(uv3_l[..., None, :] * np.linalg.inv(ixt_l)[None, ...], axis=-1)
    ic2_l_crop = np.sum(uv3_l[..., None, :] * np.linalg.inv(ixt_l_crop)[None, ...], axis=-1)
    # i2c_l = np.sum(uv3[..., None, :] * (ixt_l.T)[None, ...], axis=-1)
    i2c_r = np.sum(uv3_r[..., None, :] * np.linalg.inv(ixt_r)[None, ...], axis=-1)



    print(f'i2c_l last: {i2c_l[-5:]}')
    print(np.max(d_l), np.min(d_l))
    print( i2c_l[:, 2].shape)

    # maybe my depth wrong?
    i2c_l *= d_l
    i2c_r *= d_r

    print(f'test compare: [{test_x}, {test_y}, {test_z}]')
    print(i2c_l[0])

    # make homogeneous coordinate
    i2c_l = np.hstack((i2c_l, np.ones((i2c_l.shape[0], 1)))) # (h*w, 4)
    ic2_l_crop =  np.hstack((ic2_l_crop, np.ones((ic2_l_crop.shape[0], 1)))) # (h*w, 4)
    i2c_r = np.hstack((i2c_r, np.ones((i2c_r.shape[0], 1))))  # (h*w, 4)

    print(i2c_l.shape)
    print(i2c_l[:15])



    print(f'ext_l.shape: {ext_l.shape}')
    print(np.max(i2c_l[:, [0]]))

    # convert camera space into world space
    cam2world_l = np.identity(4)
    cam2world_l[:3, :3] = (ext_l[:3, :3]).T # ext_l is world2cam
    cam2world_l[:3, 3] = -ext_l[:3, :3].T @ ext_l[:3, 3]

    cam2world_r = np.identity(4)
    cam2world_r[:3, :3] = (ext_r[:3, :3]).T
    cam2world_r[:3, 3] = -ext_r[:3, :3].T @ ext_r[:3, 3]

    # doing here what I did in bvc.py
    # ext_r[0, 3] += 12.3003
    # ext_r[:3, :3] = ext_r[:3, :3].T
    # ext_r[:3, 3] = -ext_r[:3, :3] @ ext_r[:3, 3]
    # ext_r[:3, 3] -= ext_r[:3, 3]
    # ext_r[:3, 3] -= ext_l[:3, 3]
    # ext_r[0, 3] -= 6.3003


    print(f'ext_r: {ext_r}')

    c2w_l = np.sum(i2c_l[..., None, :] * cam2world_l[None, ...], axis=-1)
    # c2w_l = np.sum(i2c_l[..., None, :] * np.linalg.inv(ext_l)[None, ...], axis=-1)
    # c2w_l = np.sum(i2c_l[..., None, :] * (ext_l.T)[None, ...], axis=-1)

    # for i in range(i2c_l.shape[0]):
    #     i2c_l[i] = ext_l @ i2c_l[i]
    # c2w_l = i2c_l

    c2w_l_crop = np.sum(ic2_l_crop[..., None, :] * (ext_l.T)[None, ...], axis=-1)

    # c2w_r = np.sum(i2c_r[..., None, :] * cam2world_r[None, ...], axis=-1)
    # c2w_r = np.sum(i2c_r[..., None, :] * np.linalg.inv(ext_r)[None, ...], axis=-1)
    # c2w_r = np.sum(i2c_r[..., None, :] * (ext_r.T)[None, ...], axis=-1)

    # convert world space to left camera space
    c2w_r = np.sum(i2c_r[..., None, :] * cam2world_r[None, ...], axis=-1)

    # point cloud to right imge
    cam_coord = c2w_l @ ext_r.T # ext_r is world to camera
                              # c2w_l.shape = (# of points, 4)
    ixt_r = np.hstack((ixt_r, np.array([0,0,0]).reshape(3,1))) # 3 * 4
    img_coord = cam_coord @ ixt_r.T
    print(f'img_coord: {img_coord[:5, :]}')
    img_coord = img_coord / img_coord[:, [2]]
    print(f'img_coord: {img_coord[:5, :]}')
    print(f'img_coord min: {img_coord[:, [0]].min()}')
    print(f'img_coord max: {img_coord[:, [0]].max()}')
    print(f'img_coord min: {img_coord[:, [1]].min()}')
    print(f'img_coord max: {img_coord[:, [1]].max()}')

    img_coord = img_coord[:, :2].reshape(h, w, 2)
    # cv2.imshow('img', img_coord)
    # cv2.waitKey(0)
    print(f'c2w_l: {c2w_l[:5]}')
    # exit(0)
    # homogenous to normal coordinate
    c2w_l = c2w_l[:, :3] #.reshape(h,w,3).astype(float)
    c2w_l_crop = c2w_l_crop[:, :3]
    c2w_r = c2w_r[:, :3]

    # make point cloud instances
    pcd_l = o3d.geometry.PointCloud()
    pcd_l_crop = o3d.geometry.PointCloud()
    pcd_r = o3d.geometry.PointCloud()

    i2c_l = i2c_l[:, :3]
    i2c_r = i2c_r[:, :3]

    # convert np array to vector of 3d vectors
    pcd_l.points = o3d.utility.Vector3dVector(c2w_l)
    # pcd_l.points = o3d.utility.Vector3dVector(i2c_l)
    # pcd_l_crop.points = o3d.utility.Vector3dVector(c2w_l_crop)

    # pointcloud to right image
    right_camera_space = c2w_l

    pcd_r.points = o3d.utility.Vector3dVector(c2w_r)
    # pcd_r.points = o3d.utility.Vector3dVector(i2c_r)

    # change image shape
    l = l.reshape(-1, 3) # (327680, 3)
    l_crop = l_crop.reshape(-1,3)
    r = r.reshape(-1, 3)
    r_crop = r_crop.reshape(-1,3)

    # add color
    pcd_l.colors = o3d.utility.Vector3dVector(l.astype(np.float64) / 255.0)
    pcd_l_crop.colors = o3d.utility.Vector3dVector(l.astype(np.float64) / 255.0)
    pcd_r.colors = o3d.utility.Vector3dVector(r.astype(np.float64) / 255.0)

    # transform point clouds. otherwise generate wrong point clouds
    # pcd_l.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd_l_crop.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd_r.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # visualize point cloud
    # o3d.visualization.draw_geometries([pcd_l])

    # toggle geometries
    vis = o3d.visualization.Visualizer()
    vis_temp = open3d.visualization.VisualizerWithKeyCallback()
    vis_temp.create_window()
    vis_temp.add_geometry(pcd_l)
    vis_temp.add_geometry(pcd_r)

    # vis_temp.add_geometry(pcd_)
    def toggle_display(vis):

        geo = input("which geo: ")
        if geo == 'l':
            vis_temp.clear_geometries()
            vis_temp.add_geometry(pcd_l, reset_bounding_box=False)
            # vis_temp.remove_geometry(pcd_r)
        elif geo == 'r':
            vis_temp.clear_geometries()
            vis_temp.add_geometry(pcd_r, reset_bounding_box=False)
            # vis_temp.remove_geometry(pcd_l)
        else:
            vis_temp.clear_geometries()
            vis_temp.add_geometry(pcd_r, reset_bounding_box=False)
            vis_temp.add_geometry(pcd_l, reset_bounding_box=False)

    def toggle_left(vis_temp):
        vis_temp.clear_geometries()
        vis_temp.add_geometry(pcd_l,reset_bounding_box=False)
        # vis_temp.remove_geometry(pcd_r)
        print('toggle left')
        # pcd_r.points.extend(pcd_l.points)
        # vis_temp.update_geometry(pcd_r)
        # vis_temp.update_renderer()
        # vis_temp.poll_events()
        # vis_temp.run()
    def toggle_right(vis_temp):
        vis_temp.clear_geometries()
        vis_temp.add_geometry(pcd_r,reset_bounding_box=False)
        # vis_temp.remove_geometry(pcd_l)
        print('toggle right')
        # vis_temp.update_geometry(pcd_r)
        # vis_temp.update_renderer()
        # vis_temp.poll_events()
        # vis_temp.run()
    # vis_temp.register_key_callback(ord(" "), toggle_display)

    def toggle_all(vis_temp):
        vis_temp.clear_geometries()
        vis_temp.add_geometry(pcd_l, reset_bounding_box=False)
        vis_temp.add_geometry(pcd_r, reset_bounding_box=False)

    # need to use GLFW key
    # vis_temp.register_key_callback(ord("L"), toggle_left)
    # vis_temp.register_key_callback(ord("R"), toggle_right)
    # vis_temp.register_key_callback(ord("A"), toggle_all)
    vis_temp.register_key_callback(49, toggle_left)
    vis_temp.register_key_callback(50, toggle_right)
    vis_temp.register_key_callback(51, toggle_all)


    vis_temp.run()
    vis.destroy_window()
if __name__ == "__main__":
    main()
