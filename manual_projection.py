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
    Cropped images
    '''
    # path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
    #          r'\rect_001_3_r5000.png'
    # path_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
    #          r'/rect_002_3_r5000.png'
    # depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Depths\scan114\depth_map_0000.pfm'

    '''
    original images
    '''
    path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
             r'\rect_001_3_r5000.png'
    path_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
             r'/rect_002_3_r5000.png'
    depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Depths\scan114\depth_map_0000.pfm'


    cam_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Cameras\train_temp\00000000_cam.txt'
    cam_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Cameras\train_temp\00000001_cam.txt'

    l = cv2.imread(path_l)
    r = cv2.imread(path_r)
    d = cv2.imread(depth_l)
    d, _ = read_pfm(depth_l)

    # max = np.max(d)
    # min = np.min(d)
    # d -= min
    # d  = d / float(max - min)
    # print(np.max(d))

    # cv2.imshow('depth', d)
    # cv2.waitKey(0)
    print(f'l.shape: {l.shape}')
    print(f'd.shape: {d.shape}')

    # height = 512  # 720
    # width = 640  # 1280

    # l_fx = 670.2325
    # l_fy = 670.2325
    # l_cx = 645.2366
    # l_cy = 350.6550
    #
    # r_fx = 670.2325
    # r_fy = 670.2325
    # r_cx = 645.2366
    # r_cy = 350.6550

    '''  l_ext = np.array([[1.0000, 0.0000, 0.0000, 0.0000],
                      [0.0000, 1.0000, 0.0000, 0.0000],
                      [0.0000, 0.0000, 1.0000, 0.0000],
                      [0.0000, 0.0000, 0.0000, 1.0000]])

    r_ext = np.array([[1.0000, 0.0000, 0.0000, 0.0006],#[1.0000, 0.0000, 0.0000, 6.3003],
                      [0.0000, 1.0000, 0.0000, 0.0000],
                      [0.0000, 0.0000, 1.0000, 0.0000],
                      [0.0000, 0.0000, 0.0000, 1.0000]])'''


    # l_ext = np.array([[0.9997, 0.0123, -0.0223, -0.6748],
    #                   [-0.0119, 0.9998, 0.0163, -0.2631],
    #                   [0.0225, -0.0160, 0.9996, 0.0991],
    #                   [0.0000, 0.0000, 0.0000, 1.0000]])
    #
    # r_ext = np.array([[0.9997, 0.0123, -0.0223, -0.6748 + 0.06],  # 5.6255
    #                   [-0.0119, 0.9998, 0.0163, -0.2631],
    #                   [0.0225, -0.0160, 0.9996, 0.0991],
    #                   [0.0000, 0.0000, 0.0000, 1.0000]])


    ixt_l, ext_l, _ = read_cam_file(cam_l)
    ixt_r, ext_r, _ = read_cam_file(cam_r)
    print(f'ixt_l: {ixt_l}')
    '''
    adjust focal length
    '''
    img_path =  r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data/Rectified/scan114_train/rect_{' \
                r':03d}_3_r5000.png'.format(1)  # (720, 1280, 3)
    img = cv2.imread(img_path)
    factor_y = 512 / img.shape[0]
    factor_x = 640 / img.shape[1]

    print(f'factor_y : {factor_y}')

    ixt_l_crop = np.copy(ixt_l)
    ixt_l_crop[0,0] = ixt_l[0,0] / factor_x
    ixt_l_crop[1,1] = ixt_l[1,1] / factor_y
    # ixt_r[0, 0] = ixt_r[0, 0] / factor_x
    # ixt_r[1, 1] = ixt_r[1, 1] / factor_y


    print(f'ixt_l_crop: {ixt_l_crop}')
    h = 512 # l.shape[0]
    w = 640 # l.shape[1]

    h_crop = 512
    w_crop = 640
    # l = l.reshape(h*w, -1)


    '''# what i'm doing here is multiplying rgb by ixt matrix. Wrong. I need to multiply pixel position by ixt matrix.
    # but the issue is that rgb * ixt matrix doesn't match.
    test = l[..., None, :] * np.linalg.inv(ixt_l)[None, ...] # check BVC/Meeting/4.12
    print(test[0])
    test = np.sum(test, axis=-1)
    print(f'test.shape: {test.shape}')
    print(f'l[0]: {l[0]}')
    print(f'test[0]: {test[0]}')

    first = np.linalg.inv(ixt_l) @ l[0, :]
    print(f'first: {first}')
    print(f'inverse: {np.linalg.inv(ixt_l)}')
    # '''

    xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    uv2 = np.column_stack((xs, ys)).reshape(h, w, 2)

    xs_crop, ys_crop = np.meshgrid(np.arange(w_crop), np.arange(h_crop), indexing='xy')
    xs_crop = xs_crop.reshape(-1, 1)
    ys_crop = ys_crop.reshape(-1, 1)
    uv2_crop = np.column_stack((xs_crop, ys_crop)).reshape(h_crop, w_crop, 2)



    ones = np.ones(shape=(uv2.shape[0], uv2.shape[1], 1))
    # print(ones.shape)
    uv3 = np.concatenate((uv2, np.ones(shape=(uv2.shape[0], uv2.shape[1], 1))), axis=-1)
    # print(uv3.shape)
    # print(uv3[0, :10, :])
    # print(uv[0, :10, :]) # print columns
    uv3 = uv3.reshape(h*w, -1) # (h*w, 3)






    d = d.reshape(-1, 1)
    # inverse depth
    # d *= -1

    # multiply depth
    # uv3[:, [2]] = uv3[:, [2]] * d
    # uv3[:, [2]] = uv3[:, [2]]
    uv3 = uv3 * d

    print(f'uv3 last: {uv3[-5:]}')
    # pixel space to image to camera space
    i2c_l = np.sum(uv3[..., None, :] * np.linalg.inv(ixt_l)[None, ...], axis=-1)
    ic2_l_crop = np.sum(uv3[..., None, :] * np.linalg.inv(ixt_l_crop)[None, ...], axis=-1)

    # i2c_l = np.sum(uv3[..., None, :] * (ixt_l.T)[None, ...], axis=-1)
    i2c_r = np.sum(uv3[..., None, :] * np.linalg.inv(ixt_r)[None, ...], axis=-1)
    # d = d.reshape(-1, 1)


    print(f'i2c_l last: {i2c_l[-5:]}')

    print(np.max(d), np.min(d))
    print( i2c_l[:, 2].shape)
    # i2c_l = i2c_l * d # multiply depth. is this correct way?
    # print(i2c_l[0])
    # print(f'inverse: {np.linalg.inv(ixt_l)}')

    # make homogeneous coordinate
    i2c_l = np.hstack((i2c_l, np.ones((i2c_l.shape[0], 1)))) # (h*w, 4)
    ic2_l_crop =  np.hstack((ic2_l_crop, np.ones((ic2_l_crop.shape[0], 1)))) # (h*w, 4)

    i2c_r = np.hstack((i2c_r, np.ones((i2c_r.shape[0], 1))))  # (h*w, 4)

    print(i2c_l.shape)
    print(i2c_l[:15])
    print(f'ext_l.shape: {ext_l.shape}')

    print(np.max(i2c_l[:, [0]]))

    # c2w_l = np.sum(i2c_l[..., None, :] * np.linalg.inv(ext_l)[None, ...], axis=-1)
    c2w_l = np.sum(i2c_l[..., None, :] * (ext_l.T)[None, ...], axis=-1)
    c2w_l_crop = np.sum(ic2_l_crop[..., None, :] * (ext_l.T)[None, ...], axis=-1)

    c2w_r = np.sum(i2c_r[..., None, :] * (ext_r.T)[None, ...], axis=-1)
    print(c2w_l[:5])

    # homogenous to normal coordinate
    c2w_l = c2w_l[:, :3] #.reshape(h,w,3).astype(float)
    c2w_l_crop = c2w_l_crop[:, :3]
    c2w_r = c2w_r[:, :3]
    '''
    option 1
    '''
    # print(c2w_l.shape)
    # c2w_l = c2w_l.reshape(h,w,-1)
    # # c2w_l = np.flip(c2w_l, 1)
    # c2w_l = np.flipud(c2w_l)
    # # c2w_l = np.flip(c2w_l, 2)
    # # c2w_l = np.fliplr(c2w_l)
    # c2w_l = c2w_l.reshape(-1, 3)
    #
    # # reverse the depth
    # c2w_l[:, 2] = c2w_l[:, 2] * -1

    '''
    option 2
    '''
    print(c2w_l.shape)
    # c2w_l = c2w_l.reshape(h, w, -1)
    # c2w_l = c2w_l.reshape(-1, 3)

    # reverse the depth
    # c2w_l[:, 2] = c2w_l[:, 2] * -1


    # make point cloud
    pcd_l = o3d.geometry.PointCloud()
    pcd_l_crop = o3d.geometry.PointCloud()

    pcd_r = o3d.geometry.PointCloud()
    print(type(c2w_l))
    pcd_l.points = o3d.utility.Vector3dVector(c2w_l)
    pcd_l_crop.points = o3d.utility.Vector3dVector(c2w_l_crop)
    pcd_r.points = o3d.utility.Vector3dVector(c2w_r)
    l = l.reshape(-1, 3) # (327680, 3)
    r = r.reshape(-1, 3)
    pcd_l.colors = o3d.utility.Vector3dVector(l.astype(np.float64) / 255.0)
    pcd_l_crop.colors = o3d.utility.Vector3dVector(l.astype(np.float64) / 255.0)
    pcd_r.colors = o3d.utility.Vector3dVector(r.astype(np.float64) / 255.0)
    pcd_l.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_l_crop.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_r.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


    # o3d.visualization.draw_geometries([pcd_r])
    o3d.visualization.draw_geometries([pcd_l, pcd_l_crop])
    # o3d.visualization.draw_geometries([pcd_l, pcd_r])
if __name__ == "__main__":
    main()
