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
    path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
             r'\rect_001_3_r5000.png'
    path_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Rectified\scan114_train' \
             r'/rect_002_3_r5000.png'
    depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\cropped-Depths\scan114\depth_map_0000.pfm'

    '''
    original images
    '''
    # path_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
    #          r'\rect_001_3_r5000.png'
    # path_r = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Rectified\scan114_train' \
    #          r'/rect_002_3_r5000.png'
    # depth_l = r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data\Depths\scan114\depth_map_0000.pfm'
    #

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

    cv2.imshow('depth', d)
    cv2.waitKey(0)
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

    # adjust focal length
    # img_path =  r'\\wsl.localhost\Ubuntu-20.04\home\junsukhaa\BVC\data\dtu\zed_data/Rectified/scan114_train/rect_{' \
    #             r':03d}_3_r5000.png'.format(1)  # (720, 1280, 3)
    # img = cv2.imread(img_path)
    # factor_y = 512 / img.shape[0]
    # factor_x = 640 / img.shape[1]
    #
    # ixt_l[0,0] = ixt_l[0,0] / factor_x
    # ixt_l[1,1] = ixt_l[1,1] / factor_y

    print(f'lxt_l: {ixt_l}')
    h = l.shape[0]
    w = l.shape[1]
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



    ones = np.ones(shape=(uv2.shape[0], uv2.shape[1], 1))
    # print(ones.shape)
    uv3 = np.concatenate((uv2, np.ones(shape=(uv2.shape[0], uv2.shape[1], 1))), axis=-1)
    # print(uv3.shape)
    # print(uv3[0, :10, :])
    # print(uv[0, :10, :]) # print columns

    uv3 = uv3.reshape(h*w, -1) # (h*w, 3)




    d = d.reshape(-1, 1) * -1
    # multiply depth
    # uv3[:, [2]] = uv3[:, [2]] * d
    # uv3[:, [2]] = uv3[:, [2]]
    uv3 = uv3 * d

    print(f'uv3 last: {uv3[-5:]}')
    # pixel space to image to camera space
    i2c = np.sum(uv3[..., None, :] * np.linalg.inv(ixt_l)[None, ...], axis=-1)
    # i2c = np.sum(uv3[..., None, :] * (ixt_l.T)[None, ...], axis=-1)

    # d = d.reshape(-1, 1)


    print(f'i2c last: {i2c[-5:]}')

    print(np.max(d), np.min(d))
    print( i2c[:, 2].shape)
    # i2c = i2c * d # multiply depth. is this correct way?
    # print(i2c[0])
    # print(f'inverse: {np.linalg.inv(ixt_l)}')

    # make homogeneous coordinate
    i2c = np.hstack((i2c, np.ones((i2c.shape[0], 1)))) # (h*w, 4)

    print(i2c.shape)
    print(i2c[:15])
    print(f'ext_l.shape: {ext_l.shape}')

    print(np.max(i2c[:, [0]]))

    # c2w = np.sum(i2c[..., None, :] * np.linalg.inv(ext_l)[None, ...], axis=-1)
    c2w = np.sum(i2c[..., None, :] * (ext_l.T)[None, ...], axis=-1)

    print(c2w[:5])

    # homogenous to normal coordinate
    c2w = c2w[:, :3] #.reshape(h,w,3).astype(float)

    '''
    option 1
    '''
    # print(c2w.shape)
    # c2w = c2w.reshape(h,w,-1)
    # # c2w = np.flip(c2w, 1)
    # c2w = np.flipud(c2w)
    # # c2w = np.flip(c2w, 2)
    # # c2w = np.fliplr(c2w)
    # c2w = c2w.reshape(-1, 3)
    #
    # # reverse the depth
    # c2w[:, 2] = c2w[:, 2] * -1

    '''
    option 2
    '''
    print(c2w.shape)
    c2w = c2w.reshape(h, w, -1)
    # c2w = np.flip(c2w, 1)
    # c2w = np.flipud(c2w)
    # c2w = np.flip(c2w, 2)
    # c2w = np.fliplr(c2w)
    c2w = c2w.reshape(-1, 3)

    # reverse the depth
    # c2w[:, 2] = c2w[:, 2] * -1


    # make point cloud
    pcd = o3d.geometry.PointCloud()
    print(type(c2w))
    pcd.points = o3d.utility.Vector3dVector(c2w)

    l = l.reshape(-1, 3) # (327680, 3)
    pcd.colors = o3d.utility.Vector3dVector(l.astype(np.float64) / 255.0)



    o3d.visualization.draw_geometries([pcd])
if __name__ == "__main__":
    main()
