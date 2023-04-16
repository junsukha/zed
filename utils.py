import open3d.geometry
import pyzed.sl as sl
import numpy as np
import cv2
from open3d import *
import re

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

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()] # rstrip removes trailing spaces. one string for each line
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    return intrinsics, extrinsics, depth_min

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
