import os
import sys
import numpy as np
import scipy.io as sio
from glob import glob
from tqdm import tqdm
import math


def load_annotation(dataset_dir, phase):
    data_dir = os.path.join(dataset_dir, phase)
    joint_data = sio.loadmat(os.path.join(data_dir, 'joint_data.mat'))
    joint_2d = joint_data['joint_uvd'].astype(np.float32)
    V, N, J, C = joint_2d.shape
    joint_2d = joint_2d.reshape([V*N, J, C])
    depth_path = glob(os.path.join(data_dir, "depth_*.png"))
    depth_name = [path.split('/')[-1] for path in depth_path]
    depth_name.sort()
    return joint_2d, depth_name


def gen_data(dataset_dir, phase, output_path, fx, fy):
    print('Generate {} data.'.format(phase))
    joint_2d, depth_name = load_annotation(dataset_dir, phase)
    step = 10 if phase == 'train' else 10
    joint_2d = joint_2d[::step]
    depth_name = depth_name[::step]
    print(joint_2d.shape[0])

    com_2d = np.mean(joint_2d, axis=1)
    left = np.min(joint_2d[:, :, 0], axis=1)
    right = np.max(joint_2d[:, :, 0], axis=1)
    up = np.min(joint_2d[:, :, 1], axis=1)
    down = np.max(joint_2d[:, :, 1], axis=1)
    offset = 0
    left = (left * com_2d[:, 2] / fx - offset) / com_2d[:, 2] * fx
    right = (right * com_2d[:, 2] / fx + offset) / com_2d[:, 2] * fx
    up = (up * com_2d[:, 2] / fy - offset) / com_2d[:, 2] * fy
    down = (down * com_2d[:, 2] / fy + offset) / com_2d[:, 2] * fy
    with open(output_path, 'w') as f:
        for name, l, r, u, d in zip(depth_name, left, right, up, down):
            print('{} {:.0f},{:.0f},{:.0f},{:.0f},0'.format(os.path.join(phase, name), l, u, r, d), file=f)


if __name__ == '__main__':
    # dataset_dir = '/media/sda1/dataset/nyu_hand_dataset_v2/dataset'
    dataset_dir = '/home/dataset/nyu_hand_dataset_v2/dataset'
    output_dir = 'data/'
    fx = 588.03
    fy = 587.07
    gen_data(dataset_dir, 'train', os.path.join(output_dir, 'nyu_train_label.txt'), fx, fy)
    gen_data(dataset_dir, 'test', os.path.join(output_dir, 'nyu_test_label.txt'), fx, fy)