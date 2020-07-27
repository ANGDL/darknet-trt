import os
import numpy as np


def create_list(path, n, save_path):
    image_list = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            image_list.append(file_path)

    rand_list = np.random.choice(image_list, n)

    with open(save_path, 'w') as f:
        for s in rand_list:
            f.write(str(s) + '\n')


if __name__ == '__main__':
    path = '/mnt/d/datasets/coco/val2017'
    n = 1000
    save_path = './calib_image_list.txt'
    create_list(path, n, save_path)
