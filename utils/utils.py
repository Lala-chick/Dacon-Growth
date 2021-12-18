import random
import os
import numpy as np
import torch
import cv2
import argparse

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def extract_day(images):
    day = int(images.split('.')[-2][-2:])
    return day

def make_day_array(images):
    day_array = np.array([extract_day(x) for x in images])
    return day_array

def resize_module(path, dsize=224):
    image = cv2.imread(path)
    x, y, _ = image.shape

    if x != y:
        size = max(x,y)
        new_image = np.zeros((size, size, 3), np.uint8)
        offset = (round(abs(x-size)/2), round(abs(y-size)/2))
        new_image[offset[0]:offset[0]+x][offset[1]:offset[1]+y] = image
    else:
        new_image = image
    new_image = cv2.resize(new_image, dsize=(dsize, dsize))
    cv2.imwrite(path, new_image)

def str2bool(x):
    if isinstance(x, bool):
        return x
    if x.lower() in ('true'):
        return True
    elif x.lower() in('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')