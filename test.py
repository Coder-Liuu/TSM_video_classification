import os
import cv2
import copy
import time
import paddle
import random
import traceback

import numpy as np
import os.path as osp
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as init

from PIL import Image
from tqdm import tqdm
from paddle import ParamAttr
from collections import OrderedDict
from collections.abc import Sequence
from paddle.regularizer import L2Decay
from paddle.nn import (Conv2D, BatchNorm2D, Linear, Dropout, MaxPool2D,
                       AdaptiveAvgPool2D)

from settings import *
from data_preprocessing import *
from model import *
from utils import *


def test():
    pipeline = Compose(train_mode=False)
    data = VideoDataset(file_path=train_file_path, pipeline=pipeline, suffix=suffix)

    sampler = paddle.io.DistributedBatchSampler(
        data,
        batch_size=4,
        shuffle=True,
        drop_last=True
    )

    data_loader = paddle.io.DataLoader(
        data,
        batch_sampler=sampler,
        places=paddle.set_device(device),
    )

    for img, label in data_loader():
        print(img.shape)
        print(label)
        break
