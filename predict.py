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

def inference(model_file='./output/TSM/TSM_best.pdparams'):
    # 1. Construct model
    tsm = ResNetTSM(pretrained=None,
                    layers=layers,
                    num_seg=num_seg)
    head = TSMHead(num_classes=num_classes,
                   in_channels=in_channels,
                   drop_ratio=drop_ratio)
    model = Recognizer2D(backbone=tsm, head=head)

    # 2. Construct dataset and dataloader.
    test_pipeline = Compose(train_mode=False)
    test_dataset = VideoDataset(file_path=valid_file_path,
                                     pipeline=test_pipeline,
                                     suffix=suffix)
    test_sampler = paddle.io.DistributedBatchSampler(test_dataset,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset,
                                       batch_sampler=test_sampler,
                                       places=paddle.set_device(device),
                                       return_list=return_list)

    model.eval()
    state_dicts = paddle.load(model_file)
    model.set_state_dict(state_dicts)

    for batch_id, data in enumerate(test_loader):
        _, labels = data
        outputs = model.test_step(data)
        scores = F.softmax(outputs)
        class_id = paddle.argmax(scores, axis=-1)
        pred = class_id.numpy()[0]
        label = labels.numpy()[0][0]
        
        print('预测类别：{}, 真实类别：{}'.format(pred, label))
        if batch_id > 5:
            break

model_file='./output/TSM/TSM_best.pdparams'
inference(model_file)
