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

from settings import Color

def coloring(message, color="OKGREEN"):
    return message


def build_record(framework_type):
    record_list = [
        ("loss", AverageMeter('loss', '7.5f')),
        ("lr", AverageMeter('lr', 'f', need_avg=False)),
        ("batch_time", AverageMeter('elapse', '.3f')),
        ("reader_time", AverageMeter('reader', '.3f')),
    ]

    if 'Recognizer' in framework_type:
        record_list.append(("top1", AverageMeter("top1", '.5f')))
        record_list.append(("top5", AverageMeter("top5", '.5f')))

    record_list = OrderedDict(record_list)
    return record_list


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name='', fmt='f', need_avg=True):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        if isinstance(val, paddle.Tensor):
            val = val.numpy()[0]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(s=self.sum / 60,
                                                            self=self)

    @property
    def mean(self):
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)

def log_batch(metric_list, batch_id, epoch_id, total_epoch, mode, ips, total_steps):
    # NOTE: 删除top5
    if metric_list.get("top5") != None:
        del metric_list["top5"]
        
    metric_str = ' '.join([str(m.value) for m in metric_list.values()])
    epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch_id, total_epoch)
    step_str = "{:s} step:{} / {}".format(mode, batch_id, total_steps)
    print("{:s} {:s} {:s}s {}".format(
        coloring(epoch_str, "HEADER") if batch_id == 0 else epoch_str,
        coloring(step_str, "PURPLE"), coloring(metric_str, 'OKGREEN'), ips))


def log_epoch(metric_list, epoch, mode, ips):
    metric_avg = ' '.join([str(m.mean) for m in metric_list.values()] +
                          [metric_list['batch_time'].total])

    end_epoch_str = "END epoch:{:<3d}".format(epoch)

    print("{:s} {:s} {:s}s {}".format(coloring(end_epoch_str, "RED"),
                                            coloring(mode, "PURPLE"),
                                            coloring(metric_avg, "OKGREEN"),
                                            ips))

class CenterCropMetric(object):
    def __init__(self, data_size, batch_size, log_interval=20):
        """prepare for metrics
        """
        super().__init__()
        self.data_size = data_size
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.top1 = []
        self.top5 = []

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        labels = data[1]

        top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        
        top5 = np.zeros_like(top1.numpy())
        # NOTE: 打开Top5
        # top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)

        self.top1.append(top1.numpy())
        self.top5.append(top5)
        # preds ensemble
        if batch_id % self.log_interval == 0:
            print("[TEST] Processing batch {}/{} ...".format(batch_id, self.data_size // self.batch_size))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        print('[TEST] finished, avg_acc1= {}, avg_acc5= {} '.format(
            np.mean(np.array(self.top1)), np.mean(np.array(self.top5)))
        )
