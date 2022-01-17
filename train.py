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


def train_model(validate=True):
    """Train model entry
    Args:
        weights (str): weights path for finetuning.
        validate (bool): Whether to do evaluation. Default: False.
    """
    output_dir = f"./output/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Construct model
    tsm = ResNetTSM(pretrained=pretrained,
                    layers=layers,
                    num_seg=num_seg)
    head = TSMHead(num_classes=num_classes,
                   in_channels=in_channels,
                   drop_ratio=drop_ratio)
    model = Recognizer2D(backbone=tsm, head=head)

    # 2. Construct dataset and dataloader
    train_pipeline = Compose(train_mode=True)
    train_dataset = VideoDataset(file_path=train_file_path,
                                 pipeline=train_pipeline,
                                 suffix=suffix)
    train_sampler = paddle.io.DistributedBatchSampler(train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=train_shuffle,
                                                      drop_last=True)  

    train_loader = paddle.io.DataLoader(train_dataset,
                                        batch_sampler=train_sampler,
                                        places=paddle.set_device(device),
                                        return_list=return_list)

    if validate:
        valid_pipeline = Compose(train_mode=False)
        valid_dataset = VideoDataset(file_path=valid_file_path,
                                     pipeline=valid_pipeline,
                                     suffix=suffix)
        valid_sampler = paddle.io.DistributedBatchSampler(valid_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=valid_shuffle,
                                                          drop_last=True)
        valid_loader = paddle.io.DataLoader(valid_dataset,
                                            batch_sampler=valid_sampler,
                                            places=paddle.set_device(device),
                                            return_list=return_list)

    # 3. Construct solver.
    lr = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        momentum=momentum,
        parameters=model.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    )

    # 4. Train Model
    best = 0.
    train_steps = len(train_loader)
    vaild_steps = len(valid_loader)
    for epoch in range(0, epochs):
        model.train()
        record_list = build_record(framework)
        tic = time.time()
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)

            # 4.1 forward
            outputs = model.train_step(data)

            # 4.2 backward
            avg_loss = outputs['loss']
            avg_loss.backward()

            # 4.3 minimize
            optimizer.step()
            optimizer.clear_grad()

            # log record
            record_list['lr'].update(optimizer._global_learning_rate(), batch_size)
            for name, value in outputs.items():
                record_list[name].update(value, batch_size)

            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % log_interval == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, epochs, "train", ips, train_steps)

        # learning rate epoch step
        lr.step()

        ips = "avg_ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

        def evaluate(best):
            model.eval()
            record_list = build_record(framework)
            record_list.pop('lr')
            tic = time.time()
            for i, data in enumerate(valid_loader):
                outputs = model.val_step(data)

                # log_record
                for name, value in outputs.items():
                    record_list[name].update(value, batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % log_interval == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, epochs, "val", ips, vaild_steps)

            ips = "avg_ips: {:.5f} instance/sec.".format(
                batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            for top_flag in ['hit_at_one', 'top1']:
                if record_list.get(
                        top_flag) and record_list[top_flag].avg > best:
                    best = record_list[top_flag].avg
                    best_flag = True
            return best, best_flag

        # 5. Validation
        if validate or epoch == epochs - 1:
            with paddle.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                paddle.save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                paddle.save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
                print(
                    f"Already save the best model (top1 acc){int(best *10000)/10000}"
                )

        # 6. Save model and optimizer
        if epoch % save_interval == 0 or epoch == epochs - 1:
            paddle.save(
                optimizer.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1}.pdopt"))
            paddle.save(
                model.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1}.pdparams"))

    print(f'training {model_name} finished')


train_model(True)
