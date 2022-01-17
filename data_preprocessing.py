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


class VideoDecoder(object):
    """
    将视频解码为帧
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self):
        pass

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'video'

        cap = cv2.VideoCapture(file_path)
        videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampledFrames = []
        for i in range(videolen):
            ret, frame = cap.read()
            # maybe first frame is empty
            if ret == False:
                continue
            img = frame[:, :, ::-1]
            sampledFrames.append(img)
        results['frames'] = sampledFrames
        results['frames_len'] = len(sampledFrames)
        return results


class Sampler(object):
    """
    帧采样
        对一段视频进行分段采样，大致流程为：
        1. 对视频进行分段；
        2. 从每段视频随机选取一个起始位置；
        3. 从选取的起始位置采集连续的k帧。
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        valid_mode(bool): True or False.
        select_left: Whether to select the frame to the left in the middle when the sampling interval is even in the test mode.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 num_seg,
                 seg_len,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.valid_mode = valid_mode
        self.select_left = select_left
        self.dense_sample = dense_sample
        self.linspace_sample = linspace_sample

    def _get(self, frames_idx, results):
        frames = np.array(results['frames'])
        imgs = []
        for idx in frames_idx:
            imgbuf = frames[idx]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)
        results['imgs'] = imgs
        return results

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        average_dur = int(frames_len / self.num_seg)
        frames_idx = []

        if self.select_left:
            if not self.valid_mode:
                if average_dur > 0:
                    offsets = np.multiply(list(range(self.num_seg)),
                                          average_dur) + np.random.randint(
                                              average_dur, size=self.num_seg)
                elif frames_len > self.num_seg:
                    offsets = np.sort(
                        np.random.randint(frames_len, size=self.num_seg))
                else:
                    offsets = np.zeros(shape=(self.num_seg, ))
            else:
                if frames_len > self.num_seg:
                    average_dur_float = frames_len / self.num_seg
                    offsets = np.array([
                        int(average_dur_float / 2.0 + average_dur_float * x)
                        for x in range(self.num_seg)
                    ])
                else:
                    offsets = np.zeros(shape=(self.num_seg, ))

            frames_idx = list(offsets)
            frames_idx = [x % frames_len for x in frames_idx]
            return self._get(frames_idx, results)

class Scale(object):
    """
    Scale images.
    Args:
        short_size(float | int): Short size of an image will be scaled to the short_size.
        fixed_ratio(bool): Set whether to zoom according to a fixed ratio. default: True
        do_round(bool): Whether to round up when calculating the zoom ratio. default: False
        backend(str): Choose pillow or cv2 as the graphics processing backend. default: 'pillow'
    """
    def __init__(self,
                 short_size,
                 fixed_ratio=True,
                 do_round=False
                 ):
        self.short_size = short_size
        self.fixed_ratio = fixed_ratio
        self.do_round = do_round

    def __call__(self, results):
        """
        Performs resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        imgs = results['imgs']
        resized_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.size
            if (w <= h and w == self.short_size) or (h <= w
                                                     and h == self.short_size):
                resized_imgs.append(img)
                continue
            if w < h:
                ow = self.short_size
                if self.fixed_ratio:
                    oh = int(self.short_size * 4.0 / 3.0)
                else:
                    oh = int(round(h * self.short_size /
                                   w)) if self.do_round else int(
                                       h * self.short_size / w)
            else:
                oh = self.short_size
                if self.fixed_ratio:
                    ow = int(self.short_size * 4.0 / 3.0)
                else:
                    ow = int(round(w * self.short_size /
                                   h)) if self.do_round else int(
                                       w * self.short_size / h)

            resized_imgs.append(
                Image.fromarray(
                    cv2.resize(np.asarray(img), (ow, oh),
                                interpolation=cv2.INTER_LINEAR)))
        results['imgs'] = resized_imgs
        return results


class MultiScaleCrop(object):
    """
    Random crop images in with multiscale sizes
    Args:
        target_size(int): Random crop a square with the target_size from an image.
        scales(int): List of candidate cropping scales.
        max_distort(int): Maximum allowable deformation combination distance.
        fix_crop(int): Whether to fix the cutting start point.
        allow_duplication(int): Whether to allow duplicate candidate crop starting points.
        more_fix_crop(int): Whether to allow more cutting starting points.
    """
    def __init__(
            self,
            target_size,  # NOTE: named target size now, but still pass short size in it!
            scales=None,
            max_distort=1,
            fix_crop=True,
            allow_duplication=False,
            more_fix_crop=True,
            ):

        self.target_size = target_size
        self.scales = scales if scales else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.allow_duplication = allow_duplication
        self.more_fix_crop = more_fix_crop

    def __call__(self, results):
        """
        Performs MultiScaleCrop operations.
        Args:
            imgs: List where wach item is a PIL.Image.
            XXX:
        results:
        """
        imgs = results['imgs']

        input_size = [self.target_size, self.target_size]

        im_size = imgs[0].size

        # get random crop offset
        def _sample_crop_size(im_size):
            image_w, image_h = im_size[0], im_size[1]

            base_size = min(image_w, image_h)
            crop_sizes = [int(base_size * x) for x in self.scales]
            crop_h = [
                input_size[1] if abs(x - input_size[1]) < 3 else x
                for x in crop_sizes
            ]
            crop_w = [
                input_size[0] if abs(x - input_size[0]) < 3 else x
                for x in crop_sizes
            ]

            pairs = []
            for i, h in enumerate(crop_h):
                for j, w in enumerate(crop_w):
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))
            crop_pair = random.choice(pairs)
            if not self.fix_crop:
                w_offset = random.randint(0, image_w - crop_pair[0])
                h_offset = random.randint(0, image_h - crop_pair[1])
            else:
                w_step = (image_w - crop_pair[0]) / 4
                h_step = (image_h - crop_pair[1]) / 4

                ret = list()
                ret.append((0, 0))  # upper left
                if self.allow_duplication or w_step != 0:
                    ret.append((4 * w_step, 0))  # upper right
                if self.allow_duplication or h_step != 0:
                    ret.append((0, 4 * h_step))  # lower left
                if self.allow_duplication or (h_step != 0 and w_step != 0):
                    ret.append((4 * w_step, 4 * h_step))  # lower right
                if self.allow_duplication or (h_step != 0 or w_step != 0):
                    ret.append((2 * w_step, 2 * h_step))  # center

                if self.more_fix_crop:
                    ret.append((0, 2 * h_step))  # center left
                    ret.append((4 * w_step, 2 * h_step))  # center right
                    ret.append((2 * w_step, 4 * h_step))  # lower center
                    ret.append((2 * w_step, 0 * h_step))  # upper center

                    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

                w_offset, h_offset = random.choice(ret)

            return crop_pair[0], crop_pair[1], w_offset, h_offset

        crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in imgs
        ]
        ret_img_group = [
            Image.fromarray(
                cv2.resize(np.asarray(img),
                            dsize=(input_size[0], input_size[1]),
                            interpolation=cv2.INTER_LINEAR))
            for img in crop_img_group
        ]
        results['imgs'] = ret_img_group
        return results


class RandomFlip(object):
    """
    Random Flip images.
    Args:
        p(float): Random flip images with the probability p.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        """
        Performs random flip operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            flip_imgs: List where each item is a PIL.Image after random flip.
        """
        imgs = results['imgs']
        v = random.random()
        if v < self.p:
            results['imgs'] = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs
            ]
        else:
            results['imgs'] = imgs
        return results

class Image2Array(object):
    """
    transfer PIL.Image to Numpy array and transpose dimensions from 'dhwc' to 'dchw'.
    Args:
        transpose: whether to transpose or not, default True, False for slowfast.
    """
    def __init__(self, transpose=True):
        self.transpose = transpose

    def __call__(self, results):
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        imgs = results['imgs']
        np_imgs = (np.stack(imgs)).astype('float32')
        if self.transpose:
            np_imgs = np_imgs.transpose(0, 3, 1, 2)  # tchw
        results['imgs'] = np_imgs
        return results

class Normalization(object):
    """
    Normalization.
    Args:
        mean(Sequence[float]): mean values of different channels.
        std(Sequence[float]): std values of different channels.
        tensor_shape(list): size of mean, default [3,1,1]. For slowfast, [1,1,1,3]
    """
    def __init__(self, mean, std, tensor_shape=[3, 1, 1]):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}')
        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')
        self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
        self.std = np.array(std).reshape(tensor_shape).astype(np.float32)

    def __call__(self, results):
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """
        imgs = results['imgs']
        norm_imgs = imgs / 255.0
        norm_imgs -= self.mean
        norm_imgs /= self.std
        results['imgs'] = norm_imgs
        return results

class CenterCrop(object):
    """
    Center crop images.
    Args:
        target_size(int): Center crop a square with the target_size from an image.
        do_round(bool): Whether to round up the coordinates of the upper left corner of the cropping area. default: True
    """
    def __init__(self, target_size, do_round=True):
        self.target_size = target_size
        self.do_round = do_round

    def __call__(self, results):
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            ccrop_imgs: List where each item is a PIL.Image after Center crop.
        """
        imgs = results['imgs']
        ccrop_imgs = []
        for img in imgs:
            w, h = img.size
            th, tw = self.target_size, self.target_size
            assert (w >= self.target_size) and (h >= self.target_size), \
                "image width({}) and height({}) should be larger than crop size".format(
                    w, h, self.target_size)
            x1 = int(round((w - tw) / 2.0)) if self.do_round else (w - tw) // 2
            y1 = int(round((h - th) / 2.0)) if self.do_round else (h - th) // 2
            ccrop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = ccrop_imgs
        return results


class Compose(object):
    """
    Composes several pipelines(include decode func, sample func, and transforms) together.

    Note: To deal with ```list``` type cfg temporaray, like:

        transform:
            - Crop: # A list
                attribute: 10
            - Resize: # A list
                attribute: 20

    every key of list will pass as the key name to build a module.
    XXX: will be improved in the future.

    Args:
        pipelines (list): List of transforms to compose.
    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.
    """
    
    def __init__(self, train_mode=False):
        # assert isinstance(pipelines, Sequence)
        self.pipelines = list()
        self.pipelines.append(VideoDecoder())
        if train_mode:
            self.pipelines.append(Sampler(num_seg=8, seg_len=1, valid_mode=False, select_left=True))
            self.pipelines.append(MultiScaleCrop(target_size=224, allow_duplication=True))
            self.pipelines.append(RandomFlip())
        else:
            self.pipelines.append(Sampler(num_seg=8, seg_len=1, valid_mode=True, select_left=True))
            self.pipelines.append(Scale(short_size=256, fixed_ratio=False))
            self.pipelines.append(CenterCrop(target_size=224))
        self.pipelines.append(Image2Array())
        self.pipelines.append(Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    def __call__(self, data):
        # 将传入的 data 依次经过 pipelines 中对象处理
        for p in self.pipelines:
            try:
                data = p(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                            "{} and stack:\n{}".format(p, e, str(stack_info)))
                raise e
        return data


class VideoDataset(paddle.io.Dataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.
       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:
       .. code-block:: txt
           path/000.mp4 1
           path/001.mp4 1
           path/002.mp4 2
           path/003.mp4 2
       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           **kwargs: Keyword arguments for ```BaseDataset```.
    """
    def __init__(self, file_path, pipeline, num_retries=5, suffix='', test_mode=False):
        super().__init__()
        self.file_path = file_path
        self.pipeline = pipeline
        self.num_retries = num_retries
        self.suffix = suffix
        self.info = self.load_file()
        self.test_mode = test_mode
        
    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                filename, labels = line_split
                filename = filename + self.suffix
                info.append(dict(filename=filename, labels=int(labels)))
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                if ir < self.num_retries - 1:
                    print(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'], np.array([results['labels']])

    def prepare_test(self, idx):
        """TEST. Prepare the data for test given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                if ir < self.num_retries - 1:
                    print(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'], np.array([results['labels']])

    def __len__(self):
        """get the size of the dataset."""
        return len(self.info)

    def __getitem__(self, idx):
        """ Get the sample for either training or testing given index"""
        if self.test_mode:
            return self.prepare_test(idx)
        else:
            return self.prepare_train(idx)        
