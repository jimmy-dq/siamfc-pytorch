from __future__ import absolute_import, division

import cv2
import numpy as np
import numbers
from numpy.lib.function_base import interp
import torch

from . import ops
import torchvision.transforms.functional as F

__all__ = ['SiamFCTransforms']


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target_size):
        for t in self.transforms:
            img, target_size = t(img, target_size)
        return img, target_size


class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch
    
    def __call__(self, img, target_size):
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        # interp = np.random.choice([
        #     'bilinear',
        #     'bicubic',
        #     'area',
        #     'nearest',
        # ])     
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),
            round(img.shape[0] * scale)) # w, h
        # out_size = (
        #     round(img.shape[0] * scale),
        #     round(img.shape[1] * scale)) #H, W
        # return cv2.resize(img, out_size, interpolation=interp)

        target_size[0] = out_size[0] // 2 #cx
        target_size[1] = out_size[1] // 2 #cy
        target_size[2:] *= scale
        
        # return F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=out_size, mode=interp).squeeze(0).permute(1, 2, 0), F.interpolate(img_noise.permute(2, 0, 1).unsqueeze(0), size=out_size, mode=interp).squeeze(0).permute(1, 2, 0), target_size

        return cv2.resize(img, out_size, interpolation=interp), target_size


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def copy_make_border(self, input, top, bottom, left, right, border_type='constant', value=0):
        """
        类似于cv2.copyMakeBorder的功能，实现在输入张量的周围添加指定数量的边界。

        Args:
            input (torch.Tensor): 输入张量。
            top (int): 顶部边界的大小。
            bottom (int): 底部边界的大小。
            left (int): 左侧边界的大小。
            right (int): 右侧边界的大小。
            border_type (str): 边界类型，支持'constant'（常数填充）和'replicate'（复制边界值）。
            value (int): 当border_type为'constant'时，指定要填充的常数值。

        Returns:
            torch.Tensor: 填充后的张量。
        """
        if border_type == 'constant':
            return F.pad(input, (left, right, top, bottom), mode='constant', value=value)
        elif border_type == 'replicate':
            return F.pad(input, (left, right, top, bottom), mode='replicate')
        else:
            raise ValueError("Unsupported border type. Supported types are 'constant' and 'replicate'.")
        
    def __call__(self, img, target_size):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.) #top_left_y
        j = round((w - tw) / 2.) #top_left_x

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            # avg_color = torch.mean(img_noise, dim=(0, 1), dtype=torch.float32)
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            # img = self.copy_make_border(img, npad, npad, npad, npad, border_type='constant', value=avg_color.mean().item())
            # img_noise = self.copy_make_border(img_noise, npad, npad, npad, npad, border_type='constant', value=avg_color.mean().item())
            i += npad
            j += npad
        
        target_size[0] = target_size[0] + npad - j
        target_size[1] = target_size[1] + npad - i

        return img[i:i + th, j:j + tw], target_size


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img, target_size):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        target_size[0] = target_size[0] - j
        target_size[1] = target_size[1] - i 
        return img[i:i + th, j:j + tw], target_size


class ToTensor(object):

    def __call__(self, img, target_size):
        return (torch.from_numpy(img) / 255.0).float().permute((2, 0, 1)), target_size
        # return (img / 255.0).float().permute((2, 0, 1)), (img_noise / 255.0).float().permute((2, 0, 1)), target_size.float()  #C, H, W


class SiamFCTransforms(object):

    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            CenterCrop(exemplar_sz),
            ToTensor()])
        self.transforms_x = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            ToTensor()])
    
    def __call__(self, z, x, box_z, box_x, z_noise, x_noise):
        z, z_scale = self._crop(z, box_z, self.instance_sz, z_noise) 
        x, x_scale = self._crop(x, box_x, self.instance_sz, x_noise)
        target_z_bbox = torch.FloatTensor([z.shape[1]//2, z.shape[0]//2, z_scale[1], z_scale[0]]) #cxcy,w,h
        target_x_bbox = torch.FloatTensor([x.shape[1]//2, x.shape[0]//2, x_scale[1], x_scale[0]]) #cxcy,w,h
        z, adap_box_z = self.transforms_z(z, target_z_bbox)
        x, adap_box_x = self.transforms_x(x, target_x_bbox)
        return z, x, adap_box_z, adap_box_x
    
    def _crop(self, img, box, out_size, img_noise):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz

        if size < target_sz[0]:
            target_sz[0] = size
        if size < target_sz[1]:
            target_sz[1] = size


        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        # avg_color = torch.mean(img_noise, dim=(0, 1), dtype=torch.float32)
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        # interp = np.random.choice([
        #     'bilinear',
        #     'bicubic',
        #     'area',
        #     'nearest',
        # ])   
        patch, img_h_scale, img_w_scale = ops.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)

        # patch_noise, noise_img_h_scale, noise_img_w_scale = ops.crop_and_resize(
        #     img_noise, center, size, out_size,
        #     border_value=avg_color, interp=interp)
        
        # return patch, patch_noise, np.array([img_h_scale, img_w_scale]) * target_sz, np.array([noise_img_h_scale, noise_img_w_scale]) * target_sz
        return patch, np.array([img_h_scale, img_w_scale]) * target_sz
