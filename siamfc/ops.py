from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

def copy_make_border(input, top, bottom, left, right, border_type='constant', value=0):
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

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=None):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)
        # img = img.permute(2, 0, 1).unsqueeze(0) #bs, c, h, w
        # img = copy_make_border(img, npad, npad, npad, npad, border_type='constant', value=border_value.mean().item()).squeeze(0).permute(1, 2, 0) #H, W, C

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]] #H, W, C

    H_scale = out_size / patch.shape[0]
    W_scale = out_size / patch.shape[1]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)
    # patch = F.interpolate(patch.permute(2, 0, 1).unsqueeze(0), size=(out_size, out_size), mode=interp).squeeze(0).permute(1, 2, 0) #H, W, C

    

    return patch, H_scale, W_scale
