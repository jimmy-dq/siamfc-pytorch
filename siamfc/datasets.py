from __future__ import absolute_import, division

import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

__all__ = ['Pair']


class Pair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1, noise=None):
        super(Pair, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))
        self.return_meta = getattr(seqs, 'return_meta', False)
        self.noise = noise
        self.epsilon = 8 / 255
    
    def paste_noise_to_img(self, img, bbox, index, mode='bicubic'):
        noise_tensor = self.noise[index].unsqueeze(0)

        z_top_left_x =  int(bbox[0])
        z_top_left_y = int(bbox[1])
        z_bottom_right_x = int(bbox[0] + bbox[2])
        z_bottom_right_y = int(bbox[1] + bbox[3])

        H, W, C = img.shape
        if z_top_left_x < 0 or z_top_left_y < 0 or z_bottom_right_x > W or z_bottom_right_y > H or (z_bottom_right_x - z_top_left_x) <= 0 or (z_bottom_right_y - z_top_left_y) <= 0:
            return None
        else:
            new_noise = torch.nn.functional.interpolate(
                noise_tensor, size=(z_bottom_right_y - z_top_left_y, z_bottom_right_x - z_top_left_x), mode=mode, align_corners=False)
            new_noise = new_noise.clamp_(-self.epsilon, self.epsilon).mul(255).squeeze(0).permute(1, 2, 0).to('cpu').numpy() #H, W, C
            img = img.astype(np.float32)
            img[z_top_left_y:z_bottom_right_y, z_top_left_x:z_bottom_right_x, :] += new_noise
            img = np.clip(img, a_min=0, a_max=255)
            img = img.astype(np.uint8)
            return img

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        
        # filter out noisy frames
        val_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios)
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        rand_z, rand_x = self._sample_pair(val_indices)

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        if self.noise is not None:

            z = self.paste_noise_to_img(z, box_z, index)
            x = self.paste_noise_to_img(x, box_x, index)
            if z is None or x is None:
                index = np.random.choice(len(self))
                return self.__getitem__(index)

        # z = torch.from_numpy(z)
        # x = torch.from_numpy(x)

        item = (z, x, box_z, box_x, None, None)
        if self.transforms is not None:
            item = self.transforms(*item)
            # adap_box_z, adap_box_x = self.transforms(*item)
        
        return item, index, torch.from_numpy(box_z), torch.from_numpy(box_x)
    
    def __len__(self):
        return len(self.indices) * self.pairs_per_seq
    
    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x
    
    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)
        
        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices
