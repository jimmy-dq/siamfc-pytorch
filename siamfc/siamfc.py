from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from torch.utils.tensorboard import SummaryWriter
from .toolbox import PerturbationTool
import torchvision
from tqdm import tqdm

__all__ = ['TrackerSiamFC']

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

        self.writer = SummaryWriter()

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)[0]
        
        # exemplar features
        z = (torch.from_numpy(z) / 255.0).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color)[0] for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = (torch.from_numpy(x) / 255.0).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    # def train_step(self, epoch, it, batch, temp_noise, search_noise,  backward=True):
    def train_step(self, epoch, it, batch, unified_noise, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0][0].to(self.device, non_blocking=self.cuda)
        x = batch[0][1].to(self.device, non_blocking=self.cuda)

        seq_index = batch[1]

        z_adap_box = batch[0][2] #cxcywh
        x_adap_box = batch[0][3] #cxcywh
        z_box = batch[2] #xywh
        x_box = batch[3] #xywh

        if it == 0:
            z_box_show = torch.zeros_like(z_adap_box)
            z_box_show[:,0] = z_adap_box[:,0] - z_adap_box[:,2] / 2
            z_box_show[:,1] = z_adap_box[:,1] - z_adap_box[:,3] / 2
            z_box_show[:,2] = z_adap_box[:,0] + z_adap_box[:,2] / 2
            z_box_show[:,3] = z_adap_box[:,1] + z_adap_box[:,3] / 2

            x_box_show = torch.zeros_like(x_adap_box)
            x_box_show[:,0] = x_adap_box[:,0] - x_adap_box[:,2] / 2
            x_box_show[:,1] = x_adap_box[:,1] - x_adap_box[:,3] / 2
            x_box_show[:,2] = x_adap_box[:,0] + x_adap_box[:,2] / 2
            x_box_show[:,3] = x_adap_box[:,1] + x_adap_box[:,3] / 2


            z_show_int = (z.cpu()*255).type(torch.uint8)
            z_bbox_attached = [torchvision.utils.draw_bounding_boxes(z_show_int[img_index], z_box_show[img_index].unsqueeze(0)) for img_index in range(z_show_int.shape[0])]
            z_bbox_attached = torch.stack(z_bbox_attached, dim=0)

            x_show_int = (x.cpu()*255).type(torch.uint8)
            x_bbox_attached = [torchvision.utils.draw_bounding_boxes(x_show_int[img_index], x_box_show[img_index].unsqueeze(0)) for img_index in range(x_show_int.shape[0])]
            x_bbox_attached = torch.stack(x_bbox_attached, dim=0)

            img_grid_z = torchvision.utils.make_grid(z_bbox_attached)
            img_grid_x = torchvision.utils.make_grid(x_bbox_attached)
            self.writer.add_image('epoch: %d noisy_template' %(epoch), img_grid_z)
            self.writer.add_image('epoch: %d noisy_search' %(epoch), img_grid_x)

        # if it == 0:
        #     img_grid_z = torchvision.utils.make_grid(z.cpu())
        #     img_grid_x = torchvision.utils.make_grid(x.cpu())
        #     self.writer.add_image('epoch: %d template' %(epoch), img_grid_z)
        #     self.writer.add_image('epoch: %d search' %(epoch), img_grid_x)
        
        # x1 = 239 // 2 - 127 // 2
        # x2 = 239 // 2 + 127 // 2 + 1

        # for i in range(seq_index.shape[0]):
        #     z[i] = torch.clamp(z[i] + unified_noise[seq_index[i]], 0, 1)
        #     # print(x.shape)
        #     # print(search_noise.shape)
        #     # x[i] = torch.clamp(x[i] + search_noise[seq_index[i]], 0, 1)
        #     x[i][:, x1:x2, x1:x2] = torch.clamp(x[i][:, x1:x2, x1:x2] + unified_noise[seq_index[i]], 0, 1)

        # if it == 0:
        #     img_grid_z = torchvision.utils.make_grid(z.cpu())
        #     img_grid_x = torchvision.utils.make_grid(x.cpu())
        #     self.writer.add_image('epoch: %d noisy_template' %(epoch), img_grid_z)
        #     self.writer.add_image('epoch: %d noisy_search' %(epoch), img_grid_x)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, is_random_noise=False, val_seqs=None,
                   save_dir='pretrained_unified_ues'):
        
        # temp_noise = torch.load('/home/qiangqwu/Project/siamfc-pytorch/temp_perturbation.pt', map_location=device)
        # search_noise = torch.load('/home/qiangqwu/Project/siamfc-pytorch/search_perturbation.pt', map_location=device)

        # unified_noise = torch.load('/home/qiangqwu/Project/siamfc-pytorch/unified_perturbation.pt', map_location=device)
        if is_random_noise:
            print('create random noise')
            unified_noise = torch.FloatTensor(9335, 3, 127, 127).uniform_(-8 / 255, 8 / 255).to(device)
        elif not is_random_noise:
            print('loading')
            unified_noise = torch.load('/home/qiangqwu/Project/siamfc-pytorch/unified_perturbation.pt', map_location=device)

        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms, noise=unified_noise.data.cpu())
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                # loss = self.train_step(epoch, it, batch, temp_noise, search_noise, backward=True)
                loss = self.train_step(epoch, it, batch, unified_noise, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
            self.writer.add_scalar("loss", loss, epoch)
        self.writer.close()
    
    # def paste_noise_to_img(self, img, box_list, noise_tensor, mode='bicubic'):
        
    #     for box in box_list:   

    #     z_top_left_x =  int(bbox[0])
    #     z_top_left_y = int(bbox[1])
    #     z_bottom_right_x = int(bbox[0] + bbox[2])
    #     z_bottom_right_y = int(bbox[1] + bbox[3])

        
    #     if z_top_left_x < 0 or z_top_left_y < 0 or z_bottom_right_x > W or z_bottom_right_y > H or (z_bottom_right_x - z_top_left_x) <= 0 or (z_bottom_right_y - z_top_left_y) <= 0:
    #         return None
    #     else:
    #         new_noise = torch.nn.functional.interpolate(
    #             noise_tensor, size=(z_bottom_right_y - z_top_left_y, z_bottom_right_x - z_top_left_x), mode=mode, align_corners=False)
    #         new_noise = torch.clamp(new_noise, -self.epsilon, self.epsilon).squeeze(0).permute(1, 2, 0) #H, W, C
    #         img_noise = img / 255.0
    #         # img_noise = torch.from_numpy(img_noise)
    #         img_noise[z_top_left_y:z_bottom_right_y, z_top_left_x:z_bottom_right_x, :] += new_noise.cpu().data.numpy()
    #         img_noise = np.clip(img_noise, 0, 1.0) * 255.0
    #         return img_noise

    # def train_step_pertubation(self, epoch, it, batch, temp_noise, search_noise, backward=True):
    def train_step_pertubation(self, epoch, it, batch, unified_noise, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0][0].to(self.device, non_blocking=self.cuda)
        x = batch[0][1].to(self.device, non_blocking=self.cuda)

        z_adap_box = batch[0][2] #cxcywh
        x_adap_box = batch[0][3] #cxcywh
        z_box = batch[2] #xywh
        x_box = batch[3] #xywh

        seq_index = batch[1]
        
        if it == 0:
            img_grid_z = torchvision.utils.make_grid(z.cpu())
            img_grid_x = torchvision.utils.make_grid(x.cpu())
            self.writer.add_image('epoch: %d template' %(epoch), img_grid_z)
            self.writer.add_image('epoch: %d search' %(epoch), img_grid_x)


        for i in range(seq_index.shape[0]):
            z_bbox_list = [z_box[i], z_adap_box[i]]  #xywh, cxcywh
            x_bbox_list = [x_box[i], x_adap_box[i]]
            noise_tensor = unified_noise[seq_index[i]].unsqueeze(0)
            # the original annotation
            upsample_noise_z = torch.nn.functional.interpolate(
                noise_tensor, size=(int(z_bbox_list[0][3]), int(z_bbox_list[0][2])), mode='bicubic', align_corners=False)
            # the final annotation
            upsample_noise_z = torch.nn.functional.interpolate(
                upsample_noise_z, size=(int(z_bbox_list[1][3]), int(z_bbox_list[1][2])), mode='bicubic', align_corners=False) #C, H, W
            upsample_noise_z = upsample_noise_z.squeeze(0).to(self.device)
            start_y, start_x = int(z_bbox_list[1][1] - upsample_noise_z.shape[1] / 2), int(z_bbox_list[1][0] - upsample_noise_z.shape[2] / 2)
            z[i][:, start_y:(start_y+upsample_noise_z.shape[1]), start_x:(start_x+upsample_noise_z.shape[2])] += upsample_noise_z

            # the original annotation
            upsample_noise_x = torch.nn.functional.interpolate(
                noise_tensor, size=(int(x_bbox_list[0][3]), int(x_bbox_list[0][2])), mode='bicubic', align_corners=False)
            # the final annotation
            upsample_noise_x = torch.nn.functional.interpolate(
                upsample_noise_x, size=(int(x_bbox_list[1][3]), int(x_bbox_list[1][2])), mode='bicubic', align_corners=False) #C, H, W
            upsample_noise_x = upsample_noise_x.squeeze(0).to(self.device)
            start_y, start_x = int(x_bbox_list[1][1] - upsample_noise_x.shape[1] / 2), int(x_bbox_list[1][0] - upsample_noise_x.shape[2] / 2)
            x[i][:, start_y:(start_y+upsample_noise_x.shape[1]), start_x:(start_x+upsample_noise_x.shape[2])] += upsample_noise_x
        
        x = torch.clamp(x, 0, 1)
        z = torch.clamp(z, 0, 1)


            # # # print(x.shape)
            # # # print(search_noise.shape)
            # # x[i] = torch.clamp(x[i] + search_noise[seq_index[i]], 0, 1)

            # z[i] = torch.clamp(z[i] + unified_noise[seq_index[i]], 0, 1)
            # # print(x.shape)
            # # print(search_noise.shape)
            # x[i][:, x1:x2, x1:x2] = torch.clamp(x[i][:, x1:x2, x1:x2] + unified_noise[seq_index[i]], 0, 1)

        if it == 0:
            z_box_show = torch.zeros_like(z_adap_box)
            z_box_show[:,0] = z_adap_box[:,0] - z_adap_box[:,2] / 2
            z_box_show[:,1] = z_adap_box[:,1] - z_adap_box[:,3] / 2
            z_box_show[:,2] = z_adap_box[:,0] + z_adap_box[:,2] / 2
            z_box_show[:,3] = z_adap_box[:,1] + z_adap_box[:,3] / 2

            x_box_show = torch.zeros_like(x_adap_box)
            x_box_show[:,0] = x_adap_box[:,0] - x_adap_box[:,2] / 2
            x_box_show[:,1] = x_adap_box[:,1] - x_adap_box[:,3] / 2
            x_box_show[:,2] = x_adap_box[:,0] + x_adap_box[:,2] / 2
            x_box_show[:,3] = x_adap_box[:,1] + x_adap_box[:,3] / 2


            z_show_int = (z.cpu()*255).type(torch.uint8)
            z_bbox_attached = [torchvision.utils.draw_bounding_boxes(z_show_int[img_index], z_box_show[img_index].unsqueeze(0)) for img_index in range(z_show_int.shape[0])]
            z_bbox_attached = torch.stack(z_bbox_attached, dim=0)

            x_show_int = (x.cpu()*255).type(torch.uint8)
            x_bbox_attached = [torchvision.utils.draw_bounding_boxes(x_show_int[img_index], x_box_show[img_index].unsqueeze(0)) for img_index in range(x_show_int.shape[0])]
            x_bbox_attached = torch.stack(x_bbox_attached, dim=0)

            img_grid_z = torchvision.utils.make_grid(z_bbox_attached)
            img_grid_x = torchvision.utils.make_grid(x_bbox_attached)
            self.writer.add_image('epoch: %d noisy_template' %(epoch), img_grid_z)
            self.writer.add_image('epoch: %d noisy_search' %(epoch), img_grid_x)

        for param in self.net.parameters():
                param.requires_grad = True

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    def train_over_pertubation(self, seqs, val_seqs=None,
                   save_dir='pretrained'):

        noise_generator = PerturbationTool(epsilon=8 / 255,
                                           num_steps=20,
                                           step_size=0.8 / 255, cfg=self.cfg)
        # temp_noise = noise_generator.random_noise(noise_shape=[9335, 3, 127, 127])
        # search_noise = noise_generator.random_noise(noise_shape=[9335, 3, 239, 239])

        unified_noise = noise_generator.random_noise(noise_shape=[9335, 3, 127, 127])

        # unified_noise = noise_generator.random_noise(noise_shape=[1, 3, 127, 127])

        # temp_noise = noise_generator.random_noise(noise_shape=[1, 3, 127, 127])
        # search_noise = noise_generator.random_noise(noise_shape=[1, 3, 239, 239])

        # torch.save(temp_noise, os.path.join('./', 'temp_perturbation.pt'))
        # torch.save(search_noise, os.path.join('./', 'search_perturbation.pt'))

        # torch.save(unified_noise, os.path.join('./', 'unified_perturbation.pt'))

        # torch.save(unified_noise, os.path.join('./', 'unified_arbitrary_shape_perturbation_127.pt'))
        torch.save(unified_noise, os.path.join('./', 'unified_arbitrary_shape_perturbation_127_v2.pt'))
        
        # define the noise generator
        # generate random noise initially
        # run k_step for siamfc model training with noises
        # add noises to z and x independently
        # run the training, detach gradients in noises

        # run more iterations for noise optimization
        # use the toolbox method for optimziation

        # no validaiton: repeat

        # tensorboard for visualization

        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms, noise=None)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers, #0
            pin_memory=self.cuda,
            drop_last=True)
        dataloader_noise = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers, #0
            pin_memory=self.cuda,
            drop_last=False)
        
        

        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)
            # loop over dataloader
            for it, batch in enumerate(dataloader): #batch[0][0]: 8, 3, 127, 127; batch[1]: tensor([6854, 8650, 7166,  726,  946,  561, 7875, 8989]);
                # loss = self.train_step_pertubation(epoch, it, batch, temp_noise, search_noise, backward=True)
                loss = self.train_step_pertubation(epoch, it, batch, unified_noise, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

                # # for debugging
                # if it==0:
                #     break
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
            self.writer.add_scalar("loss", loss, epoch)

            # Search For Noise
            idx = 0
            for it, batch in tqdm(enumerate(dataloader_noise), total=len(dataloader_noise)):
                temp_images, search_images = batch[0][0].to(self.device), batch[0][1].to(self.device)
                seq_idxs = batch[1]

                # box annotations
                z_adap_box = batch[0][2] #cxcywh
                x_adap_box = batch[0][3] #cxcywh
                z_box = batch[2] #xywh
                x_box = batch[3] #xywh

                # Add Sample-wise Noise to each sample
                # batch_temp_noise, batch_search_noise = [], []
                batch_unified_noise = []
                batch_start_idx = idx
                for i, (temp_image, search_image) in enumerate(zip(temp_images, search_images)):
                    # sample_temp_noise = temp_noise[seq_idxs[i]]
                    # sample_search_noise = search_noise[seq_idxs[i]]
                    # sample_temp_noise = sample_temp_noise.to(self.device)
                    # sample_search_noise = sample_search_noise.to(self.device)
                    # batch_temp_noise.append(sample_temp_noise)
                    # batch_search_noise.append(sample_search_noise)
                    sample_unified_noise = unified_noise[seq_idxs[i]]
                    sample_unified_noise = sample_unified_noise.to(self.device)
                    batch_unified_noise.append(sample_unified_noise)
                    idx += 1
                
                # Update sample-wise perturbation
                self.net.eval()
                for param in self.net.parameters():
                    param.requires_grad = False
                # batch_temp_noise = torch.stack(batch_temp_noise).to(self.device)
                # batch_search_noise = torch.stack(batch_search_noise).to(self.device)
                batch_unified_noise = torch.stack(batch_unified_noise).to(self.device)

                # perturb_temp_img, eta_temp, perturb_search_img, eta_search = noise_generator.min_min_attack(temp_images, search_images, self.net, self.criterion, random_temp_noise=batch_temp_noise, random_search_noise=batch_search_noise)
                perturb_temp_img, eta_temp, perturb_search_img = noise_generator.min_min_attack(temp_images, search_images, self.net, self.criterion, z_adap_box, x_adap_box, z_box, x_box, unified_noise=batch_unified_noise)
                for j, delta in enumerate(eta_temp):
                    if torch.is_tensor(unified_noise):
                        unified_noise[seq_idxs[j]] = delta.detach().cpu().clone()
                    else:
                        unified_noise[seq_idxs[j]] = delta.detach().cpu().numpy()
                # for j, delta in enumerate(eta_search):
                #     if torch.is_tensor(search_noise):
                #         search_noise[seq_idxs[j]] = delta.detach().cpu().clone()
                #     else:
                #         search_noise[seq_idxs[j]] = delta.detach().cpu().numpy()

            # torch.save(temp_noise, os.path.join('./', 'temp_perturbation.pt'))
            # torch.save(search_noise, os.path.join('./', 'search_perturbation.pt'))

            torch.save(unified_noise, os.path.join('./', 'unified_arbitrary_shape_perturbation_127_v2.pt'))
            
        self.writer.close()
    


    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
