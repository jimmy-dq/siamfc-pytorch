import numpy as np
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725, cfg=None):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        self.cfg = cfg
        self.device = device
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon) #.to(device)
        return random_noise


    # temp_images, search_images, self.net, random_temp_noise=batch_temp_noise, random_search_noise=batch_search_noise

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
        self.labels = torch.from_numpy(labels).to(device).float()
        
        return self.labels
    
    def petur_img_w_noise(self, temp_images, search_images, z_adap_box, x_adap_box, z_box, x_box, noise):
        # temp_images_noise = Variable(temp_images.data, requires_grad=False)
        temp_images_noise = temp_images.detach().clone()
        search_images_noise = search_images.detach().clone()
        # search_images_noise = Variable(search_images.data, requires_grad=False)
        for i in range(temp_images.shape[0]):
            z_bbox_list = [z_box[i], z_adap_box[i]]  #xywh, cxcywh
            x_bbox_list = [x_box[i], x_adap_box[i]]
            noise_tensor = noise[i].unsqueeze(0)
            # the original annotation
            upsample_noise_z = torch.nn.functional.interpolate(
                noise_tensor, size=(int(z_bbox_list[0][3]), int(z_bbox_list[0][2])), mode='bicubic', align_corners=False)
            # the final annotation
            upsample_noise_z = torch.nn.functional.interpolate(
                upsample_noise_z, size=(int(z_bbox_list[1][3]), int(z_bbox_list[1][2])), mode='bicubic', align_corners=False) #C, H, W
            upsample_noise_z = upsample_noise_z.squeeze(0).to(self.device)
            start_y, start_x = int(z_bbox_list[1][1] - upsample_noise_z.shape[1] / 2), int(z_bbox_list[1][0] - upsample_noise_z.shape[2] / 2)
            temp_images_noise[i][:, start_y:(start_y+upsample_noise_z.shape[1]), start_x:(start_x+upsample_noise_z.shape[2])] += upsample_noise_z

            # the original annotation
            upsample_noise_x = torch.nn.functional.interpolate(
                noise_tensor, size=(int(x_bbox_list[0][3]), int(x_bbox_list[0][2])), mode='bicubic', align_corners=False)
            # the final annotation
            upsample_noise_x = torch.nn.functional.interpolate(
                upsample_noise_x, size=(int(x_bbox_list[1][3]), int(x_bbox_list[1][2])), mode='bicubic', align_corners=False) #C, H, W
            start_y, start_x = int(x_bbox_list[1][1] - upsample_noise_x.shape[1] / 2), int(x_bbox_list[1][0] - upsample_noise_x.shape[2] / 2)
            upsample_noise_x = upsample_noise_x.squeeze(0).to(self.device)
            search_images_noise[i][:, start_y:(start_y+upsample_noise_x.shape[1]), start_x:(start_x+upsample_noise_x.shape[2])] += upsample_noise_x
        temp_images_noise = torch.clamp(temp_images_noise, 0, 1)
        search_images_noise = torch.clamp(search_images_noise, 0, 1)
        
        return temp_images_noise, search_images_noise

    def min_min_attack(self, temp_images, search_images, model, criterion, z_adap_box, x_adap_box, z_box, x_box, random_temp_noise=None, random_search_noise=None, unified_noise=None):


        # perturb_temp_img = Variable(temp_images.data + unified_noise, requires_grad=True)
        # perturb_temp_img = Variable(torch.clamp(perturb_temp_img, 0, 1), requires_grad=True)

        # # perturb_search_img = Variable(search_images.data + random_search_noise, requires_grad=True)
        # # perturb_search_img = Variable(torch.clamp(perturb_search_img, 0, 1), requires_grad=True)

        # mask = torch.zeros((search_images.shape[0], 3, 239, 239)).to(self.device)
        # mask[:,:,56:183,56:183] = unified_noise
        # perturb_search_img = Variable(search_images.data + mask, requires_grad=True)
        # perturb_search_img = Variable(torch.clamp(perturb_search_img, 0, 1), requires_grad=True)

        # # perturb_search_img = Variable(search_images.data + random_search_noise, requires_grad=True)
        # # perturb_search_img = Variable(torch.clamp(perturb_search_img, 0, 1), requires_grad=True)

        # eta_temp = unified_noise
        # # eta_search = random_search_noise

        optimized_noise = Variable(unified_noise.data, requires_grad=True)

        perturb_temp_img, perturb_search_img = self.petur_img_w_noise(temp_images, search_images, z_adap_box, x_adap_box, z_box, x_box, optimized_noise)
        

        for _ in range(self.num_steps):
            # opt = torch.optim.SGD([perturb_temp_img,perturb_search_img], lr=1e-3)
            opt = torch.optim.SGD([optimized_noise], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            
            responses = model(perturb_temp_img, perturb_search_img)
            labels = self._create_labels(responses.size())
            loss = criterion(responses, labels)
            
            # perturb_temp_img.retain_grad()
            # perturb_search_img.retain_grad()
            optimized_noise.retain_grad()
            loss.backward()

            eta_noise = self.step_size * optimized_noise.grad.data.sign() * (-1)

            optimized_noise = Variable(torch.clamp(optimized_noise.data + eta_noise, -self.epsilon, self.epsilon), requires_grad=True)

            perturb_temp_img, perturb_search_img = self.petur_img_w_noise(temp_images, search_images, z_adap_box, x_adap_box, z_box, x_box, optimized_noise)


            # # eta_temp = self.step_size * perturb_temp_img.grad.data.sign() * (-1)
            # # eta_search = self.step_size * perturb_search_img.grad.data.sign() * (-1)
            # # perturb_temp_img = Variable(perturb_temp_img.data + eta_temp, requires_grad=True)
            # # perturb_search_img = Variable(perturb_search_img.data + eta_search, requires_grad=True)
            # # eta_temp = torch.clamp(perturb_temp_img.data - temp_images.data, -self.epsilon, self.epsilon)
            # # eta_search = torch.clamp(perturb_search_img.data - search_images.data, -self.epsilon, self.epsilon)
            # # perturb_temp_img = Variable(temp_images.data + eta_temp, requires_grad=True)
            # # perturb_temp_img = Variable(torch.clamp(perturb_temp_img, 0, 1), requires_grad=True)
            # # perturb_search_img = Variable(search_images.data + eta_search, requires_grad=True)
            # # perturb_search_img = Variable(torch.clamp(perturb_search_img, 0, 1), requires_grad=True)

            # eta_temp = self.step_size * (perturb_temp_img.grad.data + perturb_search_img.grad.data[:,:,56:183,56:183]).sign() * (-1)
            # perturb_temp_img = Variable(perturb_temp_img.data + eta_temp, requires_grad=True)
            # eta_temp = torch.clamp(perturb_temp_img.data - temp_images.data, -self.epsilon, self.epsilon)
            # perturb_temp_img = Variable(temp_images.data + eta_temp, requires_grad=True)
            # perturb_temp_img = Variable(torch.clamp(perturb_temp_img, 0, 1), requires_grad=True)

            # mask = torch.zeros((search_images.shape[0], 3, 239, 239)).to(self.device)
            # mask[:,:,56:183,56:183] = eta_temp
            # perturb_search_img = Variable(search_images.data + mask, requires_grad=True)
            # perturb_search_img = Variable(torch.clamp(perturb_search_img, 0, 1), requires_grad=True)

        return perturb_temp_img, optimized_noise, perturb_search_img

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
