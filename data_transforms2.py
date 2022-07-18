#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import pdb
import numpy as np
import torch
import torchvision.transforms.functional as F
from config import cfg
from PIL import Image
import random
import numbers
class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, f, tt , pp, gt, id):
        for t in self.transforms:
            f, tt, pp, gt = t(f, tt, pp, gt, id)
        return f, tt, pp, gt


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [Image.fromarray(np.uint8(img)) for img in seq_blur]
        seq_clear = [Image.fromarray(np.uint8(img)) for img in seq_clear]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            seq_blur  = [F.adjust_brightness(img, brightness_factor) for img in seq_blur]
            seq_clear = [F.adjust_brightness(img, brightness_factor) for img in seq_clear]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            seq_blur  = [F.adjust_contrast(img, contrast_factor) for img in seq_blur]
            seq_clear = [F.adjust_contrast(img, contrast_factor) for img in seq_clear]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            seq_blur  = [F.adjust_saturation(img, saturation_factor) for img in seq_blur]
            seq_clear = [F.adjust_saturation(img, saturation_factor) for img in seq_clear]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            seq_blur  = [F.adjust_hue(img, hue_factor) for img in seq_blur]
            seq_clear = [F.adjust_hue(img, hue_factor) for img in seq_clear]

        seq_blur  = [np.asarray(img) for img in seq_blur]
        seq_clear = [np.asarray(img) for img in seq_clear]

        seq_blur  = [img.clip(0,255) for img in seq_blur]
        seq_clear = [img.clip(0,255) for img in seq_clear]

        return seq_blur, seq_clear

class RandomColorChannel(object):
    def __call__(self, seq_blur, seq_clear):
        random_order = np.random.permutation(3)

        seq_blur  = [img[:,:,random_order] for img in seq_blur]
        seq_clear = [img[:,:,random_order] for img in seq_clear]

        return seq_blur, seq_clear

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, seq_blur, seq_clear):

        shape = seq_blur[0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        # only apply to blurry images
        seq_blur = [img + gaussian_noise for img in seq_blur]
        seq_blur = [img.clip(0, 1) for img in seq_blur]

        return seq_blur, seq_clear

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [img/self.std -self.mean for img in seq_blur]
        seq_clear = [img/self.std -self.mean for img in seq_clear]

        return seq_blur, seq_clear

class To_nor(object):
        
    def __call__(self, img_f, img_t, img_p, img_gt):
        
        for i in range(img_f.shape[2]):
            img_f[:,:,i:i+1] = img_f[:,:,i:i+1]/np.max(img_f[:,:,i])
        for i in range(img_t.shape[2]):
            img_t[:,:,i:i+1] = img_t[:,:,i:i+1]/np.max(img_t[:,:,i])
        for i in range(img_p.shape[2]):
            img_p[:,:,i:i+1] = img_p[:,:,i:i+1]/np.max(img_p[:,:,i])
        for i in range(img_gt.shape[2]):
            img_gt[:,:,i:i+1] = img_gt[:,:,i:i+1]/np.max(img_gt[:,:,i])
        pdb.set_trace()
        return img_f, img_t, img_p, img_gt

class CenterCrop(object):

    def __init__(self, crop_size1, crop_size2):
        """Set the height and weight before and after cropping"""

        self.crop_size_h  = crop_size1
        self.crop_size_w  = crop_size2

    def __call__(self, img_f, img_t, img_p, img_gt, id):
        #img_gt = np.array(img_gt)
        input_size_h, input_size_w, _ = img_gt.shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        img_f  = img_f[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        img_t = img_t[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        img_p = img_p[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        img_gt = img_gt[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        return img_f, img_t, img_p, img_gt
        #return img_f, imimg_tg_f, img_gt

class RandomCrop(object):

    def __init__(self, crop_size1, crop_size2):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size1
        self.crop_size_w  = crop_size2

    def __call__(self, img_f, img_t, img_p, img_gt, id):
        '''
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)

        seq_blur  = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]
        '''
        input_size_h, input_size_w, _ = img_gt.shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)

        img_f  = img_f[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        img_t = img_t[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        img_p = img_p[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        img_gt = img_gt[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]

        return img_f, img_t, img_p, img_gt

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, img_f, img_t, img_p, img_gt):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            '''
            seq_blur  = [np.copy(np.fliplr(img)) for img in seq_blur]
            seq_clear = [np.copy(np.fliplr(img)) for img in seq_clear]
            '''
            img_f  = np.copy(np.fliplr(img_f))
            img_t  = np.copy(np.fliplr(img_t))
            img_p  = np.copy(np.fliplr(img_p))
            img_gt  = np.copy(np.fliplr(img_gt))
            

        return img_f, img_t, img_p, img_gt


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, img_f, img_t, img_p, img_gt):
        if random.random() < 0.5:
            '''
            seq_blur  = [np.copy(np.flipud(img)) for img in seq_blur]
            seq_clear = [np.copy(np.flipud(img)) for img in seq_clear]
            '''
            img_f  = np.copy(np.flipud(img_f))
            img_t  = np.copy(np.flipud(img_t))
            img_p  = np.copy(np.flipud(img_p))
            img_gt  = np.copy(np.flipud(img_gt))

        return img_f, img_t, img_p, img_gt



class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, img_f, img_t, img_p, img_gt, id):
        '''
        seq_blur  = [np.transpose(img, (2, 0, 1)) for img in seq_blur]
        seq_clear = [np.transpose(img, (2, 0, 1)) for img in seq_clear]
        # handle numpy array
        seq_blur_tensor  = [torch.from_numpy(img).float() for img in seq_blur]
        seq_clear_tensor = [torch.from_numpy(img).float() for img in seq_clear]
        '''
        img_f, img_t, img_p, img_gt = img_f.astype(np.float32), img_t.astype(np.float32), img_p.astype(np.float32), img_gt.astype(np.float32)
        
        #if img_f.shape[1]>288:
        img_t= np.flipud(img_t)
        img_t= np.fliplr(img_t)

        img_gt= np.flipud(img_gt)
        img_gt= np.fliplr(img_gt)
        


        #img_t = np.transpose(img_t, (2, 0, 1))
        #img_f = np.transpose(img_f, (2, 0, 1))
        #img_gt = np.transpose(img_gt, (2, 0, 1))

        #img_t  = torch.from_numpy(img_t).float()
        #img_f  = torch.from_numpy(img_f).float()
        #img_gt  = torch.from_numpy(img_gt).float()
        img_gt = np.expand_dims(img_gt, 2)
        img_gt = (img_gt-np.min(img_gt))/(np.max(img_gt)-np.min(img_gt))
        
        #img_gt = img_gt/255.0
        img_t1 = np.expand_dims(img_t[:,:,0],2)
        img_t2 = np.expand_dims(img_t[:,:,1],2)
        #img_t = np.expand_dims(img_t[:,:,0],2)

        img_t1 = 1.0-(img_t1-np.min(img_t1))/(np.max(img_t1)-np.min(img_t1))
        img_t2 = (img_t2-np.min(img_t2))/(np.max(img_t2)-np.min(img_t2))
        for i in range(img_f.shape[2]):
            img_f[:,:,i]= np.flipud(img_f[:,:,i])
            img_f[:,:,i]= np.fliplr(img_f[:,:,i])
            MM = np.max(img_f[:,:,i])
            mm = np.min(img_f[:,:,i])
            img_f[:,:,i:i+1] = (img_f[:,:,i:i+1]-mm)/(MM-mm)
        if np.sum(np.isnan(img_f))>0:
            print('NAN.............................ERROR')
            pdb.set_trace()
        #img_gt  = torch.from_numpy(img_gt).float()
        img_f =1.0-img_f 
        if img_f.shape[1]>288:
            img_f = img_f[:, 3:291,:]
            img_t1 = img_t1[:, 3:291,:]
            img_t2 = img_t2[:, 3:291,:]
        #pdb.set_trace()
        for i in range(img_p.shape[2]):
            img_p[:,:,i]= np.flipud(img_p[:,:,i])
            img_p[:,:,i]= np.fliplr(img_p[:,:,i])
            MM = np.max(img_p[:,:,i])
            mm = np.min(img_p[:,:,i])
            img_p[:,:,i:i+1] = (img_p[:,:,i:i+1]-mm)/(MM-mm)
        img_p =1.0-img_p 
        if img_p.shape[1]>288:
            img_p = img_p[:, 3:291,:]
        
        #img_p = img_p * img_t1 + img_f[:,:,9:10]


        img_t = np.concatenate((img_t1, img_t2), axis=2)
        return img_f, img_t, img_p, img_gt

