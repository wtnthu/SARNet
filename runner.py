#!/usr/bin/python
# -*- coding: utf-8 -*-
# 

import matplotlib
import os
matplotlib.use('Agg')

# Fix problem: possible deadlock in dataloader
# import cv2
# cv2.setNumThreads(0)

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from build import bulid_net
import torch

def get_args_from_command_line():

    parser = ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [cuda]', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--phase', dest='phase', help='phase of CNN', default=cfg.NETWORK.PHASE, type=str)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=cfg.CONST.WEIGHTS, type=str)
    # parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default=cfg.DIR.DATASET_ROOT, type=str)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--num', type = int,default=12,  help='Set output path')
    parser.add_argument('--arch', dest='arch', help='Set dataset root_path', default=cfg.NETWORK.RESTOREDARCH, type=str)
    parser.add_argument('--in', dest='input', help='Set dataset root_path',  default='eevee', type=str)
    parser.add_argument('--lr', type = float,default=5e-4,  help='Set output path')
    parser.add_argument('--channel_num', type = int , default = 16 ,  help='Set channel reduction')
    parser.add_argument('--channel_reduction', type = int , default = 2,  help='Set channel reduction')
    parser.add_argument('--self_reduction', type = int , default = 4,  help='Set self attention reduction')
    parser.add_argument('--channel_att', type = str , default = 'cbam',  help='Set self attention method')
    parser.add_argument('--k', type = int , default = 3,  help='Set self attention method')
    parser.add_argument('--sub', type = int , default = 8,  help='Set self attention method')
    parser.add_argument('--seed', type = int , default = 0,  help='Set self attention method')
    parser.add_argument('--mode', type = str , default = 'amp',  help='Set self attention method')



    args = parser.parse_args()
    return args

def main():

    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.phase is not None:
        cfg.NETWORK.PHASE = args.phase
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    # if args.data_path is not None:
    #     cfg.DIR.DATASET_ROOT = args.data_path
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.num is not None: 
        cfg.NETWORK.NUM= args.num
    if args.arch is not None: 
        cfg.NETWORK.RESTOREDARCH = args.arch
    if args.lr is not None:
        cfg.TRAIN.LEARNING_RATE = args.lr
    cfg.input = args.input
    cfg.channel_num = args.channel_num
    cfg.channel_reduction = args.channel_reduction
    cfg.self_reduction = args.self_reduction
    cfg.channel_att = args.channel_att
    cfg.k = args.k
    cfg.sub_dim = args.sub
    cfg.seed = args.seed
    cfg.mode = args.mode


    print('Use config:')
    pprint(cfg)


    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    print('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))

    # Setup Network & Start train/test process
    bulid_net(cfg)

if __name__ == '__main__':

    main()
