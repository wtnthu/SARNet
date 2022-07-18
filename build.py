#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys, pdb
import torch.backends.cudnn
import torch.utils.data
import data_transforms2
import network_utils
import model
import model.Sarnet
from datetime import datetime as dt
from tensorboardX import SummaryWriter
import train, test, load_npy2
from multiscaleloss import *
import random


def set_seed(seed):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def bulid_net(cfg):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True
    set_seed(cfg.seed)
    
    # Set up data augmentation
    train_transforms = data_transforms2.Compose([
        data_transforms2.ToTensor(),
        data_transforms2.RandomCrop(88, 128),
    ])

    test_transforms = data_transforms2.Compose([
        data_transforms2.ToTensor(),
    ])

    # Set up data loader
    #dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()
    train_data = load_npy2.MyDataset(txt='train_'+cfg.input+'.txt', transform=train_transforms)
    test_data = load_npy2.MyDataset(txt='test_'+cfg.input+'.txt', transform=test_transforms)
    #train_data = load_npy2.MyDataset(txt=cfg.input+'.txt', transform=train_transforms)
    #test_data = load_npy2.MyDataset(txt=cfg.input+'.txt', transform=test_transforms)
    dataset_loader = train_data
    dataset_test_loader = test_data
    # Set up networks    
    if cfg.NETWORK.RESTOREDARCH == 'Sarnet':
            sarnetModel = model.Sarnet.__dict__[cfg.NETWORK.RESTOREDARCH](cfg.NETWORK.NUM, 1, cfg.channel_num, cfg.self_reduction, cfg.channel_reduction, cfg.k, cfg.channel_att, cfg.sub_dim)

    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.RESTOREDARCH,
                                                network_utils.count_parameters(sarnetModel)))
    
    # pdb.set_trace()

    # Initialize weights of networks
    sarnetModel.apply(network_utils.init_weights_xavier)

    # Set up solver
    a =  filter(lambda p: p.requires_grad, sarnetModel.parameters())
    sarnetModel_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, sarnetModel.parameters()), lr=cfg.TRAIN.LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    if torch.cuda.is_available():
        sarnetModel = torch.nn.DataParallel(sarnetModel).cuda()

    # Load pretrained model if exists
    init_epoch       = 0
    Best_Epoch       = -1
    Best_Img_PSNR    = 0


    if cfg.NETWORK.PHASE in ['test','resume', 'best', '2000', '3000']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        sarnetModel.load_state_dict(checkpoint['sarnetModel_state_dict'])
        init_epoch = checkpoint['epoch_idx']+1
        Best_Img_PSNR = checkpoint['Best_Img_PSNR']
        Best_Epoch = checkpoint['Best_Epoch']
        print('[INFO] {0} Recover complete. Current epoch #{1}, Best_Img_PSNR = {2} at epoch #{3}.' \
              .format(dt.now(), init_epoch, Best_Img_PSNR, Best_Epoch))



    # Set up learning rate scheduler to decay learning rates dynamically
    sarnetModel_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(sarnetModel_solver,
                                                                   milestones=cfg.TRAIN.LR_MILESTONES,
                                                                   gamma=cfg.TRAIN.LR_DECAY)

    # Summary writer for TensorBoard
    #output_dir = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat().replace(":","_") + '_' + cfg.NETWORK.RESTOREDARCH, '%s')
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s')
    
    log_dir      = output_dir % 'logs'
    ckpt_dir     = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join( log_dir, 'train'))
    test_writer  = SummaryWriter(os.path.join( log_dir, 'test'))
    print('[INFO] Output_dir： {0}'.format(output_dir[:-2]))

    if cfg.NETWORK.PHASE in ['train','resume']:
        #pdb.set_trace()
        train.train(cfg, init_epoch, dataset_loader, dataset_test_loader, train_transforms, test_transforms,
                              sarnetModel, sarnetModel_solver, sarnetModel_lr_scheduler,
                              ckpt_dir, train_writer, test_writer,
                              Best_Img_PSNR, Best_Epoch)
    if cfg.NETWORK.PHASE in ['test','best', '2000', '3000']:
        
        if os.path.exists(cfg.CONST.WEIGHTS):
            test.test(cfg, init_epoch, dataset_test_loader, test_transforms, sarnetModel, test_writer)
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
            sys.exit(2)
