#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, pdb
import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn

#import utils.data_loaders
import data_transforms
import network_utils
import torchvision
import random

from multiscaleloss import *
from time import time

import test


def train(cfg, init_epoch, dataset_loader, dataset_test_loader, train_transforms, val_transforms,
                                  sarnet, sarnet_solver, sarnet_lr_scheduler,
                                  ckpt_dir, train_writer, val_writer,
                                  Best_Img_PSNR, Best_Epoch):


    n_itr = 0
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Set up data loader
        device = torch.device("cuda")
        cfg.CONST.TRAIN_BATCH_SIZE=10
        '''
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)
        '''
        train_data_loader = torch.utils.data.DataLoader(dataset_loader, batch_size=cfg.CONST.TRAIN_BATCH_SIZE, shuffle=True)

        # Tick / tock
        epoch_start_time = time()
        # Batch average meterics
        batch_time = network_utils.AverageMeter()
        data_time = network_utils.AverageMeter()
        sarnet_mse_losses = network_utils.AverageMeter()
        sarnet_losses = network_utils.AverageMeter()
        img_PSNRs = network_utils.AverageMeter()

        # Adjust learning rate
        sarnet_lr_scheduler.step()
        print('[INFO] learning rate: {0}\n'.format(sarnet_lr_scheduler.get_lr()))

        batch_end_time = time()
        seq_num = len(train_data_loader)

        #vggnet = VGG19()
        #if torch.cuda.is_available():
        #    vggnet = torch.nn.DataParallel(vggnet).cuda()
        #for batch_idx, [img_freq, img_blur, img_gt] in enumerate(train_data_loader):
        for batch_idx, [img_freq, img_blur, img_p, img_gt] in enumerate(train_data_loader):
            # switch models to training mode
            seq_idx=0

            sarnet.train()
            img_freq = img_freq.permute(0,3,1,2).to(torch.float32).to(device)
            img_blur = img_blur.permute(0,3,1,2).to(torch.float32).to(device)
            img_p = img_p.permute(0,3,1,2).to(torch.float32).to(device)
            img_gt = img_gt.permute(0,3,1,2).to(torch.float32).to(device)
            img_clear=img_gt
            #pdb.set_trace()
            #img_blur_hold = img_blur
            #output_img = sarnet(img_blur, img_freq[:,:cfg.NETWORK.NUM,:,:], img_p)
            output_img = sarnet(img_blur, img_freq[:,:cfg.NETWORK.NUM,:,:], img_p)
            #output_img = sarnet(img_blur)
            #pdb.set_trace()
            '''
            img_dir = os.path.join(cfg.DIR.OUT_PATH, 'test')
            cv2.imwrite(img_dir + '/' + str(0) + '.png',(img_gt.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8))
            pdb.set_trace()
            '''
            # sarnet loss
            sarnet_mse_loss = mseLoss(output_img, img_gt)
            sarnet_mse_losses.update(sarnet_mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            sarnet_loss = sarnet_mse_loss

            sarnet_losses.update(sarnet_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            img_PSNR = PSNR(output_img, img_clear)
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            # sarnet update
            sarnet_solver.zero_grad()
            sarnet_loss.backward()
            sarnet_solver.step()
            # Append loss to TensorBoard
            train_writer.add_scalar('SARNet/sarnetLoss_0_TRAIN', sarnet_loss.item(), n_itr)
            train_writer.add_scalar('SARNet/sarnetMSELoss_0_TRAIN', sarnet_mse_loss.item(), n_itr)

            n_itr = n_itr + 1

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # print per batch
            if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                print(
                    '[TRAIN] [Ech {0}/{1}][Seq {2}/{3}] BT {4} sarnetLoss {5} PSNR {6} Best_PSNR {7}'
                    .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, seq_num, batch_time, sarnet_losses, img_PSNRs, Best_Img_PSNR))

            '''
            if seq_idx == 0 and batch_idx < cfg.TEST.VISUALIZATION_NUM:
                img_blur = img_blur[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_clear = img_clear[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                output_last_img = output_last_img[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_out = output_img[0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)

                result = torch.cat([torch.cat([img_blur, img_clear], 2),
                                    torch.cat([output_last_img, img_out], 2)], 1)
                result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                train_writer.add_image('STFANet/TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)
            
            # print per sequence
            print('[TRAIN] [Epoch {0}/{1}] [Seq {2}/{3}] ImgPSNR_avg {4}\n'
                  .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, seq_idx + 1, seq_num, img_PSNRs.avg))
            '''
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('STFANet/EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t ImgPSNR_avg {3}\n'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, img_PSNRs.avg))

        # Validate the training models
        img_PSNR = test.test(cfg, epoch_idx, dataset_test_loader, val_transforms, sarnet, val_writer)
        #img_PSNR = PSNR(output_img, img_clear)
        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                                                      epoch_idx + 1, sarnet, sarnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)
        if img_PSNR >= Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNR
            Best_Epoch = epoch_idx + 1
            network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                      epoch_idx + 1, sarnet, sarnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
