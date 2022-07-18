#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.backends.cudnn
import torch.utils.data
#import utils.data_loaders
import data_transforms
import network_utils, pdb
from multiscaleloss import *
import torchvision, pdb

import numpy as np
import scipy.io as io
import lpips, time
import cv2

from time import time

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test(cfg, epoch_idx, dataset_loader, test_transforms, sarnet, test_writer):
    # Set up data loader
    cfg.CONST.TRAIN_BATCH_SIZE=1
    test_data_loader = torch.utils.data.DataLoader(dataset_loader, batch_size= cfg.CONST.TRAIN_BATCH_SIZE, shuffle=False)

    # loss_fn_alex = lpips.LPIPS(net='alex').to('cuda') # best forward scores

    seq_num = len(test_data_loader)
    # Batch average meterics
    batch_time = network_utils.AverageMeter()
    test_time = network_utils.AverageMeter()
    data_time = network_utils.AverageMeter()
    img_PSNRs = network_utils.AverageMeter()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)


    batch_end_time = time()
    test_psnr = dict()
    g_names= 'init'
    save_num=0
    #for batch_idx, [img_freq, img_blur, img_gt] in enumerate(test_data_loader):
    for batch_idx, [img_freq, img_blur, img_p, img_gt] in enumerate(test_data_loader):
        data_time.update(time() - batch_end_time)
        seq_len = len(test_data_loader)
        #pdb.set_trace()
        # Switch models to training mode
        sarnet.eval()

        test_psnr[batch_idx] = {
            'n_samples': 0,
            'psnr': [],
        }
        with torch.no_grad():
            # Test runtime
            torch.cuda.synchronize()
            device = torch.device("cuda")
            test_time_start = time()
            # --Forwards--
            
            img_freq = img_freq.permute(0,3,1,2).to(torch.float32).to(device)
            img_blur = img_blur.permute(0,3,1,2).to(torch.float32).to(device)
            img_p = img_p.permute(0,3,1,2).to(torch.float32).to(device)
            img_gt = img_gt.permute(0,3,1,2).to(torch.float32).to(device)
            #mul=5
            #img_freq = torch.cat(mul*[img_freq], dim = 0)
            #img_blur = torch.cat(mul*[img_blur], dim = 0)
            #img_p = torch.cat(mul*[img_p], dim = 0)
            

            #'''
            #pdb.set_trace()
            #img_gt=img_gt[:,:,:,1:-1]
            if img_gt.shape[2]%8!=0:
                cut = img_gt.shape[2]%16-1
                img_freq = img_freq[:,:,cut:-1,:]
                img_p = img_p[:,:,cut:-1,:]
                img_blur = img_blur[:,:,cut:-1,:]
                img_gt = img_gt[:,:,cut:-1,:]
            if img_gt.shape[3]%8!=0:
                cut = img_gt.shape[3]%16-1
                img_freq = img_freq[:,:,:,cut:-1]
                img_p = img_p[:,:,:,cut:-1]
                img_blur = img_blur[:,:,:,cut:-1]
                img_gt = img_gt[:,:,:,cut:-1]
            #'''
            img_dir = os.path.join(cfg.DIR.OUT_PATH, 'test')
            

            img_clear = img_gt

            output_img = sarnet(img_blur, img_freq[:, :cfg.NETWORK.NUM,:,:], img_p)
            output_img = output_img[0:1,:,:,:]
            img_freq=img_freq[0:1,:,:,:]
            img_p=img_p[0:1,:,:,:]


            #output_img = sarnet(img_blur)
            torch.cuda.synchronize()
            test_time.update(time() - test_time_start)
            ##print('[RUNING TIME] {0}'.format(test_time))
            img_PSNR = PSNR(output_img, img_clear)
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)


            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # Print per batch
            if cfg.NETWORK.PHASE in ['test','best', '2000', '3000']:
                name=cfg.NETWORK.PHASE
                test_psnr[batch_idx]['n_samples'] += 1
                test_psnr[batch_idx]['psnr'].append(img_PSNR)
                img_dir = os.path.join(cfg.DIR.OUT_PATH, name)
                if not os.path.isdir(img_dir):
                    mkdir(img_dir)

                print('[TEST Saving: ]'+img_dir + '/' + str(save_num).zfill(5) + '.png')
                #pdb.set_trace()
                #'''
                cv2.imwrite(img_dir + '/' + str(save_num).zfill(5) +'.png',
                            (output_img.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                np.uint8),[int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                #'''
                #'''
                cv2.imwrite(img_dir + '/' + str(save_num).zfill(5) + '_max' + '.png',
                            (img_blur[:, 0:1, :, :].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                np.uint8),[int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                #'''
                #'''
                cv2.imwrite(img_dir + '/' + str(save_num).zfill(5) + '_gt'+ '.png',
                            (img_clear.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                np.uint8),[int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                #'''
                save_num = save_num + 1




    # Output testing results
    if cfg.NETWORK.PHASE in ['test','best', '2000', '3000']:
        name2=cfg.NETWORK.PHASE
        #pdb.set_trace()
        # Output test results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))

        for name in test_psnr:
            #test_psnr[name]['psnr'] = test_psnr[name]['psnr']
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))

        result_file = open(os.path.join(cfg.DIR.OUT_PATH, name2, 'test_result.txt'), 'w')
        sys.stdout = result_file
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        for name in test_psnr:
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))
        result_file.close()
    else:
        # Output val results
        #pdb.set_trace()
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))

        print('[TEST] [Epoch{0}]\t BatchTime_avg {1}\t DataTime_avg {2}\t ImgPSNR_avg {3}\n'
              .format(cfg.TRAIN.NUM_EPOCHES, batch_time.avg, data_time.avg, img_PSNRs.avg))

        # Add testing results to TensorBoard
        test_writer.add_scalar('STFANet/EpochPSNR_1_TEST', img_PSNRs.avg, epoch_idx + 1)

        return img_PSNRs.avg
