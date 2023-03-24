#!/usr/bin/python
# -*- coding: utf-8 -*-
import pdb, math
import functools
from torch import nn
import torch.nn.functional as F
from model.unet_parts import *
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU



class Sarnet(nn.Module):
    def __init__(self, n_channels, n_classes, channel_num, self_reduction, channel_reduction, k, channel_att, sub_dim, inF = True, dataA='in', final='sub', bilinear=True):
        super(Sarnet, self).__init__()
        channel_size = channel_num
        self.channel_size = channel_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inF=inF
        self.dataA=dataA
        #self.final=final
        self.Decode_att=0
        self.inF = False 
        if self.inF:
            self.inc = DoubleConv(4, channel_size)
        else:
            self.inc = DoubleConv(1, channel_size)
        #self.inc = DoubleConv(4, channel_size)
        #self.conv_att = DoubleConv(4, 4)
        self.down1 = Down(channel_size, channel_size*2)
        self.down2 = Down(channel_size*2, channel_size*4)
        self.down3 = Down(channel_size*4, channel_size*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(channel_size*8, channel_size*8)
        self.up1 = UpF(channel_size*16, channel_size*4, bilinear)
        self.up2 = UpF(channel_size*8, channel_size*2, bilinear)
        self.up3 = UpF(channel_size*4, channel_size, bilinear)
        self.up4 = UpF(channel_size*2, channel_size*1, bilinear)
        # if self.Decode_att==1:
        #     self.up1_att = Up(channel_size*16, channel_size*8, bilinear)
        #     self.up2_att = Up(channel_size*8, channel_size*4, bilinear)
        #     self.up3_att = Up(channel_size*4, channel_size*2, bilinear)
        #     self.up4_att = Up(channel_size*2, channel_size*1, bilinear)
        #     self.up4_F = Up2(channel_size*2, channel_size*1, bilinear)
        #self.outc = OutConv(channel_size, n_classes)
        self.final='sub'
        
        if self.final=='sub':
            #pdb.set_trace()
            self.outc = DoubleConv2(1*channel_size, n_classes)
        else:
            self.outc = DoubleConv2(1*channel_size, n_classes)


        # Input the parts of Freq. components
        self.img_down1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.img_down2 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.img_down3 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.img_down4 = nn.Upsample(scale_factor=0.0625, mode='bilinear', align_corners=True)
        #self.img_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.channel1 = ChannelGate(channel_size*2)
        #self.SpatialGate = SpatialGate()
        self.num=6
        self.attn1 = Self_subF_Att3( self.num, channel_size, 'relu', self_reduction = self_reduction, channel_reduction = channel_reduction, k=k, channel_att =channel_att, sub=sub_dim)
        self.attn2 = Self_subF_Att3( self.num, channel_size, 'relu', self_reduction = self_reduction, channel_reduction = channel_reduction, k=k, channel_att =channel_att, sub=sub_dim)
        self.attn3 = Self_subF_Att3( self.num, channel_size, 'relu', self_reduction = self_reduction, channel_reduction = channel_reduction, k=k, channel_att =channel_att, sub=sub_dim)
        self.attn4 = Self_subF_Att3( self.num, channel_size, 'relu', self_reduction = self_reduction, channel_reduction = channel_reduction, k=k, channel_att =channel_att, sub=sub_dim)


        self.first_resolution = DoubleConv2(6, 2*channel_size)   #DoubleConv(6, 2*channel_size) 
        self.second_resolution = DoubleConv2(6, 4*channel_size)
        self.third_resolution = DoubleConv2(6, 8*channel_size)
        self.four_resolution = DoubleConv2(6, 8*channel_size)


        self.calayer1=CALayer(2*2*channel_size)
        self.calayer2=CALayer(2*4*channel_size)
        self.calayer3=CALayer(2*8*channel_size)
        self.calayer4=CALayer(2*8*channel_size)

        # self.cnn1 = DoubleConv2(6,1*2*channel_size)
        # self.cnn2 = DoubleConv2(6,1*4*channel_size)
        # self.cnn3 = DoubleConv2(6,1*8*channel_size)
        # self.cnn4 = DoubleConv2(6,1*8*channel_size)

        # self.pa1=PALayer(6*1)
        # self.pa2=PALayer(6*1)
        # self.pa3=PALayer(6*1)
        # self.pa4=PALayer(6*1)

        # self.calayer1F=CALayer(4*6)
        # self.calayer2F=CALayer(4*6)
        # self.calayer3F=CALayer(4*6)
        # self.calayer4F=CALayer(4*6)


        self.calayerF=CALayer(2*channel_size)

    def forward(self, img_blur, freq_blur, img_p):

        if self.inF:
            x = torch.cat([img_blur[:,0:1], img_p[:,0:3]], 1)
        else:
            #x = torch.cat([img_blur[:,0:1], img_blur[:,0:1]], 1)
            x = img_blur[:,0:1]
            # x = img_blur

        self.dataA='de'
        if self.dataA=='de':
            freq_down1 = self.img_down1(torch.cat([img_p[:,9:12], freq_blur[:,9:12]], 1))
            freq_down2 = self.img_down2(torch.cat([img_p[:,6:9], freq_blur[:,6:9]], 1))
            freq_down3 = self.img_down3(torch.cat([img_p[:,3:6], freq_blur[:,3:6]], 1))
            freq_down4 = self.img_down4(torch.cat([img_p[:,0:3], freq_blur[:,0:3]], 1))
        else:
            freq_down1 = self.img_down1(torch.cat([img_p[:,9:12], freq_blur[:,0:3]], 1))
            freq_down2 = self.img_down2(torch.cat([img_p[:,6:9], freq_blur[:,3:6]], 1))
            freq_down3 = self.img_down3(torch.cat([img_p[:,3:6], freq_blur[:,6:9]], 1))
            freq_down4 = self.img_down4(torch.cat([img_p[:,0:3], freq_blur[:,9:12]], 1))
            
        x1 = self.inc(x)
        # x1 = x
        x_int=x1

        x_freq_collect = []
        safm1 = self.attn1(freq_down1[:,0:3],freq_down1[:,3:6])
        safm2 = self.attn2(freq_down2[:,0:3],freq_down2[:,3:6])
        safm3 = self.attn3(freq_down3[:,0:3],freq_down3[:,3:6])
        safm4 = self.attn4(freq_down4[:,0:3],freq_down4[:,3:6])

        x_freq_down1 = self.first_resolution(safm1)
        x_freq_down2 = self.second_resolution(safm2)
        x_freq_down3 = self.third_resolution(safm3)
        x_freq_down4 = self.four_resolution(safm4)  
        # pdb.set_trace()
        w1 = self.calayer1(torch.cat([self.down1(x1), x_freq_down1],1))
        w1=w1.view(-1,2,2*self.channel_size)[:,:,:,None,None]
        x2 = self.down1(x1) * w1[:,0] + x_freq_down1* w1[:,1]

        w2 = self.calayer2(torch.cat([self.down2(x2), x_freq_down2],1))
        w2=w2.view(-1,2,4*self.channel_size)[:,:,:,None,None]
        x3 = self.down2(x2) * w2[:,0] + x_freq_down2* w2[:,1]

        w3 = self.calayer3(torch.cat([self.down3(x3), x_freq_down3],1))
        w3=w3.view(-1,2,8*self.channel_size)[:,:,:,None,None]
        x4 = self.down3(x3) * w3[:,0] + x_freq_down3* w3[:,1]

        w4 = self.calayer4(torch.cat([self.down4(x4), x_freq_down4],1))
        w4=w4.view(-1,2,8*self.channel_size)[:,:,:,None,None]
        x5 = self.down4(x4) * w4[:,0] + x_freq_down4* w4[:,1]

        
        if self.Decode_att==1:
            x4 = self.up1_att(x5, x4)
            x = self.up1(x5, x4)
            x3 = self.up2_att(x, x3)
            x = self.up2(x, x3)
            x2 = self.up3_att(x, x2)
            x = self.up3(x, x2)
            #x_take = self.img_up(x)
            #pdb.set_trace()
            x1 = self.up4_att(x, x1)
            x = self.up4_F(x, x1)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1) 

        #xx=x
        if self.final=='sub':
            #pdb.set_trace()
            x = torch.cat([x, x_int],1)
            #pre = self.preF(x_int)

            wF = self.calayerF(x)
            #xF = wF*x
            ##pdb.set_trace()
            #wF=wF.view(-1,3,self.channel_size)[:,:,:,None,None]
            #xF = x[:,0:self.channel_size] * wF[:,0] + x[:,self.channel_size:2*self.channel_size] * wF[:,1] + x[:,2*self.channel_size:3*self.channel_size] * wF[:,1]
            wF=wF.view(-1,2,self.channel_size)[:,:,:,None,None]
            xF = x[:,0:self.channel_size] * wF[:,0] + x[:,self.channel_size:2*self.channel_size] #* wF[:,1] + x[:,2*self.channel_size:3*self.channel_size] * wF[:,2]          
            #wF=wF.view(-1,2,self.channel_size)[:,:,:,None,None]
            #pdb.set_trace()
            #xF = x[:,0:self.channel_size] * wF[:,0] + x[:,self.channel_size:2*self.channel_size] * wF[:,1] 
            # xF = self.sub(x_int, xF)
            #pdb.set_trace()
            #xF = self.kconv_deblur(xF, pre)
            #logits = self.palayer(xF)
            ##pre = self.preF(xF)
            #pdb.set_trace()
            ##xF = self.kconv_deblur(xx, pre)
            #pdb.set_trace()
            logits=xF
            logits = self.outc(logits)        
        else:
            xF=x
            logits = self.outc(x)      
        # return logits, xF, x_freq_collect
        return logits


class Self_subF_Att3(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim1, in_dim2, activation, self_reduction=4, channel_reduction=2, k=3, channel_att ='cbam', sub=8):
        super(Self_subF_Att3,self).__init__()
        self.chanel_in = in_dim1
        self.activation = activation
        self.num_subspace=sub
        self.subnet_q = Subspace(in_dim1//2, self.num_subspace)
        self.subnet_k = Subspace(in_dim1//2, self.num_subspace)
        self.fusion = Subspace(self.num_subspace*2, self.num_subspace)


        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        x = torch.cat((x1,x2),1)
        m_batchsize,C,width ,height = x.size()

        subnet_q = self.subnet_q(x1)
        subnet_k = self.subnet_k(x2)
        V_t = torch.cat((subnet_q, subnet_k), 1)
        V_t = self.fusion(V_t)

        proj_query  = V_t.view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  V_t.view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 

        V_t = V_t.reshape(m_batchsize, self.num_subspace, width*height)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        x1 = x1.reshape(m_batchsize, C//2, height*width)
        x2 = x2.reshape(m_batchsize, C//2, height*width)
        project_feature1 = torch.matmul(project_mat, x1.permute(0, 2, 1))
        project_feature2 = torch.matmul(project_mat, x2.permute(0, 2, 1))

        x1_out = torch.matmul(V, project_feature1).permute(0, 2, 1).reshape(m_batchsize, C//2, width, height)
        x2_out = torch.matmul(V, project_feature2).permute(0, 2, 1).reshape(m_batchsize, C//2,  width, height)


        proj_value = torch.cat((x1_out, x2_out), 1).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,-1,width,height)
        out = self.gamma*out + x

        return out


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return y

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = UNetConvBlock(in_size, out_size, False, 0.2)
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.blocks(x)

        return x + sc

class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out