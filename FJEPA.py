import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_features, out_features, kernel_size=3, stride=1, padding=1, dilation=1):   
    """Convolutional layer with BatchNorm and LeakyReLU activation"""
    
    return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1))

def flow_cnn(in_features):
    """Convolutional layer for flow estimation"""
    return nn.Sequential(
            nn.Conv2d(in_features, 2, kernel_size=3, stride=1, 
                        padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1))


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    """Deconvolutional layer with BatchNorm and LeakyReLU activation"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding)

def corr4D(X_tnext, X_hat_tnext):
    """
    Compute 4D correlation between embedding of next frame and predicted next frame embedding
    X_tnext: [B, C, H, W] (next frame embedding)
    X_hat_tnext: [B, C, H, W] (predicted next frame embedding)
    """

    batch_size, out_channels, ht, wd = X_tnext.shape
    X_tnext = X_tnext.view(batch_size, out_channels, ht*wd)
    X_hat_tnext = X_hat_tnext.view(batch_size, out_channels, ht*wd) 

    corr = torch.matmul(X_tnext.transpose(1,2), X_hat_tnext)
    corr = corr.view(batch_size, 1, ht*wd, ht*wd)
    corr = corr/torch.sqrt(torch.tensor(out_channels).float())
    correlation = corr

    mat_mul = torch.matmul(correlation, X_tnext.view(batch_size, out_channels, ht*wd, 1))
    mat_mul = mat_mul.view(batch_size, out_channels, ht, wd)
    
    return mat_mul
    

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float().to(device)

    if torch.is_tensor(flo): 
        vgrid = torch.autograd.Variable(grid) + flo
    else:
        vgrid = torch.autograd.Variable(grid)

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask

class FJepa(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.conv1 = conv(in_features, 4, 3, 2)
        self.conv1a = conv(4, 4, 3)
        self.conv1b = conv(4, 4, 3)
        self.conv1c = conv(4, 4, 3)

        self.conv1_0 = conv(4, 16, 3)
        self.conv1_1 = conv(32, 16, 3)
        self.conv1_2 = conv(48, 4, 3)
        self.conv1_3 = conv(52, 4, 3)
        
        self.flow_predictor1 = flow_cnn(4)
        self.deconv1 = deconv(2, 2)
        self.upfeat1 = deconv(4, 4)
        
        self.conv2 = conv(4, 8, 3, 2)
        self.conv2a = conv(8, 8, 3)
        self.conv2b = conv(8, 8, 3)
        self.conv2c = conv(8, 8, 3)

        self.conv2_0 = conv(8, 16, 3)
        self.conv2_1 = conv(24, 16, 3)
        self.conv2_2 = conv(40, 4, 3)
        self.conv2_3 = conv(44, 8, 3)
        
        self.flow_predictor2 = flow_cnn(8)
        self.deconv2 = deconv(2, 2)
        self.upfeat2 = deconv(8, 8)

        self.dc_conv0 = conv(4, 12, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv1 = conv(12, 16, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv2 = conv(16, 32, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv3 = conv(32, 64,  kernel_size=3, stride=1, padding=8,  dilation=8)

        self.refine_flow = flow_cnn(64)
        
        
    def forward(self, I_t, I_tnext):
        
        X_t = None
        X_tnext = None
        X_hat_tnext = None
        X_hat_t = None
        f_t_tnext = None
        f_tnext_t = []
        
        # Image t downsampling
        I_t_x1 = self.conv1c(self.conv1b(self.conv1a(self.conv1(I_t))))
        I_t_x2 = self.conv2c(self.conv2b(self.conv2a(self.conv2(I_t_x1))))
        X_t = [I_t_x1, I_t_x2]
        
        # Image t+1 downsampling
        I_tnext_x1 = self.conv1c(self.conv1b(self.conv1a(self.conv1(I_tnext))))
        I_tnext_x2 = self.conv2c(self.conv2b(self.conv2a(self.conv2(I_tnext_x1))))
        X_tnext = [I_tnext_x1, I_tnext_x2]

        ### Image t -> t+1
        corr2 = corr4D(I_t_x2, I_tnext_x2)
        corr2 = self.leakyRELU(corr2)

        x = torch.cat((self.conv2_0(corr2), corr2), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = self.conv2_3(x)
        
        flow2 = self.flow_predictor2(x)
        upflow2 = self.deconv2(flow2)
        upfeat2 = self.upfeat2(x)

        I_tnext_x1_hat = warp(I_t_x1, upflow2*0.625)
        corr1 = corr4D(I_tnext_x1_hat, I_tnext_x1)
        corr1 = self.leakyRELU(corr1)

        x = torch.cat((I_tnext_x1, self.conv1_0(corr1), corr1, upfeat2), 1)
        x = torch.cat((self.conv1_1(x), x), 1)
        x = torch.cat((self.conv1_2(x), x), 1)
        x = self.conv1_3(x)

        flow1 = self.flow_predictor1(x)
        upflow1 = self.deconv1(flow1)
        upfeat1 = self.upfeat1(x)

        # Refining flow
        x = self.dc_conv3(self.dc_conv2(self.dc_conv1(self.dc_conv0(upfeat1))))
        upflow1 = upflow1 + self.refine_flow(x)

        I_tnext_hat = warp(I_t, upflow1*1.25)
        X_hat_tnext = [I_tnext_x1_hat]
        f_t_tnext = [flow1, flow2]
        
        ### Image t+1 -> t

        rcorr2 = corr4D(I_tnext_x2, I_t_x2)
        rcorr2 = self.leakyRELU(rcorr2)

        x = torch.cat((self.conv2_0(rcorr2), rcorr2), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = self.conv2_3(x)
        
        rev_flow2 = self.flow_predictor2(x)
        rev_upflow2 = self.deconv2(rev_flow2)
        rev_upfeat2 = self.upfeat2(x)

        I_t_x1_hat = warp(I_tnext_x1, rev_upflow2*0.625)
        rcorr1 = corr4D(I_t_x1_hat, I_t_x1)
        rcorr1 = self.leakyRELU(rcorr1)

        x = torch.cat((I_t_x1, self.conv1_0(rcorr1), rcorr1, rev_upfeat2), 1)
        x = torch.cat((self.conv1_1(x), x), 1)
        x = torch.cat((self.conv1_2(x), x), 1)
        x = self.conv1_3(x)

        rev_flow1 = self.flow_predictor1(x)
        rev_upflow1 = self.deconv1(rev_flow1)
        rev_upfeat1 = self.upfeat1(x)

        # Refining flow
        x = self.dc_conv3(self.dc_conv2(self.dc_conv1(self.dc_conv0(rev_upfeat1))))
        rev_upflow1 = rev_upflow1 + self.refine_flow(x)
        
        I_t_hat = warp(I_tnext, rev_upflow1*1.25)
        X_hat_t = [I_t_x1_hat]
        f_tnext_t = [rev_flow1, rev_flow2]
        
        return X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext, f_tnext_t, I_t_hat, I_tnext_hat, upflow1