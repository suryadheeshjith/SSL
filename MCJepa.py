#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[9]:


from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os


# These numbers are mean and std values for channels of natural images. 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

motion_transform_train = transforms.Compose([normalize])

content_transform_train = transforms.Compose([
                                    transforms.RandomResizedCrop(size=(160,240)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    normalize,
                                ])

class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, num_frames=22, motion_transform=motion_transform_train):
        self.data_dir = data_dir
        self.motion_transform = motion_transform
        self.num_frames = num_frames
        self.video_list = []

        self.count = 0
        for vid_dir in os.listdir(self.data_dir):
            self.video_list.append(self.data_dir +"/"+vid_dir)
            self.count +=1
            
            if self.count == 2500:
                break

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        
        video_dir = self.video_list[idx]
        frame_list = []
        
        for i in range(self.num_frames):
            image = read_image(video_dir + "/" + "image_"+str(i)+".png")
            image = image/255.0
            
            if self.motion_transform:
                image = self.motion_transform(image)
            
            frame_list.append(image)  

        return frame_list

class LabeledDataset(Dataset):
    def __init__(self, data_dir, num_frames=22, motion_transform=motion_transform_train):
        self.data_dir = data_dir
        self.motion_transform = motion_transform
        self.num_frames = num_frames
        self.video_list = []

        self.count = 0
        for vid_dir in os.listdir(self.data_dir):
            self.video_list.append(self.data_dir +"/"+vid_dir)
            self.count +=1

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        
        video_dir = self.video_list[idx]
        frame_list = []
        
        for i in range(self.num_frames):
            image = read_image(video_dir + "/" + "image_"+str(i)+".png")
            image = image/255.0
            
            if self.motion_transform:
                image = self.motion_transform(image)
            
            frame_list.append(image)
        
        label = -1
        if os.path.isfile(video_dir+"/mask.npy"):
            try:
                label = np.load(video_dir+"/mask.npy")
            except:
                return None, None
        

        return frame_list, label


# In[10]:


unlabeled_data = UnlabeledDataset("/dataset/dataset/unlabeled")
labeled_data = LabeledDataset("/dataset/dataset/train")
val_data = LabeledDataset("/dataset/dataset/val")


train_dataloader = DataLoader(unlabeled_data, batch_size=3, shuffle=True)
downstream_dataloader = DataLoader(labeled_data, batch_size=3, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)


# In[11]:


# from matplotlib import pyplot as plt
# plt.rcParams['figure.dpi'] = 100 # change dpi to make plots bigger

# def show_normalized_image(img, title=None):
#     plt.imshow(unnormalize(img).detach().cpu().permute(1, 2, 0).numpy())
#     plt.title(title)
#     plt.axis('off')

# show_normalized_image(unlabeled_data[10][0])


# ### Updated MC Jepa

# In[12]:


def conv(in_features, out_features, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1))

def flow_cnn(in_features):
    return nn.Sequential(
            nn.Conv2d(in_features, 2, kernel_size=3, stride=1, 
                        padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1))


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding)


class FEA(nn.Module):
    def __init__ (self, in_features):
        super().__init__()
        self.flow_predictor = flow_cnn(in_features)

    def forward(self, X_tnext, X_hat_tnext):
        correlation = self.corr4D(X_tnext, X_hat_tnext)
        batch_size, out_channels, ht, wd = X_tnext.shape

        mat_mul = torch.matmul(correlation, X_tnext.view(batch_size, out_channels, ht*wd, 1))
        mat_mul = mat_mul.view(batch_size, out_channels, ht, wd)
        x = self.flow_predictor(mat_mul)
        
        return x
    
    @staticmethod
    def corr4D(X_tnext, X_hat_tnext):
        batch, dim, ht, wd = X_tnext.shape
        X_tnext = X_tnext.view(batch, dim, ht*wd)
        X_hat_tnext = X_hat_tnext.view(batch, dim, ht*wd) 

        corr = torch.matmul(X_tnext.transpose(1,2), X_hat_tnext)
        corr = corr.view(batch, 1, ht*wd, ht*wd)
        corr = corr/torch.sqrt(torch.tensor(dim).float())
        return corr
    
#     @staticmethod
#     def upsample(flow, scale = 2, mode='bilinear'):
#         new_size = (scale * flow.shape[2], scale * flow.shape[3])
#         return  scale * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

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


# In[13]:


class MCJepa(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        self.conv1 = conv(in_features, 4, 3, 2)
        self.conv1a = conv(4, 4, 3)
        self.conv1b = conv(4, 4, 3)
        self.conv1c = conv(4, 4, 3)
        
        self.conv2 = conv(4, 8, 3, 2)
        self.conv2a = conv(8, 8, 3)
        self.conv2b = conv(8, 8, 3)
        self.conv2c = conv(8, 8, 3)
        
        self.conv3 = conv(8, 12, 3, 2)
        self.conv3a = conv(12, 12, 3)
        self.conv3b = conv(12, 12, 3)
        self.conv3c = conv(12, 12, 3)
        
        self.conv4 = conv(12, 16, 3, 2)
        self.conv4a = conv(16, 16, 3)
        self.conv4b = conv(16, 16, 3)
        self.conv4c = conv(16, 16, 3)
        
        
        self.fea4 = FEA(16)
        self.deconv4 = deconv(2, 2)
        
        self.fea3 = FEA(12)
        self.deconv3 = deconv(2, 2)
        
        self.fea2 = FEA(8)
        self.deconv2 = deconv(2, 2)
        
        self.fea1 = FEA(4)
        self.deconv1 = deconv(2, 2)
        
        
    def forward(self, I_t, I_tnext): # , I_tcrop):
        
        X_t = None
        X_tnext = None
        X_hat_tnext = None
        X_hat_t = None
        f_t_tnext = None
        f_tnext_t = []
        
        # Image t downsampling
        I_t_x1 = self.conv1c(self.conv1b(self.conv1a(self.conv1(I_t))))
        I_t_x2 = self.conv2c(self.conv2b(self.conv2a(self.conv2(I_t_x1))))
        I_t_x3 = self.conv3c(self.conv3b(self.conv3a(self.conv3(I_t_x2))))
        I_t_x4 = self.conv4c(self.conv4b(self.conv4a(self.conv4(I_t_x3))))
        
        X_t = [I_t_x1, I_t_x2, I_t_x3, I_t_x4]
#         X_t = [I_t_x1, I_t_x2, I_t_x3]
#         X_t = [I_t_x1, I_t_x2]
        
        # Image t+1 downsampling
        I_tnext_x1 = self.conv1c(self.conv1b(self.conv1a(self.conv1(I_tnext))))
        I_tnext_x2 = self.conv2c(self.conv2b(self.conv2a(self.conv2(I_tnext_x1))))
        I_tnext_x3 = self.conv3c(self.conv3b(self.conv3a(self.conv3(I_tnext_x2))))
        I_tnext_x4 = self.conv4c(self.conv4b(self.conv4a(self.conv4(I_tnext_x3))))
        
        X_tnext = [I_tnext_x1, I_tnext_x2, I_tnext_x3, I_tnext_x4]
#         X_tnext = [I_tnext_x1, I_tnext_x2, I_tnext_x3]
#         X_tnext = [I_tnext_x1, I_tnext_x2]

        ### Image t -> t+1
        flow4 = self.fea4(I_t_x4, I_tnext_x4)
        upflow4 = self.deconv4(flow4)
        
        I_tnext_x3_hat = warp(I_t_x3, upflow4*0.625)
        flow3 = self.fea3(I_tnext_x3_hat, I_tnext_x3)
        flow3 = flow3 + upflow4
        upflow3 = self.deconv3(flow3)
        
        I_tnext_x2_hat = warp(I_t_x2, upflow3*1.25)
        
        flow2 = self.fea2(I_tnext_x2_hat, I_tnext_x2)
        flow2 = flow2 + upflow3
        upflow2 = self.deconv2(flow2)
        
        I_tnext_x1_hat = warp(I_t_x1, upflow2*1.5)
        
        flow1 = self.fea1(I_tnext_x1_hat, I_tnext_x1)
        flow1 = flow1 + upflow2
        upflow1 = self.deconv1(flow1)
        
        I_tnext_hat = warp(I_t, upflow1*5.0)
        
        X_hat_tnext = [I_tnext_x1_hat, I_tnext_x2_hat, I_tnext_x3_hat]
#         X_hat_tnext = [I_tnext_x1_hat, I_tnext_x2_hat]
#         X_hat_tnext = [I_tnext_x1_hat]
        f_t_tnext = [flow1, flow2, flow3, flow4]
#         f_t_tnext = [flow1, flow2, flow3]
#         f_t_tnext = [flow1, flow2]
        
        ### Image t+1 -> t
        rev_flow4 = self.fea4(I_tnext_x4, I_t_x4)
        rev_upflow4 = self.deconv4(rev_flow4)
        
        I_t_x3_hat = warp(I_tnext_x3, rev_upflow4*0.625)
        
        rev_flow3 = self.fea3(I_t_x3_hat, I_t_x3)
        rev_flow3 = rev_flow3 + rev_upflow4
        rev_upflow3 = self.deconv3(rev_flow3)
        
        I_t_x2_hat = warp(I_tnext_x2, rev_upflow3*1.25)

        rev_flow2 = self.fea2(I_t_x2_hat, I_t_x2)
        rev_flow2 = rev_flow2 + rev_upflow3
        rev_upflow2 = self.deconv2(rev_flow2)
        
        I_t_x1_hat = warp(I_tnext_x1, rev_upflow2*2.5)
        
        rev_flow1 = self.fea1(I_t_x1_hat, I_t_x1)
        rev_flow1 = rev_flow1 + rev_upflow2
        rev_upflow1 = self.deconv1(rev_flow1)
        
        I_t_hat = warp(I_tnext, rev_upflow1*5.0)
        
        X_hat_t = [I_t_x1_hat, I_t_x2_hat, I_t_x3_hat]
        f_tnext_t = [rev_flow1, rev_flow2, rev_flow3, rev_flow4]
        
#         X_hat_t = [I_t_x1_hat, I_t_x2_hat]
#         f_tnext_t = [rev_flow1, rev_flow2, rev_flow3]

#         X_hat_t = [I_t_x1_hat]
#         f_tnext_t = [rev_flow1, rev_flow2]
        
        return X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext, f_tnext_t, I_t_hat, I_tnext_hat
    


# In[14]:


# a = torch.zeros([1, 3, 160, 240]).to(device)
# b = torch.zeros([1, 3, 160, 240]).to(device)

# mcmodel = MCJepa(3).to(device)
# mcmodel(a, b)


# In[15]:


def off_diagonal(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res

def vc_reg(X_t, X_tnext, lm, mu, nu):
    N = X_t.shape[0]
    C = X_t.shape[1]
    H = X_t.shape[2]
    W = X_t.shape[3] 
    D = C + H + W 
    mse_loss = nn.MSELoss()
    sim_loss = mse_loss(X_t, X_tnext)
    
    std_z_a = torch.sqrt(X_t.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(X_tnext.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1-std_z_a)) + torch.mean(F.relu(1-std_z_b))

    X_t = X_t - X_t.mean(dim=0)
    X_tnext = X_tnext - X_tnext.mean(dim=0)
    cov_z_a = torch.matmul(X_t.view(N, C, W, H), X_t)/ (N-1)
    cov_z_b = torch.matmul(X_tnext.view(N, C, W, H), X_tnext)/ (N-1)
    conv_loss = (off_diagonal(cov_z_a).pow_(2).sum()/D) + (off_diagonal(cov_z_b).pow_(2).sum()/D) 
        
    loss = lm*sim_loss + mu*std_loss + nu*conv_loss
    return loss

def cycle_loss_fn(f_t_tnext, f_tnext_t, X_t, X_tnext, lambda_a, lambda_b):
    loss_cycle_A = torch.tensor(0.0).to(device)
    loss_cycle_B = torch.tensor(0.0).to(device)
    for i in range(1, len(X_t)):
        loss_cycle_A += F.l1_loss(warp(X_t[i], f_t_tnext[i]), X_tnext[i]) * lambda_a
        loss_cycle_B += F.l1_loss(warp(X_tnext[i], f_tnext_t[i]), X_t[i]) * lambda_b
        
    return loss_cycle_A + loss_cycle_B


# In[16]:


def MCJepa_criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext, f_tnext_t, I_hat_t, I_hat_tnext, img1, img2, lm, mu, nu, lambda_a, lambda_b,show=False):
  
    mse_loss = nn.MSELoss()    
    rec_loss = nn.MSELoss()
    reg_loss = nn.MSELoss()
    
    regress_loss_forward = torch.tensor(0.0).to(device)
    regress_loss_backward = torch.tensor(0.0).to(device)
    for i in range(len(X_hat_tnext)):
        regress_loss_forward += reg_loss(X_hat_tnext[i], X_tnext[i])
        regress_loss_backward += reg_loss(X_hat_t[i], X_t[i])

    reconst_loss_forward = rec_loss(I_hat_tnext, img2)
    reconst_loss_backward = rec_loss(I_hat_t, img1)
    vc_reg_loss = torch.tensor(0.0).to(device)
    
    for i in range(len(X_t)):
        vc_reg_loss += vc_reg(X_t[i], X_tnext[i], lm, mu, nu)

    cycle_loss = cycle_loss_fn(f_t_tnext, f_tnext_t, X_t, X_tnext, lambda_a, lambda_b)
    
    if show:
        print("regress_loss_forward: ",50*regress_loss_forward)
        print("regress_loss_backward: ",50*regress_loss_backward)
        print("reconst_loss_forward: ",1000*reconst_loss_forward)
        print("reconst_loss_backward: ",1000*reconst_loss_backward)
        print("vc_reg_loss: ",vc_reg_loss)
        print("cycle_loss: ",50*cycle_loss)

        print("\n\n")

    
    return 50*regress_loss_forward + 50*regress_loss_backward + 1000*reconst_loss_forward + 1000*reconst_loss_backward + 50*cycle_loss + vc_reg_loss 



# In[17]:


from tqdm import tqdm

def train_MCJepa(model, epochs, dataloader, criterion, optimizer, scheduler=None):
    model.train()

    train_losses = []

    best_loss = float("inf")
    best_model = model.state_dict()

    for e in range(epochs):
        total_train_loss = 0.0
        total_train_correct = 0.0
        
        pbar = tqdm(dataloader, leave=False)

        for j,batch in enumerate(pbar):
            if j == 4333: 
                break

            frame_list = batch
            total_train_loss = 0.0
            
            for i in range(len(frame_list) - 1):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)
#                 print(img1.shape)
                X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext = model(img1, img2)

                loss = criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                                f_tnext_t, I_hat_t, I_hat_tnext, img1, img2,\
                                lm, mu, nu, lambda_a, lambda_b,(i+21*j)%660 == 0)
            
                total_train_loss += loss.item()
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#                 scheduler.step(e+j/len(dataloader))
            
            pbar.set_postfix({'Video Loss': total_train_loss/(len(frame_list)-1)})

            if total_train_loss/(len(frame_list)-1) < best_loss:
                best_loss = total_train_loss/(len(frame_list)-1)
                best_model = model.state_dict()
                
            if j % 30 == 0 and j > 0:
                torch.save(best_model,"best_model_adam.pth")
                pbar.set_postfix({'Video Loss': total_train_loss/(len(frame_list)-1), 'Saved model with loss': best_loss})
            
        pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved model at j': j})
        torch.save(best_model, "best_model_adam.pth")


# In[18]:


# Constants to figure out later
in_features = 3 
lm, mu, nu, lambda_a, lambda_b = 0.02, 0.02, 0.01, 1, 1


# In[19]:


MCJepa_model = MCJepa(in_features).to(device)
optimizer = optim.Adam(MCJepa_model.parameters(), lr = 10e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,5)


# In[20]:


train_MCJepa(MCJepa_model, 10, train_dataloader, MCJepa_criterion, optimizer) # Training the MC JEPA


# In[ ]:


# PATH = "best_model.pth"
# MCJepa_model = MCJepa(in_features).to(device)
# MCJepa_model.load_state_dict(torch.load(PATH))


# In[ ]:


# flow_1 = 0
# flow_2 = 0


# In[ ]:


# for batch in train_dataloader:
# #     MCJepa_model.reset_flows()
#     frame_list = batch
#     img1 = frame_list[0].to(device)
#     img2 = frame_list[1].to(device)

#     X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
#                 f_tnext_t, I_hat_t, I_hat_tnext = MCJepa_model(img1, img2)

#     loss = MCJepa_criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
#                     f_tnext_t, I_hat_t, I_hat_tnext, img1, img2,\
#                      lm, mu, nu, lambda_a, lambda_b)
    
#     print("Loss: ", loss.item())
# #     flow_1 = f_t_tnext
#     flow_2 = f_t_tnext
# #     print(f_t_tnext[0])
#     print((f_t_tnext[0] == 0).all())
    

#     show_normalized_image(img1[0])
#     plt.show()
#     show_normalized_image(I_hat_t[0])
#     plt.show()
#     show_normalized_image(img2[0])
#     plt.show()
#     show_normalized_image(I_hat_tnext[0])
#     plt.show()
#     show_normalized_image(torch.square(I_hat_t[0]-img2[0]))
#     plt.show()
#     show_normalized_image(torch.square(I_hat_tnext[0]-img1[0]))
#     plt.show()
#     show_normalized_image(torch.square(img2[0]-img1[0]))
    
# #     print(Y1)
#     break
    
    


# ### Downstream Task

# In[ ]:


# for param in MCJepa_model.parameters():
#     param.requires_grad = False


# In[ ]:


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()

#         if not mid_channels:
#             mid_channels = (out_channels + in_channels)//2
            
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
 
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
    
# class UNet(nn.Module):
#     def __init__(self, n_channels = 3, n_classes = 49, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(512, 1024 // factor))
#         self.up1 = (Up(1024, 512 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


# In[ ]:


# def train_fine_tune(downstream_model, epochs, dataloader, criterion, optimizer):
#     downstream_model.train()

#     train_losses = []

#     best_loss = float("inf")
# #     best_model = {}
#     best_model = downstream_model.state_dict()

#     for _ in range(epochs):
#         total_train_loss = 0.0
#         total_train_correct = 0.0

#         pbar = tqdm(dataloader, leave=False)

#         for j, batch in enumerate(pbar):
          
#             if j == 333:
#                 break
                
#             frame_list, mask_list = batch[0], batch[1] # TODO
#             total_train_loss = 0.0
            

#             for i in range(len(frame_list) - 1):
#                 img1 = frame_list[i].to(device)
#                 img2 = frame_list[i+1].to(device)
#                 mask_list = mask_list.type(torch.LongTensor).to(device)


#                 logits = downstream_model(img1)
#                 loss = criterion(logits, mask_list[:,i])

#                 total_train_loss += loss.item()

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1)})

#             if total_train_loss/(len(frame_list)-1) < best_loss:
#                 best_loss = total_train_loss/(len(frame_list)-1)
#                 best_model = downstream_model.state_dict()

#             if j%25 == 0:
#                 torch.save(best_model,"best_downstream_model.pth")
#                 pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved downstream model with loss': best_loss})
          
#         pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved model at j': j})
#         torch.save(downstream_model.state_dict(), "downstream_model.pth")
          
#     torch.save(best_model,"best_downstream_model.pth")


# In[ ]:


# in_features_downstream = 16

# downstream_model = UNet().to(device)
# downstream_optimizer = optim.RMSprop(downstream_model.parameters(),
#                           lr=1e-5, weight_decay=1e-8, momentum=0.999, foreach=True)

criterion = nn.CrossEntropyLoss()


# In[ ]:


# train_fine_tune(downstream_model, 5, downstream_dataloader, criterion, downstream_optimizer)


# In[ ]:


# PATH = "best_downstream_model.pth"
# downstream_model = UNet().to(device)
# downstream_model.load_state_dict(torch.load(PATH))


# In[ ]:


# !pip install torchmetrics
# import torchmetrics


# In[ ]:


# jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)


# In[ ]:


# def test(downstream_model, JepaModel, epochs, dataloader, criterion):
#     train_losses = []

#     best_loss = float("inf")
#     best_model = downstream_model.state_dict()

#     for _ in range(epochs):
#         total_train_loss = 0.0

#         pbar = tqdm(dataloader, leave=False)

#         for j,batch in enumerate(pbar):
          
#             frame_list, mask_list = batch[0], batch[1]
#             total_train_loss = 0.0
            

#             for i in range(len(frame_list) - 1):
#                 img1 = frame_list[i].to(device)
#                 img2 = frame_list[i+1].to(device)
#                 mask_list = mask_list.type(torch.LongTensor).to(device)

#                 mask_pred = downstream_model(img1)
# #               
#                 if i == 20:
#                     print(mask_pred.shape, mask_list[:,i].shape)
#                     print(mask_list[:,i][0])

#                     print(jaccard(mask_pred, mask_list[:,i]))
#                     print((torch.argmax(mask_pred[0], dim=0) == 0).all())
#                     plt.imshow(mask_list[0][i].cpu())
#                     plt.show()
#                     plt.imshow(mask_pred.argmax(dim=1)[0].float().cpu())
#                     plt.show()
#                 loss = criterion(mask_pred, mask_list[:,i])
#                 total_train_loss += loss.item()

#             pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1)})


# In[ ]:


# test(downstream_model, MCJepa_model, 1, val_dataloader, criterion)


# In[ ]:


# def real_test(downstream_model, JepaModel, epochs, dataloader, criterion, scale = 0.1):
    

#     train_losses = []

#     best_loss = float("inf")
#     best_model = downstream_model.state_dict()

#     for _ in range(epochs):
#         total_train_loss = 0.0

#         pbar = tqdm(dataloader, leave=False)
        
#         avg_jacc = 0.0

#         for j,batch in enumerate(pbar):
          
#             frame_list, mask_list = batch[0], batch[1] # TODO
#             total_train_loss = 0.0
#             X_tconcat = None
#             I_hat_t = None
#             I_hat_tnext = None
#             I_hat_t = None
            
#             final_flow = 0.0
#             for i in range(11):
#                 img1 = frame_list[i].to(device)
#                 img2 = frame_list[i+1].to(device)
#                 mask_list = mask_list.type(torch.LongTensor).to(device)


#                 X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
#                 f_tnext_t, I_hat_t, I_hat_tnext = JepaModel(img1, img2)
#                 final_flow = JepaModel.deconv1(f_t_tnext[0])*5.0 + scale*final_flow
                
            
#             mask_pred = downstream_model(frame_list[11].to(device))
# #             mask_pred_ = downstream_model(frame_list[11].to(device))
            
#             for i in range(11):
#                 X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
#                 f_tnext_t, I_hat_t, I_hat_tnext = JepaModel(I_hat_t, I_hat_tnext)
# #                 print(f_t_tnext[0].shape, mask_pred.shape)
#                 flow = JepaModel.deconv1(f_t_tnext[0]) 
#                 flow *= 0
#                 mask_pred = warp(mask_pred, flow)

# #             mask_pred = warp(mask_pred, final_flow)
            
# #             plt.imshow(unnormalize(img1[0]).permute(1, 2, 0).cpu())
# #             plt.show()
# #             plt.imshow(unnormalize(I_hat_tnext[0]).permute(1, 2, 0).cpu())
# #             plt.show()
# #             mask_pred = downstream_model(I_hat_tnext)

# #             print(f_t_tnext)
# #             print(jaccard(mask_pred, mask_list[:,21]))
# #             print((torch.argmax(mask_pred[0], dim=0) == torch.argmax(mask_pred_[0].cpu())).all())
            
# #             plt.imshow(mask_list[0][21].cpu())
# #             plt.show()
# #             plt.imshow(torch.argmax(mask_pred[0].cpu(), dim=0))
# #             plt.show()
# #             plt.imshow(torch.argmax(mask_pred_[0].cpu(), dim=0))
# #             plt.show()
#             loss = criterion(mask_pred, mask_list[:,21])
#             total_train_loss += loss.item()
            
# #             plt.imshow(unnormalize(img1[0]).permute(1, 2, 0).cpu())
# #             plt.show()
# #             plt.imshow(unnormalize(I_hat_tnext[0]).permute(1, 2, 0).cpu())
# #             plt.show()
# #             mask_pred = downstream_model(I_hat_tnext)

# #             print(f_t_tnext)
# #             print(jaccard(mask_pred, mask_list[:,21]))
# #             print((torch.argmax(mask_pred[0], dim=0) == torch.argmax(mask_pred_[0].cpu())).all())
            
# #             plt.imshow(mask_list[0][21].cpu())
# #             plt.show()
# #             plt.imshow(torch.argmax(mask_pred[0].cpu(), dim=0))
# #             plt.show()
# #             plt.imshow(torch.argmax(mask_pred_[0].cpu(), dim=0))
# #             plt.show()

#             avg_jacc += jaccard(mask_pred, mask_list[:,21])
#             pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'avg_jaccard': avg_jacc.item() / (j+1)}) 


# In[ ]:


# real_test(downstream_model, MCJepa_model, 1, val_dataloader, criterion)


# In[ ]:




