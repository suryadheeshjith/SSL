#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


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

motion_transform_train = transforms.Compose([
                                    # transforms.Resize((256, 256)),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(hue=.2, saturation=.2, contrast=.2),
                                    # transforms.RandomRotation(20, interpolation=Image.BILINEAR),
                                    # transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    normalize,
                                ])

content_transform_train = transforms.Compose([
                                    transforms.RandomResizedCrop(size=(160,240)),
                                    # transforms.Resize((256, 256)),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(hue=.2, saturation=.2, contrast=.2),
#                                     transforms.RandomRotation(20, interpolation=Image.BILINEAR),
                                    transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    normalize,
                                ])



class CompDataset(Dataset):
    def __init__(self, data_dir, num_frames=22, motion_transform=motion_transform_train, content_transform=content_transform_train):
        self.data_dir = data_dir
        self.motion_transform = motion_transform
        self.content_transform = content_transform
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
        frame_list_aug = []
        
        for i in range(self.num_frames):
            image = read_image(video_dir + "/" + "image_"+str(i)+".png")
            image = image/255.0
            image_aug= None
            
            if self.motion_transform:
                image = self.motion_transform(image)
            
            if self.content_transform:
                image_aug = self.content_transform(image)
            
            frame_list.append(image)
            frame_list_aug.append(image_aug)

        label = -1
        
        
        if os.path.isfile(video_dir+"/mask.npy"):
            try:
                label = np.load(video_dir+"/mask.npy")
            except:
                return None, None, None
        

        return frame_list, frame_list_aug, label


# In[4]:


unlabeled_data = CompDataset("/dataset/dataset/unlabeled")
labeled_data = CompDataset("/dataset/dataset/train")
val_data = CompDataset("/dataset/dataset/val")


train_dataloader = DataLoader(unlabeled_data, batch_size=3, shuffle=True)
downstream_dataloader = DataLoader(labeled_data, batch_size=3, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)


# In[5]:


from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 100 # change dpi to make plots bigger

def show_normalized_image(img, title=None):
    plt.imshow(unnormalize(img).detach().cpu().permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis('off')

show_normalized_image(unlabeled_data[10][0][0])


# In[6]:


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

    # if W==128:
      # np.save('mask.npy', mask.cpu().data.numpy())
      # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask


# In[7]:


class FEA(nn.Module):
    def __init__ (self, in_features, out_features):
        super().__init__()
        self.cnn = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(2)
        
        self.cnn2 = nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(2)

    def forward(self, X_tnext, X_hat_tnext, f_tprev_t):
        correlation = self.corr4D(X_tnext, X_hat_tnext)
        batch_size, out_channels, ht, wd = X_tnext.shape
#         mat_mul = torch.matmul(correlation.view(batch_size, 1, ht*wd, ht*wd), X_tnext.view(batch_size, ht*wd, out_channels, 1))
#         print(correlation.shape)
        mat_mul = torch.matmul(correlation, X_tnext.view(batch_size, out_channels, ht*wd, 1))
        mat_mul = mat_mul.view(batch_size, out_channels, ht, wd)
        x = self.maxPool(self.relu(self.bn(self.cnn(mat_mul))))
#         x = self.maxPool2(self.relu2(self.bn2(self.cnn2(x))))
        x = self.upsample(x)
        x += 0.2*f_tprev_t
        x.requires_grad
        return x
    
    @staticmethod
    def corr4D(X_tnext, X_hat_tnext):
        batch, dim, ht, wd = X_tnext.shape
        X_tnext = X_tnext.view(batch, dim, ht*wd)
        X_hat_tnext = X_hat_tnext.view(batch, dim, ht*wd) 

        corr = torch.matmul(X_tnext.transpose(1,2), X_hat_tnext)
#         corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr.view(batch, 1, ht*wd, ht*wd)
        corr = corr/torch.sqrt(torch.tensor(dim).float())
        return corr
    
    @staticmethod
    def upsample(flow, scale = 2, mode='bilinear'):
        new_size = (scale * flow.shape[2], scale * flow.shape[3])
        return  scale * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


# In[8]:


class Encoder(nn.Module):
    def __init__ (self, in_features, out_features):
        super().__init__()
        self.cnn = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_features)
        self.maxPool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        self.cnn2 = nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_features)
        self.maxPool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        
        self.cnn3 = nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(out_features)
        self.maxPool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()

    def forward(self, I):
        x = self.maxPool(self.relu(self.bn(self.cnn(I))))
        x = self.maxPool2(self.relu2(self.bn2(self.cnn2(x))))
        x = self.maxPool3(self.relu3(self.bn3(self.cnn3(x))))
        return x


# In[9]:


class MCJepa(nn.Module):
    def __init__(self, numLayers, in_features, out_features, out_features_fea):
        super().__init__()
        self.numLayers = numLayers;
        self.enc1 = nn.ModuleList([Encoder(in_features, out_features) for i in range(numLayers)])
        # self.enc2 = [Encoder(in_features, out_features) for i in range(numLayers)]
        self.fea = nn.ModuleList([FEA(out_features, out_features_fea) for i in range(numLayers)])
        self.f_tprev_t = 0.0 # TODO: this has to be a tensor
        self.f_tnext_t = 0.0 # TODO: this has to be a tensor
        
        # content

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(out_features*numLayers, out_features, kernel_size=(2,2), stride=(2,2)),
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(out_features, out_features, kernel_size=(2,2), stride=(2,2)),
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(inplace=True),
#         )
        
        self.expander_in_features = self.numLayers*out_features*20*30
        
        self.expander = nn.Sequential(
            nn.Linear(self.expander_in_features,self.expander_in_features*4)
        )

    def reset_flows(self):
        self.f_tprev_t = 0.0
        self.f_tnext_t = 0.0 


    def forward(self, I_t, I_tnext, I_tcrop):
        X_hat_tnext = []
        X_hat_t = []
        X_t = []
        X_tnext = []
        f_t_tnext = []
        f_tnext_t = []

        
        # motion component

        for i in range(self.numLayers): # Feature extracting layer
            X_t.append(self.enc1[i](I_t))
            X_tnext.append(self.enc1[i](I_tnext))

        for i in range(self.numLayers): # Finding the flow t -> t + 1
            if i == 0:
                f_t_tnext.append(self.fea[i](X_tnext[i], X_t[0], self.f_tprev_t))
            else:
                X_hat_tnext.append(warp(X_t[i], f_t_tnext[i-1]))
                f_t_tnext.append(self.fea[i](X_tnext[i], X_hat_tnext[i-1], f_t_tnext[i-1]))

        self.f_tprev_t = torch.detach(f_t_tnext[-1]) # Remembering the flow info across frames
        
        I_hat_tnext = warp(I_t, FEA.upsample(f_t_tnext[-1], 8)) # Predicting future frame

        for i in range(self.numLayers): # Finding the flow t + 1 -> t
            if i == 0:
                f_tnext_t.append(self.fea[i](X_t[i], X_tnext[0], self.f_tnext_t))
            else:
                X_hat_t.append(warp(X_tnext[i], f_tnext_t[i-1]))
                f_tnext_t.append(self.fea[i](X_t[i], X_hat_t[i-1], f_tnext_t[i-1]))

        self.f_tnext_t = torch.detach(f_tnext_t[-1]) # Remembering the flow info across backward frames

        I_hat_t = warp(I_tnext, FEA.upsample(f_tnext_t[-1], 8)) # Predicting past frame
        
        # content component
        
        X_tconcat = X_t[0]
        X_tcrop = self.enc1[0](I_tcrop)

        for i in range(1, self.numLayers): # Feature extracting layer
            X_tconcat = torch.concat((X_tconcat,X_t[i]),dim=1)
            X_tcrop = torch.concat((X_tcrop,self.enc1[i](I_tcrop)),dim=1)
            

        X_tconcat_shape = X_tconcat.shape
        X_tconcat_flattened = X_tconcat.view(X_tconcat.shape[0],-1)
        X_tcrop_flattened = X_tcrop.view(X_tcrop.shape[0],-1)
        
        Y1 = self.expander(X_tconcat_flattened)
        Y2 = self.expander(X_tcrop_flattened)
        
        Y1 = Y1.view(X_tconcat_shape[0], X_tconcat_shape[1], 2*X_tconcat_shape[2], 2*X_tconcat_shape[3])
        Y2 = Y2.view(X_tconcat_shape[0], X_tconcat_shape[1], 2*X_tconcat_shape[2], 2*X_tconcat_shape[3])
        
#         Y1 = self.deconv(X_tcrop)
#         Y2 = self.deconv(X_tconcat)
        
#         Y1 = X_tcrop
#         Y2 = X_tconcat

#         print(Y1.shape)
        
        return X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext, f_tnext_t, I_hat_t, I_hat_tnext, Y1, Y2, X_tcrop, X_tconcat


# In[10]:


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


# In[11]:


def MCJepa_criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext, f_tnext_t, I_hat_t, I_hat_tnext, img1, img2, Y1, Y2, X_tcrop, X_tconcat, lm, mu, nu, lambda_a, lambda_b):
  
    mse_loss = nn.MSELoss()    
    rec_loss = nn.MSELoss()
    reg_loss = nn.MSELoss()
    
    regress_loss_forward = torch.tensor(0.0).to(device)
    regress_loss_backward = torch.tensor(0.0).to(device)
    for i in range(len(X_hat_tnext)):
        regress_loss_forward += torch.sqrt(reg_loss(X_hat_tnext[i], X_tnext[i+1]))
        regress_loss_backward += torch.sqrt(reg_loss(X_hat_t[i], X_t[i+1]))
    

    reconst_loss_forward = rec_loss(I_hat_tnext, img2) 
    reconst_loss_backward = rec_loss(I_hat_t, img1) 
    vc_reg_loss = torch.tensor(0.0).to(device)
    
    for i in range(len(X_t)):
        vc_reg_loss += vc_reg(X_t[i], X_tnext[i], lm, mu, nu)

    cycle_loss = cycle_loss_fn(f_t_tnext, f_tnext_t, X_t, X_tnext, lambda_a, lambda_b)


    # content 

    vc_content_loss = vc_reg(X_tcrop, X_tconcat, lm, mu, nu)
    vc_content_expander_loss = vc_reg(Y1, Y2, lm, mu, nu)

    content_regress_loss = torch.sqrt(reg_loss(Y1,Y2))
    
    
#     print("Regress Loss Forward: ", regress_loss_forward)
#     print("Regress Loss Backward: ", regress_loss_backward)
#     print("Reconst Loss Forward: ", reconst_loss_forward)
#     print("Reconst Loss Backward: ", reconst_loss_backward)
#     print("VicReg Motion Loss: ", vc_reg_loss)
#     print("Cycle Loss: ", cycle_loss)
#     print("VicReg Content Loss: ", vc_content_loss)
#     print("VicReg Content Expander Loss: ", vc_content_expander_loss)
#     print("Content Regress Loss: ", content_regress_loss)
    

    return regress_loss_forward + regress_loss_backward + reconst_loss_forward + reconst_loss_backward + vc_reg_loss + cycle_loss + vc_content_loss + vc_content_expander_loss + content_regress_loss



# In[12]:


from tqdm import tqdm

def train_MCJepa(model, epochs, dataloader, criterion, optimizer):
    model.train()

    train_losses = []

    best_loss = float("inf")
    best_model = model.state_dict()

    for _ in range(epochs):
        total_train_loss = 0.0
        total_train_correct = 0.0
        
        pbar = tqdm(dataloader, leave=False)

        for j,batch in enumerate(pbar):
            
            if j == 4333: 
                break

            frame_list, frame_list_aug, label = batch[0], batch[1], batch[2]
            total_train_loss = 0.0
            model.reset_flows()
            
            for i in range(len(frame_list) - 1):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)
                img3 = frame_list_aug[i].to(device)
                
                

                X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext, Y1, Y2, X_tcrop,\
                                            X_tconcat = model(img1, img2, img3)

                loss = criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                                f_tnext_t, I_hat_t, I_hat_tnext, img1, img2,\
                                 Y1, Y2, X_tcrop, X_tconcat, lm, mu, nu, lambda_a, lambda_b)
            
                total_train_loss += loss.item()
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1)})

            if total_train_loss/(len(frame_list)-1) < best_loss:
                best_loss = total_train_loss/(len(frame_list)-1)
                best_model = model.state_dict()
                
                if j%15 == 0:
                    torch.save(best_model,"best_model.pth")
                    pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved model with loss': best_loss})
            
            
        pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved model at j': j})
        torch.save(model.state_dict(), "model.pth")
            


        torch.save(best_model,"best_model_.pth")


# In[13]:


# Constants to figure out later
numLayers = 5
in_features = 3 
out_features = 4
out_features_fea = 2
lm, mu, nu, lambda_a, lambda_b = 0.02, 0.02, 0.01, 1, 1


MCJepa_model = MCJepa(numLayers, in_features, out_features, out_features_fea).to(device)
optimizer = optim.Adam(MCJepa_model.parameters(), lr = 0.0001)


# In[14]:


train_MCJepa(MCJepa_model, 1, train_dataloader, MCJepa_criterion, optimizer) # Training the MC JEPA


# In[15]:


PATH = "best_model.pth"
MCJepa_model = MCJepa(numLayers, in_features, out_features, out_features_fea).to(device)
MCJepa_model.load_state_dict(torch.load(PATH))


# In[16]:


flow_1 = 0
flow_2 = 0


# In[17]:


for batch in train_dataloader:
    MCJepa_model.reset_flows()
    frame_list, frame_list_aug, label = batch[0], batch[1], batch[2]
    img1 = frame_list[0].to(device)
    img2 = frame_list[1].to(device)
    img3 = frame_list_aug[0].to(device)
    X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext, Y1, Y2, X_tcrop,\
                                            X_tconcat = MCJepa_model(img1, img2, img3)

    loss = MCJepa_criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                    f_tnext_t, I_hat_t, I_hat_tnext, img1, img2,\
                     Y1, Y2, X_tcrop, X_tconcat, lm, mu, nu, lambda_a, lambda_b)
    print("Loss: ", loss.item())
#     flow_1 = f_t_tnext
    flow_2 = f_t_tnext
    print(f_t_tnext[0].shape)
    print((f_t_tnext[0] == 0).all())
    

#     show_normalized_image(img1[0])
#     plt.show()
#     show_normalized_image(I_hat_t[0])
#     plt.show()
#     show_normalized_image(img2[0])
#     plt.show()
#     show_normalized_image(I_hat_tnext[0])
#     plt.show()
# #     print(Y1)
#     show_normalized_image(X_tconcat[1][0:3])
#     plt.show()
#     show_normalized_image(X_tconcat[0][1:4])
#     plt.show()
#     show_normalized_image(X_tconcat[0][4:7])
#     plt.show()
    
    break


# ### Downstream Task

# In[18]:


for param in MCJepa_model.parameters():
    param.requires_grad = False


# In[19]:


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=49):
        super().__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Up5 = up_conv(ch_in=img_ch,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(128,output_ch,kernel_size=1,stride=1,padding=0)
        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        # encoding path
#         print(x.shape)
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

        # decoding + concat path
        x = self.Up5(x)
#         d5 = torch.cat((x4,d5),dim=1)
        
        x = self.Up_conv5(x)
        
        x = self.Up4(x)
#         d4 = torch.cat((x3,d4),dim=1)
        x = self.Up_conv4(x)
        
        x = self.Up3(x)
# #         d3 = torch.cat((x2,d3),dim=1)
        x = self.Up_conv3(x)

#         x = self.Up2(x)
# #         d2 = torch.cat((x1,d2),dim=1)
#         x = self.Up_conv2(x)

        x = self.Conv_1x1(x)

#         x =  self.softmax(x)
#         print(x.shape)
        return x


# In[20]:


def downstream_criterion(mask_pred, mask):
  # jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

  # return jaccard(mask_pred, mask)
    loss = nn.CrossEntropyLoss()
    return loss(mask_pred, mask)

def soft_dice_loss(y_pred, y_true, epsilon=1e-6): 
    C = 49
    y_true=y_true.clone()
    y_true.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(y_true.size(0), C, y_true.size(2), y_true.size(3)).zero_()
    one_hot.scatter_(1, y_true, 1) 
    
    y_pred = y_pred.permute(0, 2, 3, 1)
    y_true = one_hot.permute(0, 2, 3, 1)
    
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)
    
    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

def focal_loss(pred, target, alpha = 0.25, gamma = 2):
    C = 49
    ce_loss = F.cross_entropy(pred, target, reduction='mean', weight=torch.ones(C).to(device))
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss)
    
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_loss = alpha_t * focal_loss
        
    focal_loss = focal_loss.sum()
    
    return focal_loss



# In[21]:


def train_fine_tune(downstream_model, JepaModel, epochs, dataloader, criterion, optimizer):
    downstream_model.train()

    train_losses = []

    best_loss = float("inf")
    best_model = downstream_model.state_dict()

    for _ in range(epochs):
        total_train_loss = 0.0
        total_train_correct = 0.0

        JepaModel.reset_flows()

        pbar = tqdm(dataloader, leave=False)

        for j, batch in enumerate(pbar):
          
            if j == 333:
                break
                
            frame_list, frame_list_aug, mask_list = batch[0], batch[1], batch[2] # TODO
            total_train_loss = 0.0
            

            for i in range(len(frame_list) - 1):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)
                img3 = frame_list_aug[i].to(device)
                mask_list = mask_list.type(torch.LongTensor).to(device)


                X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext, Y1, Y2, X_tcrop,\
                                          X_tconcat = JepaModel(img1, img2, img3)

#                 mask_pred = downstream_model(Y1)
                mask_pred = downstream_model(X_tconcat)
                loss = criterion(mask_pred, mask_list[:,i])

                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1)})

            if total_train_loss/(len(frame_list)-1) < best_loss:
                best_loss = total_train_loss/(len(frame_list)-1)
                best_model = downstream_model.state_dict()

                if j%25 == 0:
                    torch.save(best_model,"best_downstream_model.pth")
                    pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved downstream model with loss': best_loss})
          
        pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'Saved model at j': j})
        torch.save(downstream_model.state_dict(), "downstream_model.pth")
          
    torch.save(best_model,"best_downstream_model.pth")


# In[ ]:


in_features_downstream = numLayers*out_features

downstream_model = U_Net(in_features_downstream).to(device)
downstream_optimizer = optim.Adam(downstream_model.parameters(), lr = 0.001)
train_fine_tune(downstream_model, MCJepa_model, 10, downstream_dataloader, focal_loss, downstream_optimizer)



# In[ ]:


PATH = "best_downstream_model.pth"
downstream_model = U_Net(in_features_downstream).to(device)
downstream_model.load_state_dict(torch.load(PATH))



