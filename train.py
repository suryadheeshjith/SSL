import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 100 # change dpi to make plots bigger

from FJEPA import FJepa,warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# !gdown https://drive.google.com/uc?id=1tYDQwL-0Ve3wUnl9U2UhpwuWOMDO9M6h # Surya
# # !gdown https://drive.google.com/uc?id=1gZkg_cf_BCH0E8Iv3w--jnt71yAD9dTE # Mayank
# # !gdown https://drive.google.com/uc?id=1nAH-is1PiRwrKtfstGLsiE1P2pHouU5R # Pranav

# !unzip Dataset_Student_V2_.zip

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

def FJepa_criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext, f_tnext_t, I_hat_t, I_hat_tnext, img1, img2, lm, mu, nu, lambda_a, lambda_b,show=False):
  
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
    
    return 50*regress_loss_forward + 50*regress_loss_backward + 1000*reconst_loss_forward + 1000*reconst_loss_backward + 50*cycle_loss + vc_reg_loss

def train_FJepa(model, epochs, dataloader, criterion, optimizer, model_name, frame_diff = 1, scheduler=None):
    model.train()

    train_losses = []

    best_loss = float("inf")
    best_model = model.state_dict()
    frame_list = None
    j=0

    for e in range(epochs):
        print("Epoch ",e)
        total_train_loss = 0.0
        total_train_correct = 0.0
        
        pbar = tqdm(dataloader, leave=False)

        for j,batch in enumerate(pbar):
            if j == 833: 
                break

            frame_list = batch
            total_train_loss = 0.0
            for i in range(len(frame_list) - frame_diff):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+frame_diff].to(device)
#                 print(img1.shape)
                X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext, upflow1 = model(img1, img2)

                loss = criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                                f_tnext_t, I_hat_t, I_hat_tnext, img1, img2,\
                                lm, mu, nu, lambda_a, lambda_b,(i+21*j)%660 == 0)
            
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
            
            pbar.set_postfix({'Video Loss': total_train_loss/(len(frame_list)-frame_diff)})

            if total_train_loss/(len(frame_list)-frame_diff) < best_loss:
                best_loss = total_train_loss/(len(frame_list)-frame_diff)
                best_model = model.state_dict()
                
            if j % 30 == 0 and j > 0:
                torch.save(best_model,"best_"+model_name+".pth")
                pbar.set_postfix({'Video Loss': total_train_loss/(len(frame_list)-frame_diff), 'Saved model with loss': best_loss})
            
        pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-frame_diff), 'Saved model at j': j})
        torch.save(model.state_dict(), model_name+".pth")


def visualize_flow(FJepa_model, dataloader):
    flow_1 = 0
    flow_2 = 0

    for batch in dataloader:
    #     FJepa_model.reset_flows()
        frame_list = batch
        img1 = frame_list[0].to(device)
        img2 = frame_list[1].to(device)

        X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                    f_tnext_t, I_hat_t, I_hat_tnext = FJepa_model(img1, img2)

        loss = FJepa_criterion(X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                        f_tnext_t, I_hat_t, I_hat_tnext, img1, img2,\
                        lm, mu, nu, lambda_a, lambda_b)
        
        print("Loss: ", loss.item())
    #     flow_1 = f_t_tnext
        flow_2 = f_t_tnext
    #     print(f_t_tnext[0])
        print((f_t_tnext[0] == 0).all())
        

        show_normalized_image(img1[0])
        plt.show()
        show_normalized_image(I_hat_t[0])
        plt.show()
        show_normalized_image(img2[0])
        plt.show()
        show_normalized_image(I_hat_tnext[0])
        plt.show()
        show_normalized_image(torch.square(I_hat_t[0]-img2[0]))
        plt.show()
        show_normalized_image(torch.square(I_hat_tnext[0]-img1[0]))
        plt.show()
        show_normalized_image(torch.square(img2[0]-img1[0]))
        
    #     print(Y1)
        break


def train_fine_tune(downstream_model, epochs, dataloader, criterion, optimizer):
    downstream_model.train()

    train_losses = []
    frame_list, mask_list = None, None
    j=0

    best_loss = float("inf")
#     best_model = {}
    best_model = downstream_model.state_dict()

    for _ in range(epochs):
        total_train_loss = 0.0
        total_train_correct = 0.0

        pbar = tqdm(dataloader, leave=False)

        for j, batch in enumerate(pbar):
          
            if j == 333:
                break
                
            frame_list, mask_list = batch[0], batch[1] # TODO
            total_train_loss = 0.0
            

            for i in range(len(frame_list) - 1):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)
                mask_list = mask_list.type(torch.LongTensor).to(device)


                logits = downstream_model(img1)
                loss = criterion(logits, mask_list[:,i])

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


def show_normalized_image(img, title=None):
    """Denormalize image and display"""
    
    plt.imshow(unnormalize(img).detach().cpu().permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis('off')



### MAIN ###

if __name__ == "__main__":

    unlabeled_data = UnlabeledDataset("/dataset/dataset/unlabeled")
    labeled_data = LabeledDataset("/dataset/dataset/train")
    val_data = LabeledDataset("/dataset/dataset/val")
    # hidden_data = UnlabeledDataset("/dataset/dataset/hidden", 11)

    train_dataloader = DataLoader(unlabeled_data, batch_size=3, shuffle=True)
    downstream_dataloader = DataLoader(labeled_data, batch_size=3, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
    # hidden_dataloader = DataLoader(hidden_data, batch_size=1, shuffle=True)


    print("Loaded data")

    # Train FJepa model

    model_name = "dummy"
    in_features = 3 
    lm, mu, nu, lambda_a, lambda_b = 0.02, 0.02, 0.01, 1, 1

    FJepa_model = FJepa(in_features).to(device)
#     PATH = "best_"+model_name+".pth"
#     FJepa_model.load_state_dict(torch.load(PATH))
    optimizer = optim.SGD(FJepa_model.parameters(), lr=1e-5, weight_decay=1e-4, foreach=True)

    train_FJepa(FJepa_model, 5, train_dataloader, FJepa_criterion, optimizer, model_name) # Training the MC JEPA


    ### Train downstream model

    # in_features_downstream = 16

    # downstream_model = UNet().to(device)
    # downstream_optimizer = optim.RMSprop(downstream_model.parameters(),
    #                           lr=1e-5, weight_decay=1e-8, momentum=0.999, foreach=True)

    # criterion = nn.CrossEntropyLoss()

    # train_fine_tune(downstream_model, 5, downstream_dataloader, criterion, downstream_optimizer)
