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
import scipy.ndimage as nd

from FJEPA import FJepa,warp
from downstream import UNet
from train import UnlabeledDataset, LabeledDataset

import torchmetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

def test(downstream_model, JepaModel, epochs, dataloader, criterion):
    train_losses = []

    best_loss = float("inf")
    best_model = downstream_model.state_dict()

    for _ in range(epochs):
        total_train_loss = 0.0

        pbar = tqdm(dataloader, leave=False)
        avg_jacc = 0.0
        for j,batch in enumerate(pbar):
          
            frame_list, mask_list = batch[0], batch[1]
            total_train_loss = 0.0
            mask_pred = None
            

            for i in range(len(frame_list) - 1):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)
                mask_list = mask_list.type(torch.LongTensor).to(device)

                mask_pred = downstream_model(img1)
#               
                # if i == 20:
                #     print(mask_pred.shape, mask_list[:,i].shape)
                #     print(mask_list[:,i][0])

                #     print(jaccard(mask_pred, mask_list[:,i]))
                #     print((torch.argmax(mask_pred[0], dim=0) == 0).all())
                #     plt.imshow(mask_list[0][i].cpu())
                #     plt.show()
                #     plt.imshow(mask_pred.argmax(dim=1)[0].float().cpu())
                #     plt.show()
                loss = criterion(mask_pred, mask_list[:,i])
                total_train_loss += loss.item()

            avg_jacc += jaccard(mask_pred, mask_list[:,21])
            pbar.set_postfix({'Per frame Loss': total_train_loss/(len(frame_list)-1), 'avg_jaccard': avg_jacc.item() / ((j+1))})

def real_test(downstream_model, JepaModel, epochs, dataloader, criterion, scale = 0.1, r = 1):
    
    downstream_model.eval()
    JepaModel.eval()
    
    train_losses = []

    best_loss = float("inf")
    best_model = downstream_model.state_dict()


    final_submit = None
    mask_tensor = None
    for _ in range(epochs):
        total_train_loss = 0.0

        pbar = tqdm(dataloader, leave=False)
        
        avg_jacc = 0.0
        

        for j,batch in enumerate(pbar):
          
            frame_list, mask_list = batch[0], batch[1] # TODO
            total_train_loss = 0.0
            X_tconcat = None
            I_hat_t = None
            I_hat_tnext = None
            I_hat_t = None
            img11 = None  
            img1 = None
            
            final_flow = 0.0
            for i in range(10):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)
                if i == 9:
                  img11 = img2
                mask_list = mask_list.type(torch.LongTensor).to(device)


                X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext, upflow1 = JepaModel(img1, img2)
                final_flow = upflow1*1.25 + scale*final_flow

                
            
            mask_pred = downstream_model(frame_list[10].to(device))
            mask_pred_ = downstream_model(frame_list[10].to(device))
            
            # img1 = None
            # img2 = None
            # for i in range(10):
            #     if i == 0:
            #       img1 = img11
                
            #     img2 = warp(img1, upflow1*1.25)
            #     X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
            #         f_tnext_t, I_hat_t, I_hat_tnext, upflow1 = JepaModel(img1, img2)
                
            #     img1 = img2

                # print(f_t_tnext[0].shape, mask_pred.shape)
                # flow = upflow1*1.25
            #     mask_pred = warp(mask_pred, upflow1*1.25)
            #     flow_np = flow.cpu().detach().numpy()
            #     smoothed = torch.tensor(nd.gaussian_filter(flow_np, sigma=5,radius=(0,0,r,r))).to(device)
            #     mask_pred = warp(mask_pred, smoothed)
            # mask_pred = downstream_model(img1)
            
            
            # final_flow *= 11
            final_flow_np = final_flow.cpu().detach().numpy()
            # smoothed = torch.tensor(nd.maximum_filter(final_flow_np, size=(1,1,r,r))).to(device)
            final_flow = torch.tensor(nd.gaussian_filter(final_flow_np, sigma=5,radius=(0,0,r,r))).to(device)
            mask_pred = warp(mask_pred, final_flow)
            img1 = warp(img1, final_flow)


            # print(smoothed)

            # print(final_flow_np)

            # print(smoothed == final_flow)

            # plt.imshow(unnormalize(img11[0]).detach().cpu().permute(1, 2, 0).numpy())
            # plt.title("11th Frame")
            # plt.show()
            # plt.imshow(unnormalize(frame_list[21][0]).detach().cpu().permute(1, 2, 0).numpy())
            # plt.title("22nd Frame")
            # plt.show()
            # plt.imshow(unnormalize(img1[0]).detach().cpu().permute(1, 2, 0).numpy())
            # plt.title("Predicted 22nd Frame")
            # plt.show()
            

            # print(f_t_tnext)
            # print(jaccard(mask_pred, mask_list[:,21]))
            # print((torch.argmax(mask_pred[0], dim=0) == torch.argmax(mask_pred_[0].cpu())).all())
            # plt.imshow(torch.argmax(mask_pred_[0].cpu(), dim=0))
            # plt.title("Predicted 11th Mask")
            # plt.show()
            # plt.imshow(mask_list[0][21].cpu())
            # plt.title("22nd Mask")
            # plt.show()
            # plt.imshow(torch.argmax(mask_pred[0].cpu(), dim=0))
            # plt.title("Predicted 22nd Mask")
            # plt.show()


            # Calculate True Jaccard
            mask_pred = torch.argmax(mask_pred[0], dim=0)
            mask_pred = mask_pred.unsqueeze(0)
            
            if final_submit is None:
              final_submit = mask_pred
              mask_tensor = mask_list[:,21]
            else:
              final_submit = torch.cat([final_submit, mask_pred], dim=0)
              mask_tensor = torch.cat([mask_tensor, mask_list[:,21]], dim=0)

            pbar.set_postfix({'Video': j+1}) 
    print(jaccard(final_submit.cpu(), mask_tensor.cpu()))


def save_tensor(downstream_model, JepaModel, epochs, dataloader, scale = 0.1, r = 1):
    """Obtain mask predictions for last frame and save predictions"""
    
    train_losses = []

    best_loss = float("inf")
    best_model = downstream_model.state_dict()
    final_submit = None

    for _ in range(epochs):
        total_train_loss = 0.0

        pbar = tqdm(dataloader, leave=False)
        
        avg_jacc = 0.0


        for j,batch in enumerate(pbar):
          
            frame_list = batch # TODO
            total_train_loss = 0.0
            X_tconcat = None
            I_hat_t = None
            I_hat_tnext = None
            I_hat_t = None
            img11 = None
            
            #### 1
            final_flow = 0.0
            for i in range(10):
                img1 = frame_list[i].to(device)
                img2 = frame_list[i+1].to(device)

                X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
                f_tnext_t, I_hat_t, I_hat_tnext, upflow1 = JepaModel(img1, img2)
                final_flow = upflow1*1.25 + scale*final_flow

                
            
            mask_pred = downstream_model(frame_list[10].to(device))
            final_flow_np = final_flow.cpu().detach().numpy()
            final_flow = torch.tensor(nd.gaussian_filter(final_flow_np, sigma=5,radius=(0,0,r,r))).to(device)
            mask_pred = warp(mask_pred, final_flow)


            #### 2
            # for i in range(10):
            #     img1 = frame_list[i].to(device)
            #     img2 = frame_list[i+1].to(device)
            #     if i == 9:
            #       img11 = img2
            #     X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
            #     f_tnext_t, I_hat_t, I_hat_tnext, upflow1 = JepaModel(img1, img2)


            # mask_pred = downstream_model(frame_list[10].to(device))
            # img1 = None
            # img2 = None
            # for i in range(10):
            #     if i == 0:
            #       img1 = img11
                
            #     img2 = warp(img1, upflow1*1.25)
            #     X_t, X_tnext, X_hat_t, X_hat_tnext, f_t_tnext,\
            #         f_tnext_t, I_hat_t, I_hat_tnext, upflow1 = JepaModel(img1, img2)
                
            #     img1 = img2

            #     # print(f_t_tnext[0].shape, mask_pred.shape)
            #     # flow = upflow1*1.25
            #     mask_pred = warp(mask_pred, upflow1*1.25)


            #### Common
            
            mask_pred = torch.argmax(mask_pred[0], dim=0)
            mask_pred = mask_pred.unsqueeze(0)
            
            if final_submit is None:
              final_submit = mask_pred
            else:
              final_submit = torch.cat([final_submit, mask_pred], dim=0)
            
            pbar.set_postfix({'Video': j+1}) 

    print(final_submit.shape)
    torch.save(final_submit, "/content/drive/My Drive/Colab Notebooks/Spring23/DL/Project/"+model_name+"_submission.pt")




### MAIN ###

## Loading data ##

unlabeled_data = UnlabeledDataset("/content/Dataset_Student/unlabeled")
labeled_data = LabeledDataset("/content/Dataset_Student/train")
val_data = LabeledDataset("/content/Dataset_Student/val")
# hidden_data = UnlabeledDataset("/content/drive/My Drive/Colab Notebooks/Spring23/DL/Project/hidden", 11)

train_dataloader = DataLoader(unlabeled_data, batch_size=3, shuffle=True)
downstream_dataloader = DataLoader(labeled_data, batch_size=3, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
# hidden_dataloader = DataLoader(hidden_data, batch_size=1, shuffle=True)


# show_normalized_image(unlabeled_data[10][0])

## Loading best model weights #

in_features = 3
model_name = "model_Context_10ep_2500vid2"
PATH = "/content/drive/My Drive/Colab Notebooks/Spring23/DL/Project/best_"+model_name+".pth"
FJepa_model = FJepa(in_features).to(device)
FJepa_model.load_state_dict(torch.load(PATH))
FJepa_model.eval()

PATH = "/content/drive/My Drive/Colab Notebooks/Spring23/DL/Project/best_downstream_model.pth"
downstream_model = UNet().to(device)
downstream_model.load_state_dict(torch.load(PATH))
downstream_model.eval()


## Evaluate on validation set ##

criterion = nn.CrossEntropyLoss()
# test(downstream_model, FJepa_model, 1, val_dataloader, criterion)
real_test(downstream_model, FJepa_model, 1, val_dataloader, criterion, scale=0.75, r=3)


## Save mask predictions ##

hidden_data = UnlabeledDataset("/content/hidden", 11)
hidden_dataloader = DataLoader(hidden_data, batch_size=1, shuffle=False)

save_tensor(downstream_model, FJepa_model, 1, hidden_dataloader, scale=0.75, r=3)

# torch.load("/content/drive/My Drive/Colab Notebooks/Spring23/DL/Project/"+model_name+"_submission.pt").shape
