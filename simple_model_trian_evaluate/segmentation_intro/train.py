"""
https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-efficientnetb7/notebook
"""

import os
import time
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import albumentations as A
from scipy.ndimage.morphology import binary_dilation
import segmentation_models_pytorch as smp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from utils import training_loop
from model import Models


def main(model, loss_fn, optimizer, lr_scheduler, train_loader, valid_loader, epochs):
    result_dir = "./result/"
    os.makedirs(result_dir, exist_ok=True)  
    save_path = os.path.join(result_dir, "best_model.pth")
    
    model_factory = Models()
    model = model_factory.Unet()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  
        
    history = training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler,save_path) 
    return history

    
    
# if __name__ == '__main__':
#     loss_fn = BCE_dice
#     optimizer = Adam(model.parameters(), lr=0.001)
#     epochs = 5
#     lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2,factor=0.2)
  
    
 
#     history = main(model, loss_fn, optimizer, epochs,lr_scheduler, train_loader, valid_loader)  
    
#     print_IOU_DICE(epochs, history)
#     print_train_val_loss(epochs,history)