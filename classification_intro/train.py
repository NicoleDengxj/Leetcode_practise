import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import torchvision
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time
import copy
import os
from torchsummary import summary

from model import AlexNet_generate_chatgpt,AlexNet
from utils import train_val_data_process, show_samples,train_model_process, matplot_acc_loss


def main():
    # check enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if  torch.cuda.is_available():
        print("CUDA Device Name: ", torch.cuda.get_device_name(0))  # Name of the GPU
    else:
        print("The training is based on CPU")

    ##check the path 
    result_dir = "./result/"
    os.makedirs(result_dir, exist_ok=True)

    # prepare train/val/test data
    train_dataloader,val_dataloader,train_data,val_data,num_classes= train_val_data_process()

    sample_image, _ = train_data[0]
    input_size = sample_image.shape
    print(f"Detected input size from dataset: {input_size}")
    
    # show samples from data set
    print("Sample images from the training set:")
    show_samples(train_data) 

    # prepare model
    model = AlexNet(num_classes).to(device)
    print(summary(model, input_size))

    train_process = train_model_process(model,train_dataloader,val_dataloader,num_epochs=1)
    matplot_acc_loss(train_process)



if __name__ == '__main__':
    main()








    
