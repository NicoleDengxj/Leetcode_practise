# part 1: import the library
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

"""
torch: The main PyTorch library for tensor operations.
torch.nn: Contains modules for building neural networks (e.g., nn.Conv2d, nn.Linear).
torch.nn.functional: Contains functions for operations like activations, pooling, etc., which can be used directly without defining layers.
"""


# Part 2: Built the model
"""
class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()

-- in pytorch, custom models inherit from nn.Module, the base class for all neural networks.
-- __init__: Constructor to define the layers of the model.
-- num_classes=1000: Sets the number of output classes (default is 1000 for ImageNet).
-- super(AlexNet, self).__init__(): Initializes the nn.Module parent class.

"""
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        
        self.ReLu = nn.ReLU()
        
        self.c1 = nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 11, stride = 4)
        self.s2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.c3 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
        self.s4 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.c5 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        self.c6 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        self.c7 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        
        self.s8 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.flatten = nn.Flatten()        
        self.fc1 = nn.Linear(in_features = 256 * 6 * 6, out_features = 4096)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.fc3 = nn.Linear(in_features = 4096, out_features = num_classes)  #out_features is the number of classes
        
    def forward(self, x):
        x = self.ReLu(self.c1(x))
        x = self.s2(x)
        x = self.ReLu(self.c3(x))
        x = self.s4(x)
        x = self.ReLu(self.c5(x))
        x = self.ReLu(self.c6(x))
        x = self.ReLu(self.c7(x))
        
        x = self.s8(x)
        x = self.flatten(x)
        
        x = self.ReLu(self.fc1(x))        
        x = F.dropout(x, 0.5)
        
        x = self.ReLu(self.fc2(x))
        x = F.dropout(x, 0.5)
        
        x = self.fc3(x)
        return x
    



class AlexNet_generate_chatgpt(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_generate_chatgpt, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Instantiate the model
model_2 = AlexNet_generate_chatgpt(num_classes=1000)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))
