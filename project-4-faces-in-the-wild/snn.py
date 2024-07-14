import torch
import torch.nn as nn

class SingleUnitNetwork(nn.Module):
    def __init__(self):
        super(SingleUnitNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  

        # Ajustar la dimensión de entrada de la capa lineal
        self.fclayer1 = nn.Sequential(
            nn.Linear(256 * 10 * 10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # Aplanar para la capa lineal
        
        x = self.fclayer1(x)
        x = self.fc2(x)
        return x

class SiameseNeuralNetwork(nn.Module):
    '''
    Arquitectura de una red neuronal siamesa que 
    detecta si pares de imágenes son del mismo rostro
    '''
    def __init__(self):
        super(SiameseNeuralNetwork, self).__init__()
        self.unitNetwork= SingleUnitNetwork()

        # Ajustar el número de capas lineales
        self.layer1 = torch.nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        self.layer2 = torch.nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4))

        self.layer3 = nn.Linear(256, 1)
        self.layer5 = torch.nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4))
        self.layer4 = nn.Linear(256, 1)

    def forward(self, img1, img2):
        features1 = self.unitNetwork(img1)  # Feature Vector First twin
        features2 = self.unitNetwork(img2)  # Feature Vector Second twin
        dist = torch.abs(features1 - features2)

        out = self.layer1(dist)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
