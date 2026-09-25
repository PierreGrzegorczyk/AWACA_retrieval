from torch import nn
import torch
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, padding_size, batch_normalization=True):
        super().__init__()
        self.batch_normalization = batch_normalization
        self.conv1 = nn.Conv1d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = padding_size, padding_mode='replicate')
        
        self.bn1 = nn.BatchNorm1d(nb_channels)

        self.conv2 = nn.Conv1d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = padding_size, padding_mode='replicate')

        self.bn2 = nn.BatchNorm1d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization:
            y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization:
            y = self.bn2(y)
        y = y + x

        return y    
    
    
    
class Direct_w_Resblock(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__()
#         self.encoder = nn.Sequential(
        self.conv1 = nn.Conv1d(2,nb_channels,kernel_size=7)
#             nn.ReLU()
            
        self.resblock1 = ResBlock(nb_channels,kernel_size=7, padding_size=3)
        self.avgpool1 = nn.AvgPool1d(kernel_size=3, stride = 3)
#             nn.ReLU(),
            
        self.resblock2 = ResBlock(nb_channels,kernel_size=7, padding_size=3)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3, stride = 3)
#             nn.ReLU(),
            
        self.resblock3 = ResBlock(nb_channels,kernel_size=7, padding_size=3)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride = 2)
#             nn.ReLU(),
            
        self.resblock4 = ResBlock(nb_channels,kernel_size=7, padding_size=3)
#         nn.ReLU(),
            
        self.conv2 = nn.Conv1d(nb_channels, 1, kernel_size=3, padding = 1, padding_mode='replicate')
        
    def encode(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        print('conv1', x.shape)
        x = self.resblock1(x)
        print('resblock1', x.shape)
        x = self.avgpool1(x)
        print('avgpool1', x.shape)
        x = F.relu(x)
        x = self.resblock2(x)
        print('resblock2', x.shape)
        x = self.avgpool2(x)
        print('avgpool2', x.shape)
        x = F.relu(x)
        x = self.resblock3(x)
        print('resblock3', x.shape)
        x = self.avgpool3(x)
        print('avgpool3', x.shape)
        x = F.relu(x)
        x = self.resblock4(x)
        print('resblock4', x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        print('conv2', x.shape)
        return x
    
    
    
class Direct_w_Resblock_seq(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2,nb_channels,kernel_size=7),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.AvgPool1d(kernel_size=3, stride = 3),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.AvgPool1d(kernel_size=3, stride = 3),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.AvgPool1d(kernel_size=2, stride = 2),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.ReLU(),
            
            nn.Conv1d(nb_channels, 1, kernel_size=3, padding = 1, padding_mode='replicate')
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class Direct_w_Resblock_seq1(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2,nb_channels,kernel_size=7),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.AvgPool1d(kernel_size=3, stride = 3),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.AvgPool1d(kernel_size=3, stride = 3),
            nn.ReLU(),
            
            nn.Conv1d(nb_channels,nb_channels,kernel_size=7, padding_size=3),
            nn.AvgPool1d(kernel_size=2, stride = 2),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=7, padding_size=3),
            nn.ReLU(),
            
            nn.Conv1d(nb_channels, 1, kernel_size=3, padding = 1, padding_mode='replicate')
        )
        
    def forward(self, x):
        x = self.model(x)
        return x