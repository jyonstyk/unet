import torch.nn as nn
import torch

def conv_twice(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = conv_twice(1,32)
        self.conv2 = conv_twice(32,32)
        self.conv3 = conv_twice(32,32)
        self.conv4 = conv_twice(32,32)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.deconv3 = conv_twice(32+32,32)
        self.deconv2 = conv_twice(32+32,32)
        self.deconv1 = conv_twice(32+32,32)
        self.conv_last = nn.Conv2d(32,2,1)
    
    def forward(self,x):
        x_conv1 = self.conv1(x)
        x = self.maxpool(x_conv1)
        
        x_conv2 = self.conv2(x)
        x = self.maxpool(x_conv2)
        
        x_conv3 = self.conv3(x)
        x = self.maxpool(x_conv3)
        
        x = self.conv4(x)
        
        x = self.upsample(x)
        x = torch.cat([x,x_conv3],dim=1)
        x = self.deconv3(x)
        
        x = self.upsample(x)
        x = torch.cat([x,x_conv2],dim=1)
        x = self.deconv2(x)
        
        x = self.upsample(x)
        x = torch.cat([x,x_conv1],dim=1)
        x = self.deconv1(x)
        
        return self.conv_last(x)