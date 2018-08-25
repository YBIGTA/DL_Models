import torch
from torch import nn
from torchvision import models

class SSD300(nn.Module):
    def __init__(self,num_classes):
        super(SSD300,self).__init__()
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        for layer in features:
            if isinstance(layer,nn.MaxPool2d):
                layer.ceil_mode = True
            elif isinstance(layer,nn.ReLU):
                layer.inplace = True

        self.conv4_3 = nn.Sequential(*features[:23])
        self.conv5_3 = nn.Sequential(*features[23:30])
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride = 1, padding=1)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=1, stride = 1),
            nn.ReLU(inplace=True)
        )
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(1024,256,kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3, stride= 2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_1 = nn.Sequential(
            nn.Conv2d(512,128,kernel_size= 1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_2 = nn.Conv2d(128,256,kernel_size=3,stride=1)

    def forward(self, input):
        conv4_3 = self.conv4_3(input)
        conv5_3 = self.conv5_3(conv4_3)
        pool5 = self.pool5(conv5_3)
        conv6 = self.conv6(pool5)
        conv7 = self.conv7(conv6)
        conv8_2 = self.conv8_2(self.conv8_1(conv7))
        conv9_2 = self.conv9_2(self.conv9_1(conv7))
        conv10_2 = self.conv10_2(self.conv10_1(conv7))
        conv11_2 = self.conv11_2(self.conv11_1(conv7))

