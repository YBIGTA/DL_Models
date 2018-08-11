import torch
from torch import nn
from torch.nn.init import xavier_normal
from torchvision import models


class FCN8s(nn.Module):
    def __init__(self, num_class):
        super(FCN8s,self).__init__()
        vgg = models.vgg16(pretrained=True)
        vgg_conv = list(vgg.children())[0]
        self.pool3 = None
        self.pool4 = None
        self.pool5 = None
        cnt = 0
        layers = []

        for layer in list(vgg_conv.children()):
            layers.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                cnt+=1
                if cnt == 3:
                    self.pool3 = nn.Sequential(*layers)
                    layers = []
                elif cnt == 4:
                    self.pool4 = nn.Sequential(*layers)
                    layers = []
                elif cnt == 5:
                    self.pool5 = nn.Sequential(*layers)
                else:
                    continue

        self.fc6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=4096,
                kernel_size=7,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Dropout2d()
        )


        self.fc7 = nn.Sequential(
            nn.Conv2d(
                in_channels=4096,
                out_channels=4096,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(
                in_channels=4096,
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.score_pool3 = nn.Conv2d(
                in_channels=256,
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.score_pool4 = nn.Conv2d(
                in_channels=512,
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.score_pool5 = nn.Conv2d(
                in_channels=512,
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )

        self.upscore_fr = nn.ConvTranspose2d(
                in_channels=num_class,
                out_channels=num_class,
                kernel_size=7,
                stride=2,
                bias=False
        )


        self.upscore_pool5 = nn.ConvTranspose2d(
            in_channels=num_class,
            out_channels=num_class,
            kernel_size=2,
            stride=2,
            bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            in_channels=num_class,
            out_channels=num_class,
            kernel_size=2,
            stride=2,
            bias=False
        )

        self.upscore_pool3 = nn.ConvTranspose2d(
            in_channels=num_class,
            out_channels=num_class,
            kernel_size=8,
            stride=8,
            bias=False
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                xavier_normal(m.weight.data)

    def forward(self, x):
        pool3 = self.pool3(x) # now 1/8 (?, 256, 28, 28)
        pool4 = self.pool4(pool3) # now 1/16 (?, 512, 14, 14)
        pool5 = self.pool5(pool4) # now 1/32 ( ?, 512, 7, 7 )

        x = self.fc6(pool5) # ( ?, 4096, 1, 1 )
        x = self.fc7(x) # ( ?, 4096, 1, 1 )

        score = self.score_fr(x) # (?, num_class)
        score_pool5 = self.score_pool5(pool5)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore_fr = self.upscore_fr(score) # now 1/32 ( ?, num_class, 7, 7 )
        upscore_pool5 = self.upscore_pool5(score_pool5+upscore_fr) # now 1/16 ( ?, num_class, 14, 14 )
        upscore_pool4 = self.upscore_pool4(score_pool4+upscore_pool5) # now 1/8 ( ?, num_class, 28, 28 )
        upscore_final = self.upscore_pool3(score_pool3+upscore_pool4)  # now  ( ?, num_class, 224, 224 )

        return upscore_final
