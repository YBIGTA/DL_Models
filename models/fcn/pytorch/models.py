import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torchvision import models
import torch.nn.functional as F
import numpy as np

class FCN8s_voc(nn.Module):
    def __init__(self, config):
        super(FCN8s_voc, self).__init__()
        """
        Referenced the implementation below
        https://github.com/zijundeng/pytorch-semantic-segmentation/
        """
        pretrained = True if config.mode == 'finetuning' else False
        self.num_class = config.num_class

        self.feature3, self.feature4, self.feature5, self.classifier = self._parse_vgg()
        self.score_pool3 = nn.Conv2d(256, self.num_class, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_class, kernel_size=1)
        self.score_fr = nn.Sequential(
             nn.Conv2d(512, 4096, kernel_size=7),
             nn.ReLU(inplace=True),
             nn.Dropout(),
             nn.Conv2d(4096, 4096, kernel_size=1),
             nn.ReLU(inplace=True),
             nn.Dropout(),
             nn.Conv2d(4096, self.num_class, kernel_size=1)
        )

        self.upscore2 = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=4, stride=2, bias=False)
        #self.upscore8 = nn.UpsamplingBilinear2d(scale_factor = 8)
        self._init_weights()

    def _parse_vgg(self):
        vgg = models.vgg16(pretrained=True)
        cnt = 0
        layers = []

        for layer in list(vgg.features):
            layers.append(layer)
            if isinstance(layer, nn.ReLU):
                layers[-1].inplace = True
            if isinstance(layer, nn.MaxPool2d):
                layers[-1].ceil_mode = True
                cnt+=1
                if cnt == 3:
                    layers[0].padding = (100,100)
                    pool3 = nn.Sequential(*layers)
                    layers = []
                elif cnt == 4:
                    pool4 = nn.Sequential(*layers)
                    layers = []
                elif cnt == 5:
                    pool5 = nn.Sequential(*layers)
                else:
                    continue
        return pool3, pool4, pool5, vgg.classifier

    def _get_upsampling_weight(self,in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        center = factor - 1 if  kernel_size % 2 == 1 else factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
        weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
        return torch.from_numpy(weight).float()

    def _init_weights(self):
        # score_fr: inject weights from classifier
        fc6_weight = self.classifier[0].weight.data.view(4096,512,7,7)
        fc6_bias = self.classifier[0].bias
        fc7_weight = self.classifier[3].weight.data.view(4096,4096,1,1)
        fc7_bias = self.classifier[3].bias
        self.score_fr[0].weight.data.copy_(fc6_weight)
        self.score_fr[0].bias.data.copy_(fc6_bias)
        self.score_fr[3].weight.data.copy_(fc7_weight)
        self.score_fr[3].bias.data.copy_(fc7_bias)
        kaiming_normal_(self.score_fr[6].weight.data, mode='fan_out', nonlinearity='relu')
        self.score_fr[6].bias.data.fill_(0)
        self.classifier = None

        self.upscore2.weight.data.copy_(self._get_upsampling_weight(self.num_class, self.num_class, 4))
        self.upscore_pool4.weight.data.copy_(self._get_upsampling_weight(self.num_class, self.num_class, 4))


    def forward(self, x):
        input_size = x.size()
        pool3 = self.feature3(x)
        pool4 = self.feature4(pool3)
        pool5 = self.feature5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        score_pool3 = self.score_pool3(0.0001 * pool3)

        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]
                                          +upscore2)
        fuse = score_pool3[:, :, 9:9+upscore_pool4.size()[2], 9:9+upscore_pool4.size()[3]] + upscore_pool4
        out = F.interpolate(upscore_pool4,scale_factor = 8, mode='bilinear', align_corners=False)[:, :, 31:31+input_size[2], 31:31+input_size[3]].contiguous()#upscore8[:, :, 31:31+input_size[2], 31:31+input_size[3]].contiguous()
        return out


class FCN8s_224(nn.Module):
    def __init__(self,config):
        super(FCN8s_224,self).__init__()
        self.num_class = config.num_class
        self.mode = config.mode
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
                out_channels=self.num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.score_pool3 = nn.Conv2d(
                in_channels=256,
                out_channels=self.num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.score_pool4 = nn.Conv2d(
                in_channels=512,
                out_channels=self.num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.score_pool5 = nn.Conv2d(
                in_channels=512,
                out_channels=self.num_class,
                kernel_size=1,
                stride=1,
                padding=0
        )

        self.upscore_fr = nn.ConvTranspose2d(
                in_channels=self.num_class,
                out_channels=self.num_class,
                kernel_size=7,
                stride=2,
                bias=False
        )


        self.upscore_pool5 = nn.ConvTranspose2d(
            in_channels=self.num_class,
            out_channels=self.num_class,
            kernel_size=2,
            stride=2,
            bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            in_channels=self.num_class,
            out_channels=self.num_class,
            kernel_size=2,
            stride=2,
            bias=False
        )

        self.upscore_pool3 = nn.ConvTranspose2d(
            in_channels=self.num_class,
            out_channels=self.num_class,
            kernel_size=8,
            stride=8,
            bias=False
        )

        self._init_weights()

    def _init_weights(self):
        if self.mode == 'finetuning':
            self.pool3.requires_grad = False
            self.pool4.requires_grad = False
            self.pool5.requires_grad = False

        for param in self.parameters():
            if not param.requires_grad:
                continue
            elif isinstance(param, nn.Conv2d):
                xavier_normal(param.weight.data)
                if param.bias is not None:
                    param.bias.data.fill_(0)
            elif isinstance(param, nn.Conv2d):
                xavier_normal(param.weight.data)

        self.pool3.requires_grad = True
        self.pool4.requires_grad = True
        self.pool5.requires_grad = True

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
