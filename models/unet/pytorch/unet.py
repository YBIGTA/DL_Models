import torch 
from torch import nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvLayer,self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_features,out_features,kernel_size=3, stride=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_features,out_features,kernel_size=3, stride=1))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self,input):
        output = self.layers(input)
        return output

class Unet(nn.Module):
    def __init__(self, config):
        super(Unet, self).__init__()
        ndim = config['hidden_dim']
        conv_layers = [ConvLayer(config['input_dim'],ndim)]
        successive_layers = []
        upconv_layers = []

        for _ in range(4):
            conv_layers.append(ConvLayer(ndim,ndim*2))
            upconv_layers.append(nn.ConvTranspose2d(ndim*2, ndim, kernel_size=2, stride=2))
            successive_layers.append((ConvLayer(ndim*2,ndim)))
            ndim *= 2

        self.conv_layers = nn.ModuleList(conv_layers)
        self.upconv_layers = nn.ModuleList(upconv_layers)
        self.successive_layers = nn.ModuleList(successive_layers)
        self.conv1x1 = nn.Conv2d(config['hidden_dim'],config['num_class'], kernel_size=1, stride=1)

    def forward(self, x):
        features = []
        input_size = x.size()

        for i in range(4):
            x = self.conv_layers[i](x)
            features.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv_layers[4](x)
        for i in range(3,-1,-1):
            x = self.upconv_layers[i](x)
            x = torch.cat([x, F.upsample(features[i], size=x.size()[2:],mode='bilinear')],dim=1 )
            x = self.successive_layers[i](x)
        x = self.conv1x1(x)
        out = F.upsample(x, size = input_size[2:], mode='bilinear')
        return out