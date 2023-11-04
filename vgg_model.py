import torch
from torch import nn

# basic model setupt & load

vgg19_bn = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=False)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = vgg19_bn
        num_ftrs = 25088
        self.vgg.classifier = nn.Sequential(
                                    nn.Linear(num_ftrs, 4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Linear(4096, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Linear(512, 2))
        
    def forward(self, x):
        return self.vgg(x)

model = VGG()
model = model.to("cpu")

