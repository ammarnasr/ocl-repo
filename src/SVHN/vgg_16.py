import torch.nn as nn


### VGG Block of 2 conv layers
def VGG_Block(ni, nf, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size, stride, padding=1,groups=1),
        nn.BatchNorm2d(nf),
        nn.ReLU(),
        #nn.LeakyReLU()
       
        nn.Conv2d(nf, nf, kernel_size, stride, padding=1,groups=1),
        nn.BatchNorm2d(nf),
        nn.ReLU(),
     
        nn.MaxPool2d(2,2)
    )
 
def Pconv_layer(ni, nf, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size, stride, padding=0,groups=1),
        #nn.BatchNorm1d(nf),
        #nn.LeakyReLU()
        #nn.ReLU()
    ) 


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = VGG_Block(3, 64)
        self.layer2 = VGG_Block(64, 128)
        self.layer3 = VGG_Block(128, 512)
        self.layer4 = Pconv_layer(512,255)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.drop = nn.Dropout(0.25)

        #self.fc = nn.Linear(512, 255)

        #self.fc_BN = nn.BatchNorm1d(255)
        # self.fc_activation=nn.Sigmoid()
        #self.fc_activation = nn.ReLU()

        self.fc_k = nn.Linear(255, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        # out = self.drop(out)
        #out = self.fc(out)
        #out = self.fc_BN(out)
        #out = self.fc_activation(out)
        #out = (0.1 * out) -2
        out_k = out
        out = self.fc_k(out)
        return out_k, out  # torch.softmax(out, dim=-1)
