from torch import nn
import torch as t
from torch.nn import functional as F

class VGGBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(VGGBlock,self).__init__()
        self.block=nn.Sequential(nn.Conv2d(inchannel,outchannel,3,1,1,bias=False),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU())
    def forward(self,x):
        return self.block(x)
    
class Vgg16(nn.Module):
    def __init__(self,nums_class=100):
        super(Vgg16,self).__init__()
        
        self.block1=self.make_block(3,64,2)
        self.block2=self.make_block(64,128,2)
        self.block3=self.make_block(128,256,3)
        self.block4=self.make_block(256,512,3)
        self.block5=self.make_block(512,512,3)
        
        self.fc1=nn.Linear(512*7*7,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,nums_class)
        
    def make_block(self,inchannel,outchannel,num_block):
        blocks=[]
        blocks.append(VGGBlock(inchannel,outchannel))
        for i in range(1,num_block):
            blocks.append(VGGBlock(outchannel,outchannel))

        return nn.Sequential(*blocks)
    
    def forward(self,x):
        x=F.max_pool2d(self.block1(x),(2,2))
        x=F.max_pool2d(self.block2(x),(2,2))
        x=F.max_pool2d(self.block3(x),(2,2))
        x=F.max_pool2d(self.block4(x),(2,2))
        x=F.max_pool2d(self.block5(x),(2,2))
        
        x=x.view(x.size(0),-1)
        x=self.fc1(F.dropout(F.relu(x),p=0.5))
        x=self.fc2(F.dropout(F.relu(x),p=0.5))
        x=self.fc3(x)
        
        return x

