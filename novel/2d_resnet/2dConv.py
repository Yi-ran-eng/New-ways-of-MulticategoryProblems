import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type,Literal
from torchvision import transforms
from GOGSdataset import DogSet,Manipulating,testprepare
from AdamReg import createAdam
from prop2d import Propagation
def regLayers(cls:Type,addParameters=True):
    rawinit=cls.__init__
    if addParameters:
        def new_init(self,*args,**kwargs):
            rawinit(self,*args,**kwargs)
            self.Layer_cache={}
    cls.__init__=new_init
    return cls
@regLayers
class NetCNN(nn.Module):
    def __init__(self,num_classes=10,useAdam=False):
        super().__init__()
        self.conv1=createAdam(nn.Conv2d,in_channels=3,out_channels=8,kernel_size=3,padding=1,adam=useAdam)
        self.conv1.opt=None
        #if you don't want the size to change,use formular:p=(k-1)/2 to calculate the value p
        self.conv2=createAdam(nn.Conv2d,in_channels=8,out_channels=16,kernel_size=3,padding=1,adam=useAdam)
        self.conv2.opt=None
        self.conv3=createAdam(nn.Conv2d,16,32,kernel_size=3,padding=1,adam=useAdam)
        self.conv3.opt=None
        self.conv1.manual_back=0
        self.conv1.name='conv1'
        self.conv2.manual_back=0
        self.conv2.name='conv2'
        self.conv3.manual_back=0
        self.conv3.name='conv3'
        #pooling layer
        self.pool1=nn.AdaptiveMaxPool2d((12,12))#whatever how big the input picture is,the size will be compressed to 4x4
        self.pool2=nn.AdaptiveMaxPool2d((6,6))
        self.pool3=nn.AdaptiveMaxPool2d((4,4))
        self.pool1.manual_back=0
        self.pool1.name='pool1'
        self.pool2.manual_back=0
        self.pool2.name='pool2'
        self.pool3.manual_back=0
        self.pool3.name='pool3'
        #linear layers
        self.fc1=createAdam(nn.Linear,32*4*4,128,adam=useAdam)#one picture has three types of size:channels,width,height,we stretch it into a 1d vector
        #and then the size turns into channels x 4 x 4
        self.fc1.manual_back=0
        self.fc1.name='linear1'
        self.outputlayer=createAdam(nn.Linear,128,num_classes,adam=useAdam)
        self.outputlayer.manual_back=0
        self.outputlayer.name='linear2'
        # self.dropout=nn.Dropout(0.5)
    def apply_opt(self,optimizer):
        for layer in self.modules():
            if isinstance(layer,(nn.AdaptiveMaxPool2d,nn.Conv2d,nn.Linear)):
                layer.opt=optimizer
    def forward(self,x,epoch):
        if epoch >= 1:
            # print('epoc')
            dic=getattr(self,'Layer_cache')
            for name,param in dic.items():
                layer=param[-1]
                if isinstance(layer,nn.Linear) and len(x.shape) == 4:
                    x=x.view(x.size(0),-1)
                dic[layer.name][1]=x
                x=layer(x)
                dic[layer.name][0]=x
            return x
        # print(x.shape,'inputshape')
        x1=F.relu(self.conv1(x))
        self.filling(x,x1,self.conv1)
        x2=self.pool1(x1)
        self.filling(x1,x2,self.pool1)
        x3=F.relu(self.conv2(x2))
        self.filling(x2,x3,self.conv2)
        x4=self.pool2(x3)
        self.filling(x3,x4,self.pool2)
        x5=F.relu(self.conv3(x4))
        self.filling(x4,x5,self.conv3)
        x6=self.pool3(x5)
        self.filling(x5,x6,self.pool3)
        x_flatten=x6.view(x6.size(0),-1)
        xf1=F.relu(self.fc1(x_flatten))
        self.filling(x_flatten,xf1,self.fc1)
        # x=self.dropout(x)
        out=self.outputlayer(xf1)
        self.filling(xf1,out,self.outputlayer)
        return out
    def filling(self,input,output,layer):
        dic=getattr(self,'Layer_cache')
        if not input.requires_grad:
            input=input.detach().requires_grad_(True)
        dic[layer.name]=[output,input,layer]

def train(model,datasets,epoches,lr,lt):
    opt=Propagation(model)
    lossfn=nn.CrossEntropyLoss()
    losses=[]
    for epoch in range(epoches):
        # print(f'epoch:{epoch}......................')
        epochloss=[]
        for features,label in datasets:
            yhat=model(features,epoch)
            # print(yhat.shape,'yhat')
            loss=lossfn(yhat,label.squeeze())
            epochloss.append(loss)
            opt.runall(yhat,label,lr,lt,True,epoch)
        epoavg=sum(epochloss)/len(epochloss)
        losses.append(epoavg)
        if epoch % 10 == 0:
            print(f'epoch:{epoch}......................')
            print(f'rpoch{epoch},loss={epoavg:.4f}')
    return model
transform = transforms.Compose([
    transforms.Resize((25,25)),
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
dog = Manipulating()
dog.construction(label_path='D:/networks_basic/dogs/labels.csv')
newpak=dog.deleteCate(80)
newselect=dog.reselect(10,newpak)

nopack=DogSet("D:/networks_basic/dogs/train/train",'D:/networks_basic/dogs/labels.csv',transform,newselect,batch_size=100)
labelpairs=nopack.label_pairs
feats,labels=testprepare(nopack.idDic,nopack.res,5,labelpairs,25,25,transform)
nopack.adjust()
nums=len(nopack.getlabels())
model=NetCNN(nums,useAdam=True)
optimi=torch.optim.Adam(model.parameters(),lr=0.00002)
model.apply_opt(optimi)
model=train(model,nopack,400,0.00002,lt='crossentropy')
# optimi=torch.optim.Adam(model.parameters(),lr=0.00002)
model.eval()
with torch.no_grad():
    out=model(feats,2)
    predict=torch.argmax(out,dim=1)
    print(predict)
    print(labels)