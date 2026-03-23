import torch
from torch import nn
import torch.nn.functional as F
from dataLoad_t import FluentData
import numpy as np
from Auto_Manual_Mixed import AtributeLayer,Propagation,with_adam
from name_register import getcount
@with_adam
@getcount
class Normallayer(AtributeLayer):
    def __init__(self, units, input_shape,activate,name,usemanual=True,**kw):
        super().__init__()
        self.manual_back=usemanual
        if not usemanual:
            self.opt=None
        self.units = units
        self.activation = activate
        if activate != '':
            self.act = getattr(F, activate)  # Get the activation function from F
        self.kw = kw
        self.name=name
        self.build(input_shape)
    def forward(self, inputs):
        if hasattr(self,'act'):
            return self.act(torch.matmul(inputs, self.weight) + self.bias)
        else:
            return torch.matmul(inputs, self.weight) + self.bias

    def build(self, input_shape):
        self.weight = nn.Parameter(torch.empty(input_shape, self.units).uniform_(0.,0.1))  # Initialize kernel
        # print(self.weight[:10,:10],'normalweight')
        self.bias = nn.Parameter(torch.zeros(self.units))  # Initialize bias

@with_adam
@getcount
class CustconvolutionLayer(AtributeLayer):
    def __init__(self, filters, input_shape,kernel_size,name,usemanual=True,strides=1,padding='same',**kw):
        super().__init__()
        self.manual_back=usemanual
        if not usemanual:
            self.opt=None
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name=name
        self.build(input_shape)
    def forward(self, inputs):
        conv_output = F.conv1d(inputs, self.weight, stride=self.strides, padding=self.padding)
        conv_output = conv_output + self.bias.unsqueeze(0).unsqueeze(-1)
        return F.relu(conv_output)

    def build(self, input_shape):
        self.weight=nn.Parameter(torch.empty(self.filters,1,self.kernel_size))  # Initialize kernel
        nn.init.xavier_uniform_(self.weight)
        # self.weight.data=torch.abs(self.weight)
        # print(self.weight[:10,:10],'convweight')
        self.bias = nn.Parameter(torch.zeros(self.filters))  # Initialize bias

class CosConvModel(nn.Module):
    def __init__(self, layerlist, conv_size, pooltype: str = 'maxing', use_adam=False, **kw):
        super().__init__()
        self.normalConv1=Normallayer(units=32,input_shape=86,activate=layerlist[0][1],name='dense1',usemanual=1)
        self.normalConv2=Normallayer(units=32,input_shape=86,activate=layerlist[0][1],name='dense2',usemanual=1)
        self.layerDense1 = nn.Sequential(*[Normallayer(units=layerlist[idx][0],input_shape=layerlist[idx-1][0],activate=layerlist[idx][1],name=f'dense-{idx}',usemanual=True)
                                            for idx in range(1,len(layerlist))])
        self.layerDense2 = nn.Sequential(*[Normallayer(units=layerlist[idx][0],input_shape=layerlist[idx-1][0],activate=layerlist[idx][1],name=f'Dense-{idx}',usemanual=1)
                                            for idx in range(1,len(layerlist))])
        self.convLayer = CustconvolutionLayer(filters=conv_size,input_shape=187,kernel_size=5,usemanual=1,name='conv1')
        self.convLayer2 = CustconvolutionLayer(filters=conv_size,input_shape=64,kernel_size=5,usemanual=1,name='conv2')

        if pooltype == 'maxing':
            self.poollayer = nn.AdaptiveMaxPool1d(1)
            self.poollayer2 = nn.AdaptiveMaxPool1d(1)
        else:
            self.poollayer = nn.AdaptiveAvgPool1d(1)
            self.poollayer2 = nn.AdaptiveAvgPool1d(1)
        self.poollayer.name='pool1'
        self.poollayer2.name='pool2'
        self.poollayer.manual_back=1
        self.poollayer.opt=None
        self.poollayer2.manual_back=1
        self.poollayer2.opt=None
        self.outputlayer = Normallayer(5,layerlist[-1][0],'',name='linear')  # Assuming the final layer's output has the same number of features as `layerlist[-1][0]`
        self.layers_cache={}
    def apply_opt(self,optimizer):
        for module in self.modules():
            if hasattr(module,'opt'):
                module.opt=optimizer
    def forward(self, x, epoch):
        layers = [layer for layer in self.children()]
        
        if epoch >= 1:
            layers=[x[-1] for x in self.layers_cache.values()]
            for layerobj in layers:
                if isinstance(layerobj,CustconvolutionLayer) and x.ndim == 2:
                    x=x.unsqueeze(1)
                elif isinstance(layerobj,Normallayer) and x.ndim == 3:
                    x=x.view(x.size(0), -1)
                self.layers_cache[layerobj.name][1]=x
                x = layerobj(x)
                self.layers_cache[layerobj.name][0]=x
            return x
        assert x.ndimension() == 3, "Conv layer needs 3D input, containing batch_size, timespan, and channel."
        # print(x[:10,:10],'????step1')
        self.layers_cache[self.convLayer.name] = [None,x,self.convLayer]
        outconv = self.convLayer(x)
        self.layers_cache[self.convLayer.name][0] = outconv
        # print(outconv[:10,:10],'outshastep2')
        poolout = self.poollayer(outconv)
        self.layers_cache[self.poollayer.name] = [poolout,outconv, self.poollayer]
        # print(poolout[:5,:5],'pool')
        flatten_out = poolout.view(poolout.size(0), -1)
        self.layers_cache[self.normalConv1.name]=[flatten_out,self.normalConv1]
        flatten_out=self.normalConv1(flatten_out)
        self.layers_cache[self.normalConv1.name].insert(0,flatten_out)

        for layer in self.layerDense1:
            self.layers_cache[layer.name] = [None,flatten_out, layer]
            flatten_out = layer(flatten_out)
            self.layers_cache[layer.name][0] = flatten_out
        aftflaten = flatten_out.unsqueeze(1)
        self.layers_cache[self.convLayer2.name] = [None,aftflaten, self.convLayer2]
        convout = self.convLayer2(aftflaten)
        self.layers_cache[self.convLayer2.name][0] = convout
        poolout2 = self.poollayer2(convout)
        self.layers_cache[self.poollayer2.name] = [poolout2,convout, self.poollayer2]
        flatten_out2 = poolout2.view(poolout2.size(0), -1)
        self.layers_cache[self.normalConv2.name]=[flatten_out2,self.normalConv2]
        flatten_out2=self.normalConv2(flatten_out2)
        self.layers_cache[self.normalConv2.name].insert(0,flatten_out2)
        print(flatten_out2[30:35,:5],'flat2')
        for layer in self.layerDense2:
            self.layers_cache[layer.name] = [None,flatten_out2, layer]
            flatten_out2=layer(flatten_out2)
            self.layers_cache[layer.name][0] = flatten_out2
        self.layers_cache[self.outputlayer.name] = [None,flatten_out2, self.outputlayer]
        output = self.outputlayer(flatten_out2)
        self.layers_cache[self.outputlayer.name][0] = output
        # print(output[:10,:],'pi')
        return output
    
def train(dataset,model,epoch,useadam,lr,losstype,preview_n):
    optimizer=Propagation(model)
    loss_fn =nn.MSELoss()
    for epoc in range(epoch):
        print(f"Epoch {epoc+1} of {epoch}")
        epochLoss=[]
        batchloss=[]
        for eback in dataset:
            features,label=eback
            features=features.unsqueeze(1)
            features=features.requires_grad_(True)
            # encode labels first
            yhat=model(features,epoch=epoc)
            label=label.to(torch.int64)
            loss=loss_fn(yhat,label)
            optimizer.runall(yhat,label,lr,losstype,useadam,epoc)
            batchloss.append(loss.detach().numpy())
        avg=np.mean(batchloss)
        if epoc%1 == 0:
            print(f'epoch{epoc+1},loss:{avg:.4f}')
        epochLoss.append(avg)
    print(f"--- predict the first n samples (the first {preview_n} samples) ---")
    preview_count = 0
    
    for features, label in dataset:
        batch_size = features.shape[0]
        yhat = model(features, epoch=epoc)
        for i in range(min(batch_size, preview_n - preview_count)):
            print(f"  sample{preview_count+1}: True value={label[i].numpy()}, predict value={yhat[i].detach().numpy()}")
            preview_count += 1
            if preview_count >= preview_n:
                break
        
        if preview_count >= preview_n:
            break
    print("-" * 40)
    print()


bonm=FluentData("D:/networks_basic/heartbeat/mitbih_train.csv",-1,True,batch=1000)
bonm.shuffle()
bonm.proc()
model=CosConvModel([(32,'relu'),(64,'relu')],86,'maxing',False)
optimi=torch.optim.Adam(model.parameters(),lr=0.0000005)
model.apply_opt(optimi)
train(bonm,model,5,0,0.0005,'mseloss',2)
