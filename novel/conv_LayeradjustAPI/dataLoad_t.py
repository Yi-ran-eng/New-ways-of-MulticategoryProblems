import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Iterator
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
class FluentData(Dataset):
    def __init__(self,datasetpath:str,labelidx:int,processing=False,createfile=False,one_hot=True,**kw):
        self.raw_dataset=pd.read_csv(datasetpath)
        n_cols=self.raw_dataset.shape[1]
        lab=labelidx if labelidx >= 0 else n_cols+labelidx
        self.labels=torch.tensor(self.raw_dataset.iloc[:,lab])

        if one_hot:
            num_classes = len(torch.unique(self.labels))
            self.labels = self.labels.long()
            self.labels = F.one_hot(self.labels, num_classes=num_classes).float()
        self.features=self.raw_dataset.iloc[:,[i for i in range(self.raw_dataset.shape[1]) if i != lab]]
        self.features=torch.tensor(self.features.values)
        self.features,self.labels=self.features[:1000,:],self.labels[:1000]

        self.processed=processing
        if createfile:
            Dapath=Path(datasetpath)
            parent=Dapath.parent
            newpath=parent/'processed_dataset.csv'
            self.newpath2=kw.get('savepath',newpath)
        self.batch_size=kw.get('batch')
    def __len__(self):
        return len(self.raw_dataset)
    def shuffle(self):
        Idx=torch.randperm(len(self.features))
        self.features=self.features[Idx]
        self.labels=self.labels[Idx]
    def __getitem__(self,idx:int | slice):
        label_tensor=self.labels[idx]
        if not self.processed:
            feats_tensor=self.features[idx,:]
            return feats_tensor,label_tensor
        featprocessed:torch.Tensor=self.proc()
        return featprocessed[idx,:],label_tensor
    def proc(self):
        packdatas=torch.zeros(*self.features.shape)
        for bais in range(self.features.shape[0]):
            tensor_row=self.features[bais,:]
            k=0
            while k < self.features.shape[1]-2:
                nmtens=tensor_row[k:k+3]
                if torch.all(nmtens == 0):
                    break
                k+=1
            #when breaking,the value of tensor_row[k] is zero and it is the start of the fill section
            valid=tensor_row[:k]
            target_valid=packdatas[bais,:]
            n_valid = len(valid)
            n_cols = self.features.shape[1]
            
            repeat_times = n_cols // n_valid
            remainder = n_cols % n_valid
            
            for i in range(repeat_times):
                target_valid[i*n_valid : (i+1)*n_valid] = valid
            if remainder > 0:
                target_valid[repeat_times*n_valid:] = valid[:remainder]
            self.repeat=packdatas
        return packdatas
    def __iter__(self):
        k=0
        samps=self.features.shape[0]
        if hasattr(self,'repeat'):
            while k < self.features.shape[0]//self.batch_size:
                feats=self.repeat[self.batch_size*k:min(self.batch_size*(k+1),samps),:]
                labels=self.labels[self.batch_size*k:min(self.batch_size*(k+1),samps)]
                if len(feats) > 0:
                    yield feats,labels
                k+=1
        else:
            while k < self.features.shape[0]//self.batch_size:
                feats=self.features[self.batch_size*k:min(self.batch_size*(k+1),samps),:]
                labels=self.labels[self.batch_size*k:min(self.batch_size*(k+1),samps)]
                if len(feats) > 0:
                    yield feats,labels
                k+=1
    def save(self):
        if not hasattr(self,'repeat'):
            raise ValueError('please run proc first to get the new filled tensor')
        combined=torch.cat([self.repeat,self.labels.unsqueeze(1)],dim=1)
        df=pd.DataFrame(combined.numpy())
        df.to_csv(self,self.newpath2,index=False,header=False)
    def take(self,n_batches)->Iterator:
        for i,(feat,labs) in enumerate(self):
            if i >= n_batches:
                break
            yield feat,labs
