import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import random
class DogSet(Dataset):
    def __init__(self,pic_dirpath:str,label_path:str=None,transform=None,showdict:dict=None,**kw):
        '''
        showdict:{name:[id1,id2,...],name2:[id1,id2,...],...}
        '''
        self.transform=transform or transforms.ToTensor()
        self.nums=pd.read_csv(label_path)
        self.idDic = {self.nums.iloc[idx, 0]: self.nums.iloc[idx, 1] 
            for idx in range(1, len(self.nums))}
        self.label_path=label_path
        self.pic_dir=pic_dirpath
        if showdict is not None:
            self.vatots={}
            func=lambda lis:[os.path.join(pic_dirpath,x+'.jpg') for x in lis]
            self.vatots={k:func(v) for k,v in showdict.items()}
            self.res=[item for sublis in self.vatots.values() for item in sublis]
            random.shuffle(self.res)
            allspecies=list(self.vatots.keys())
            self.label_pairs={allspecies[idx]:idx for idx in range(len(allspecies))}
        else:
            allspeices=list(set(self.idDic.values()))
            spei=len(allspeices)
            self.label_pairs={allspeices[idx]:idx for idx in range(spei)}
        self.batch_size=kw.get('batch_size')
    def adjust(self):
        instances=torch.randint(0,len(self.res)-1,(20,))
        seeshapes=[]
        for picts in instances:
            seeshapes.append(self[picts])
        if len(set(seeshapes)) != 1:
            print('these dataset has different shapes, adjusting them is to processing next')
            flag=False
            if isinstance(self.transform,transforms.ToTensor):
                raise ValueError('no resize method contained in transform, go add it!')
            for t in self.transform.transforms:
                if isinstance(t,transforms.Resize):
                    flag=True
            if not flag:
                raise ValueError('no resize method contained in transform, go add it!')
        else:
            print('they have same shape')
    def __len__(self):
        return len(self.nums)
    def __getitem__(self,idx):
        if hasattr(self,'vatots'):
            getpath=self.res[idx]
            img=Image.open(getpath).convert('RGB')
            if self.transform:
                img=self.transform(img)
            return img
        dir=Path(self.pic_dir)
        self.allpic=[]

        for key in self.idDic:
            path=str(dir/key)+'.jpg'
            if os.path.exists(path):
                self.allpic.append(path)
            else:
                path=str(dir/key).strip()+'.jpg'
                if os.path.exists(path):
                    self.allpic.append(path)
                else:
                    print(path)
                    break
        imgs=Image.open(self.allpic[idx]).convert('RGB')
        getdir=os.path.basename.split('.')[0]
        getcorrespoding_title=self.label_pairs[self.idDic[getdir]]
        if self.transform:
            imgs=self.transform(imgs)
        return imgs,(self.idDic[getdir],getcorrespoding_title)
    def __iter__(self):
        if hasattr(self,'vatots'):
            l=0
            res=[item for sublis in self.vatots.values() for item in sublis]
            random.shuffle(res)
            mp=Image.open(res[0]).convert('RGB')
            if self.transform:
                mp=self.transform(mp)
            example=mp.shape
            l=len(res)
            for j in range(0,l,self.batch_size):
                initialset=torch.zeros(self.batch_size,*example)
                coreslables=torch.zeros(self.batch_size,1,dtype=torch.int64)
                batch_pics=res[j:j+self.batch_size]
                if len(batch_pics) < self.batch_size:
                    initialset=torch.zeros(len(batch_pics),*example)
                    coreslables=torch.zeros(len(batch_pics),1,dtype=torch.int64)
                for idx,rawpath in enumerate(batch_pics):
                    subpath=os.path.basename(rawpath).split('.')[0]
                    coreslables[idx,0]=self.label_pairs[self.idDic[subpath]]
                    img=Image.open(rawpath).convert('RGB')
                    opt=self.transform(img)
                    initialset[idx,:,:,:]=opt
                yield initialset,coreslables
        # initialset=torch.zeros(self.batch_size,*example)
        else:
            example=self[0][0].shape
            qut=len(self)//40
            for i in range(0,qut,self.batch_size):
                initialset=torch.zeros(self.batch_size,*example)
                coreslables=torch.zeros(self.batch_size,1,dtype=torch.int64)
                batch_pics=getattr(self,'allpic')[i:i+self.batch_size]
                if len(batch_pics) < self.batch_size:
                    initialset=torch.zeros(len(batch_pics),*example)
                    coreslables=torch.zeros(len(batch_pics),1,dtype=torch.int64)
                for idx,path in enumerate(batch_pics):
                    img=Image.open(path).convert('RGB')
                    optimg=self.transform(img)
                    initialset[idx,:,:,:]=optimg
                    getid=os.path.basename(path)
                    getid=os.path.splitext(getid)[0]
                    coreslables[idx,0]=self.label_pairs[self.idDic[getid]]
                yield initialset,coreslables
    def getlabels(self):
        return self.label_pairs
class Manipulating:
    def construction(self,label_path:str):
        self.nums=pd.read_csv(label_path)
        self.idDic = {self.nums.iloc[idx, 0]: self.nums.iloc[idx, 1] 
              for idx in range(1, len(self.nums))}
        species=list(self.idDic.values())
        uniquespe=np.unique(species)
        self.sparsek={}
        for species in uniquespe:
            filtered=dict(filter(lambda item:item[1] == species,self.idDic.items()))
            self.sparsek[species]=list(filtered.keys())#{species1:[id0,id1,id2,...],species2:[id0,id1,id2,...]}
        lebels=list(self.sparsek.keys())
        size=[len(x) for x in self.sparsek.values()]
        self.lebels,self.size=lebels,size
        fig,ax=plt.subplots(figsize=(8,8))
        wedge,tests,autotexts=ax.pie(
            size,
            labels=lebels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4)
        )
        for text in tests:
            x, y = text.get_position()
            angle = np.arctan2(y, x) * 180 / np.pi 
            if angle > 90 or angle < -90:
                angle += 180 
            text.set_rotation(angle)
            text.set_ha('center')
            text.set_va('center')
            text.set_fontsize(9)
        plt.setp(autotexts,size=12)
        ax.set_title("dogs' species")
        plt.tight_layout()
        plt.show()
    def reselect(self,select_number,speciesdict=None):
        if speciesdict is not None:
            minnum=min([len(x) for x in speciesdict.values()])
        else:
            minnum=min([len(x) for x in self.sparsek.values()])
        if select_number > minnum:
            raise ValueError('please reduce your select number in order to make sure every category has enough margin')
        newselect={k:random.sample(v,select_number) for k,v in self.sparsek.items()} if speciesdict is None else\
        {k:random.sample(v,select_number) for k,v in speciesdict.items()}
        lebels=list(self.sparsek.keys()) if speciesdict is None else list(newselect.keys())
        size=[len(x) for x in self.sparsek.values()] if speciesdict is None else [len(x) for x in newselect.values()]
        self.lebels,self.size=lebels,size
        fig,ax=plt.subplots(figsize=(8,8))
        wedge,tests,autotexts=ax.pie(
            size,
            labels=lebels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4)
        )
        for text in tests:
            x, y = text.get_position()
            angle = np.arctan2(y, x) * 180 / np.pi 
            if angle > 90 or angle < -90:
                angle += 180 
            text.set_rotation(angle)
            text.set_ha('center')
            text.set_va('center')
            text.set_fontsize(9)
        plt.setp(autotexts,size=12)
        ax.set_title("dogs' species")
        plt.tight_layout()
        plt.show()
        return newselect
    def deleteCate(self,modes):
        '''
        modes means discard the smallest modes % of the proportion
        '''
        nimp={key:var for key,var in zip(self.lebels,self.size)}
        numb_paires={k:nimp[k] for k in sorted(nimp,key=lambda x:nimp[x])}
        pmod=np.percentile(self.size,modes)
        newsizes=dict(filter(lambda item: item[1] >= pmod,numb_paires.items()))
        lebels=list(newsizes.keys())
        size=list(newsizes.values())
        self.lebels,self.size=lebels,size
        fig,ax=plt.subplots(figsize=(8,8))
        wedge,tests,autotexts=ax.pie(
            size,
            labels=lebels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4)
        )
        for text in tests:
            x, y = text.get_position()
            angle = np.arctan2(y, x) * 180 / np.pi 
            if angle > 90 or angle < -90:
                angle += 180 
            text.set_rotation(angle)
            text.set_ha('center')
            text.set_va('center')
            text.set_fontsize(9)
        plt.setp(autotexts,size=12)
        ax.set_title("dogs' species")
        plt.tight_layout()
        plt.show()
        repacked={k:var for k,var in self.sparsek.items() if k in lebels}
        return repacked
def testprepare(idDic,path,selectnumber,inpair:dict,width,height,transform):
    '''
    inpair:{name1:int,name2:int,...}
    '''
    # df=pd.read_csv(labelpath)
    # idDic = {df.iloc[idx, 0]:df.iloc[idx, 1] 
    #         for idx in range(1, len(df))}
    pathli=random.sample(path,selectnumber)
    # func=lambda lis:[os.path.join(path,x+'.jpg') for x in lis]
    # pathli=func(sampletest)
    labelpack=torch.zeros(selectnumber,1)
    zeropack=torch.zeros(selectnumber,3,height,width)
    for idx,rawpath in enumerate(pathli):
        mp=Image.open(rawpath).convert('RGB')
        optimg=transform(mp)
        zeropack[idx,:,:,:]=optimg
        ids=os.path.basename(rawpath).split('.')[0]
        labelpack[idx,0]=inpair[idDic[ids]]
    return zeropack,labelpack
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 强制 224x224
#     transforms.ToTensor(),             # 转为 [0,1] 张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准
#                          std=[0.229, 0.224, 0.225])
# ])

# # 传入 DogSet
# dog = Manipulating()
# dog.construction(label_path='D:/networks_basic/dogs/labels.csv')
# newpak=dog.deleteCate(80)
# newselect=dog.reselect(10,newpak)

# nopack=DogSet("D:/networks_basic/dogs/train/train",'D:/networks_basic/dogs/labels.csv',transform,newselect,batch_size=100)
# for f,l in nopack:
#     print(f.shape)
#     print(l.shape)
# dog.adjust()
# for sate in dog:
#     features,labels=sate
#     print(labels[:10,:])
# print(dog.getlabels())