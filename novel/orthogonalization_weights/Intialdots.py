import pandas as pd
from scipy.stats import norm
import numpy as np
import tensorflow as tf


class normalize_centralize:

    sortedarray=[]

    def __init__(self,*args,**kw):

        if (not kw) and args:
            x=args[0]
        elif (not args) and kw:
            x=kw.get('x')
        else:
            x=None

        if x is not None:
            self.backend = "tf" if isinstance(x,tf.Tensor) else "np"

            if self.backend == "tf":
                self.newx=tf.zeros_like(x,dtype=tf.float32)
            else:
                self.newx=np.zeros(x.shape)

    # ===============================
    # 中心化归一化
    # ===============================
    def backcentral(self,x):

        self.backend = "tf" if isinstance(x,tf.Tensor) else "np"

        if self.backend=="tf":

            maxx=tf.reduce_max(x,axis=0)
            minx=tf.reduce_min(x,axis=0)

            medial=(maxx+minx)/2
            crssed=maxx-minx+1e-8

            return (x-medial)/crssed

        else:

            self.newx=np.zeros(x.shape)

            for feat in range(x.shape[1]):

                allx_i=x[:,feat]

                maxx,minx=allx_i.max(),allx_i.min()

                medial=(maxx+minx)/2

                crssed=maxx-minx

                self.newx[:,feat]=(allx_i-medial)/crssed

            return self.newx
    def backzero_one(self,x):
        self.backend = "tf" if isinstance(x,tf.Tensor) else "np"
        if self.backend=="tf":
            maxx=tf.reduce_max(x,axis=0)
            return x/(maxx+1e-8)
        else:
            self.newx=np.zeros(x.shape)
            for feat in range(x.shape[1]):
                allx_f=x[:,feat]
                maxx=allx_f.max()
                self.newx[:,feat]=allx_f/maxx
            return self.newx
    def backBox_Nor(self,x):
        if isinstance(x,tf.Tensor):
            x=x.numpy()
        filt=x>0
        assert filt.all(),'input datas must be positive'
        samples=x.shape[0]
        self.sa=samples
        features=x.shape[1]
        p=[]

        for feat in range(x.shape[1]):
            xpiece=x[:,feat].squeeze()
            xsorted=np.sort(xpiece)
            self.sortedarray.append(xsorted)
            for s in range(1,samples+1):
                p.append(s/(samples+1))
            newps=np.array([
                norm.ppf(x) for x in p
            ])
            insetdic={
                xsorted[k]:newps[k] for k in range(samples)
            }

            setattr(self,f'featuredic_{feat}',insetdic)

        newx=np.zeros_like(x)

        m=0

        while m<features:

            newx[:,m]=np.array(
                [getattr(self,f'featuredic_{m}')[v] for v in x[:,m]]
            )

            m+=1

        return newx


    # ===============================
    # 新样本正态映射
    # ===============================
    def addnewx_Nor(self,x,featnum:int):

        if not normalize_centralize.sortedarray:
            raise ValueError('需要先运行backBox_Nor函数')

        gettarget=normalize_centralize.sortedarray[featnum]

        idx=np.searchsorted(gettarget,x)

        if 1<=idx<=self.sa-2:

            xraw,xnxt=gettarget[idx],gettarget[idx+1]

            yraw=getattr(self,f'featuredic_{featnum}')[xraw]

            ynxt=getattr(self,f'featuredic_{featnum}')[xnxt]

            ynew=yraw+(x-xraw)/(xnxt-xraw)*(ynxt-yraw)

        elif idx==0:

            ynew=norm.ppf(1e-4)

        else:

            ynew=norm.ppf(1-1e-4)

        return ynew