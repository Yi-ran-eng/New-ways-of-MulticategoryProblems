import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel,Field
from itertools import chain
class _Dataset:
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    def __init__(self,x1:tuple,x2:tuple,num:int,category=2):
        self.x1=np.linspace(x1[0],x1[1],num)
        self.x2=np.linspace(x2[0],x2[1],num)
        self.X1,self.X2=np.meshgrid(self.x1,self.x2)
        self.cate=category
        self.btnpams={'height':0.08,'width':0.12,'spacing':0.02,
                      'start_x':0.15}
        self.btns=[]
        self.enlighten=np.zeros(category,)
        #assuming that you have 4 buttons, we give each button a value(0/1) to indicate it's working status
        self.datasets={}
    def _click(self,event):
        if event.inaxes != self.ax:
            return 
        ax=event.inaxes
        clicked=None
        assert np.any(self.enlighten),'please choose one category'
        x,y=event.xdata,event.ydata
        self.xpo.append(x)
        self.ypo.append(y)
        #update graph
        # [scs.set_offsets(np.c_[self.xpo,self.])]
        active=int(np.where(self.enlighten)[0][0])
        try:
            self.datasets[active].append((x,y))
        except KeyError:
            self.datasets[active]=[]
            self.datasets[active].append((x,y))
        self.scse[active].set_offsets(np.c_[self.xpo,self.ypo])
        plt.draw()
    def rawpic(self):
        self.fig,self.ax=plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        plt.grid(True)
        self.ax.set_title('category_dataset')
        self.ax.set_xlim(self.x1[0],self.x1[-1])
        self.ax.set_ylim(self.x2[0],self.x2[-1])
        self.xpo,self.ypo=[],[]
        self.scse=[]
        for cat in range(self.cate+1):
            targetcolor=_Dataset.colors[cat % len(_Dataset.colors)]
            scs=self.ax.scatter([],[],c=targetcolor,s=50,zorder=5)
            setattr(self,f'scatter_{cat}',scs)
            self.scse.append(scs)
        # for btn in range(self.cate):
            x_pos=self.btnpams['start_x']+cat*(self.btnpams['width']+self.btnpams['spacing'])
            btnax=plt.axes([x_pos,0.05,self.btnpams['width'],self.btnpams['height']])
            #set buttons
            if cat != self.cate:
                Btn=Button(btnax,label=f'cat_{cat}',color=targetcolor,hovercolor='0.975')
                Btn.on_clicked(lambda event,idx=cat:self.callback(event,idx,Btn))
                self.btns.append(Btn)
        self.fig.canvas.mpl_connect('button_press_event',lambda event:self._click(event))
        donex=0.75
        doneax=plt.axes([donex,0.05,0.12,0.08])
        self.Done=Button(doneax,label='done',color='lightgreen',hovercolor='0.9')
        self.Done.on_clicked(self._on_click)
    def _on_click(self,event):
        plt.close(self.fig)
    def callback(self,event,idx,obj:Button):
        if hasattr(self,'actidx') and self.actidx == idx:
            return 
        self.actidx=idx
        self.enlighten[:]=0
        self.enlighten[idx]=1
        self.xpo,self.ypo=[],[]
        plt.draw()
    def _to_Dataframe(self,heads=None,**kw):
        print('???')
        colms=[]
        for key,dots in self.datasets.items():
            #where dots are a list of a sort of points
            for dot in dots:
                ixcloms=[dot[0],dot[1],key] if heads is None else [dot[0],dot[1],heads[key]]
                colms.append(ixcloms)
        df=pd.DataFrame(colms,columns=['feature1','feature2','target'])
        savepath=kw.get('outpath')
        if savepath is not None:
            df.to_excel(savepath,index=False)
def _stackindata(datasets:dict,seefront=None):
    #obtain the features' shape
    shape0=len([x for dotgrp in datasets.values() for x in dotgrp])
    idn=0
    for dotsgroup in datasets.items():
        dot,combination=dotsgroup[0],dotsgroup[1]
        shape1=len(combination[0])
        if not 'features' in locals():
            features=tf.Variable(tf.zeros((shape0,shape1),dtype=tf.float32))
        if dot == 0:
            len_0=len(combination)
        # zero_detact=tf.Variab?le(tf.reduce_all(tf.equal(features,0),axis=1))
        # targetrow=tf.where(zero_detact)[0][0]
        for idx,dotpairs in enumerate(combination):
            features[idn,0].assign(dotpairs[0])
            features[idn,1].assign(dotpairs[1])
            idn+=1
    indop=tf.range(len_0,shape0,dtype=tf.int64)
    indecs=tf.stack([indop,tf.zeros_like(indop)],axis=1)
    values=[[dot]*len(colen) for dot,colen in datasets.items() if dot != 0]
    fatvalue=list(chain.from_iterable(values))
    targets=tf.SparseTensor(indices=indecs,values=fatvalue,dense_shape=(shape0,1))
    # print(tf.sparse.to_dense(targets))
    # print(features)
    datas9=tf.data.Dataset.from_tensor_slices((features,targets))
    print(datas9.element_spec)
    if seefront is not None:
        for x,y in datas9.take(seefront):
            print(f'features:{x.numpy()}')
            print(f'targets:{tf.sparse.to_dense(y).numpy()}')
# ds = _Dataset((0, 10), (0, 10), 100, category=4)
# ds.rawpic()
# plt.show()
# _stackindata(ds.datasets,3)

# ds._to_Dataframe(outpath="D:/data_try.xlsx")