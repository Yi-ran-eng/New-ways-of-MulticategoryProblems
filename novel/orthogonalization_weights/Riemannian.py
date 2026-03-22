import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from initial_orth import Orthogonalize,Propagation,Adamregistry
import pandas as pd
import matplotlib.pyplot as plt
from simple_Datasets import _Dataset
import warnings
import Intialdots
warnings.filterwarnings("ignore", category=UserWarning)
def with_adam(cls):
    class Wrapper(cls):
        '''
        this Wrapper is a subclass of layerclass
        '''
        def build(self,input_shape):
            #this is rewriting build method
            super().build(input_shape)
            #after build ,we  already have trainable_variables
            Adamregistry.register(self)
    return Wrapper
@with_adam
class Normallayer(tf.keras.layers.Layer):
    def __init__(self,units,activate,normalize=True,**kw):
        super().__init__()
        self.units=units
        self.activate=activate
        self.act=tf.keras.activations.get(activate)
        self.nor=normalize
        self.kw=kw
    def build(self,input_shape):
        self.kernel=self.add_weight(name='kernel',shape=(input_shape[-1],self.units),trainable=True,initializer='random_normal')
        self.bias=self.add_weight(name='bias',shape=(self.units,),trainable=True,initializer='zeros')
        if self.nor:
            with Orthogonalize(way=self.kw.get('way','qr')) as orthor:
                EPS={'eps':self.kw.get('eps')} if self.kw.get('eps') is not None else {}
                q=orthor(self.kernel,**EPS)
                self.kernel.assign(q)
        if self.kernel.shape[0] >= self.kernel.shape[1]:
            # column orth
            orth_err = tf.norm(
                tf.matmul(self.kernel, self.kernel, transpose_a=True) - tf.eye(self.kernel.shape[1])
            )
        else:
            # row orth
            orth_err = tf.norm(
                tf.matmul(self.kernel,self.kernel, transpose_b=True)-tf.eye(self.kernel.shape[0])
            )
        print("orth error:", orth_err.numpy(),'intial')
    def call(self,inputs):
        return self.act(tf.matmul(inputs,self.kernel)+self.bias)
class Forward(tf.keras.Model):
    def __init__(self,outputdim,layers:list,useadam=True,userieman=True):
        super().__init__()
        if useadam:
            self.t_wb=0
        self.Layers={}#name:[output,input,layerobj]
        self.layersq=tf.keras.Sequential([Normallayer(number,activate=act,normalize=userieman) for number,
                                            act in layers])
        self.outlayer=tf.keras.layers.Dense(outputdim,activation='softmax')#a multicategory problem
    def call(self,features,epoch):
        for layer in self.layersq.layers:
            if epoch == 0:
                self.Layers[layer.name]=[features,layer]
            else:
                self.Layers[layer.name][1]=features
            features=layer(features)
            if epoch == 0:
                self.Layers[layer.name].insert(0,features)
            else:
                self.Layers[layer.name][0]=features
        # self.Layers[self.outlayer.name]=[features,self.outlayer]
        output=self.outlayer(features)
        if epoch == 0:
            self.Layers[self.outlayer.name]=[features,self.outlayer]
            self.Layers[self.outlayer.name].insert(0,output)
        else:
            self.Layers[self.outlayer.name][0]=output
            self.Layers[self.outlayer.name][1]=features
        return output
    # def build(self,input_shape):
    #     self.layersq.build(input_shape)
    #     self.outlayer.build((None,self.layersq.layers[-1].units))
    #     super().build(input_shape)
#prepare the datasets
def xlsx_tf(path,head:list[tuple],batch=None,useshuffle=True)->list[tf.Tensor]:
    '''
    head:[(head1,head2,...),(headi,headk,...)]
    return [tensor(head1,head2,...),tensor(headi,headk,...)]
    '''
    df=pd.read_excel(path)
    lispack=[]
    for cluster in head:
        df_toprocessing=df[list(cluster)[:-1]]
        tensor=tf.convert_to_tensor(df_toprocessing.values,dtype=tf.float32)
        if cluster[-1]:
            nor=Intialdots.normalize_centralize(tensor)
            tensor=nor.backcentral(tensor)
        lispack.append(tensor)
    if batch is None:
        return lispack
    #use generator to throw out batch 
    # long=lispack[0].shape[0]
    if useshuffle:
        long=lispack[0].shape[0]
        indices = tf.random.shuffle(tf.range(long))
        tensor = [tf.gather(t, indices) for t in lispack]
    else:
        tensor=lispack
    dataset=tf.data.Dataset.from_tensor_slices(tuple(tensor))
    datasets=dataset.batch(batch)
    return datasets
def register(name):
    def decorator(func):
        def wrapper(*args,**kwargs):
            return func(*args,**kwargs)
        wrapper._register_name=name
        wrapper._register_func=func
        return wrapper
    return decorator
class preL:
    registry={}
    def __init_subclass__(cls,**kw):
        super().__init_subclass__(**kw)
        for name,method in cls.__dict__.items():
            if hasattr(method,'_register_name'):
                cls.registry[method._register_name]=method._register_func
class Losses(preL):
    # @staticmethod
    # def register(name):
    #     def decorator(func):
    #         Losses.registry[name] = func
    #         return func
    #     return decorator
    @register('mseloss')
    @staticmethod
    def manual_mse_loss(y_true, y_pred):
        error = y_true-y_pred # (batch_size, output_dim)
        squared_error = tf.square(error)  # (batch_size, output_dim)
        sample_loss = tf.reduce_sum(squared_error, axis=1)  # (batch_size,)
        loss = tf.reduce_mean(sample_loss)
        
        return loss
    @register('categorycrossentropy')
    @staticmethod
    def manual_categorical_crossentropy(y_true,y_pred):
        eps = 1e-7
        y_pred=tf.clip_by_value(y_pred,eps,1-eps)
        log_probs=tf.math.log(y_pred)               # (batch, classes)
        sample_loss=-tf.reduce_sum(y_true*log_probs, axis=1)  # (batch,)
        loss = tf.reduce_mean(sample_loss)
        return loss

def train(datasets,epoches,adaming,reiman,learning_rate,onehot_process=False,**kw):
    lt=kw.get('losstype','categorycrossentropy')
    warnings.warn('if loss type is categorycrossentropy, please notice that labels must be one_hot codes,' \
    'if not,use onehot_process=True and pass the number of categories in kw')
    model=Forward(4,[(12,'relu'),(36,'relu'),(18,'relu')],useadam=adaming)
    betas={}
    if kw.get('beta1') is not None:
        betas['beta1']=kw.get('beta1')
    if kw.get('beta2') is not None:
        betas['beta2']=kw.get('beta2')
    backward=Propagation(model,reiman,adaming,**betas)
    lossins=Losses()
    losses=[]
    for epoch in range(epoches):
        epoch_loss=[]
        for batchx,batchy in datasets:
            batchint=tf.cast(batchy,tf.int32)
            labels=tf.one_hot(batchint,depth=kw.get('categories')) if onehot_process else batchy
            labels=tf.squeeze(labels)
            yhat=model(batchx,epoch=epoch)
            epoch_loss.append(lossins.registry[lt](labels,yhat))
            backward.runall(yhat,labels,learning_rate,losstype=lt,useadam=adaming,e=epoch)
        avgloss=tf.reduce_mean(epoch_loss)
        if epoch % 100 == 0:
            print(f'{epoch}-loss{avgloss:.4f}')
        losses.append(avgloss)
    if kw.get('searesult',False):
        for batchx, batchy in datasets.take(1):
            batchint = tf.cast(batchy, tf.int32)
            labels = tf.one_hot(batchint, depth=kw.get('categories')) if onehot_process else batchy
            labels = tf.squeeze(labels)
            yhat = model(batchx, epoch=epoches)
            true_cls = tf.argmax(labels[:5], axis=1)
            pred_cls = tf.argmax(yhat[:5], axis=1)

            print("true:",true_cls.numpy())
            print("pred:",pred_cls.numpy())
    X=tf.concat([x for x,_ in datasets], axis=0)
    Y=tf.concat([y for _,y in datasets], axis=0)
    yhat=model(X,epoch=0)   # forward
    pred_cls=tf.argmax(yhat, axis=1)
    plt.figure(figsize=(6,6))
    plt.scatter(
        X[:,0],
        X[:,1],
        c=pred_cls,
        cmap="tab10",
        s=30
    )

    plt.colorbar(ticks=[0,1,2,3])
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.title("Prediction on Predict Data")

    plt.show()
    return model,losses
def predicting(model,datas,onezero_opt:bool=False,onhot_opt:bool=False):
    '''
    datas:[features:tf.tensors,labels:tf.tensors],they are wrapped in a list
    '''
    if onezero_opt:
        norm=Intialdots.normalize_centralize()
        features=norm.backcentral(datas)
    yhat = model(features,epoch=0)

    if onhot_opt:
        yhat=tf.argmax(yhat,axis=1)
    return yhat
#prepare dataset:plot dots by hands
# ds = _Dataset((0, 10), (0, 10), 100, category=4)
# ds.rawpic()
# plt.show()
# ds._to_Dataframe(outpath="D:/data_try.xlsx")
datas=xlsx_tf("D:/data_try.xlsx",[('feature1','feature2',True),('target',False)],batch=180)
model,_=train(datas,5000,True,True,0.0001,True,categories=4)
#try to predict
x=tf.linspace(0.,10.,10)
y=tf.linspace(0.,10.,10)
X,Y=tf.meshgrid(x,y)
points=tf.reshape(tf.stack([X,Y],axis=-1),[-1,2])
pre=predicting(model,points,True,True)
print(y.shape,'shape')
result = pd.DataFrame({
    "x": tf.reshape(X,[-1]).numpy(),
    "y": tf.reshape(Y,[-1]).numpy(),
    "pred": tf.reshape(pre,[-1]).numpy()
})
path=os.getcwd()
result.to_excel(os.path.dirname(path)+'/resul.xlsx',index=False)
print('pk')
plt.scatter(
    points[:,0],
    points[:,1],
    c=pre,
    cmap='viridis' 
)
plt.colorbar()
plt.show()