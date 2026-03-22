import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from initial_orth import Orthogonalize,Propagation,preL,register,Adamregistry
import Intialdots
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from abc import abstractmethod
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
class cOrthoLoss(preL):
    @register('categorycrossentropy')
    @staticmethod
    def categoryLoss(y_true,y_pred):
        eps = 1e-7
        y_pred=tf.clip_by_value(y_pred,eps,1-eps)
        log_probs=tf.math.log(y_pred)               # (batch, classes)
        sample_loss=-tf.reduce_sum(y_true*log_probs, axis=1)  # (batch,)
        loss = tf.reduce_mean(sample_loss)
        return loss
    @register('Frobenius')
    @staticmethod
    def frobeniusLoss(W:tf.Tensor,leta=0.3):
        '''
        monitor and estimate the deviation
        '''
        WT_W=tf.matmul(W,W,transpose_b=True)
        I=tf.eye(W.shape[0])
        diff=WT_W-I
        loss=tf.reduce_sum(diff**2)
        return leta*loss
    @register('singular')
    @staticmethod
    def singularLoss(W:tf.Tensor):
        s=tf.linalg.svd(W,compute_uv=False)
        loss=tf.reduce_sum((s-1.)**2)
        return loss
class orthPropogation(Propagation):
    def __init__(self,model,useriemann,leta=0.3,useadam=True,**kw):
        super().__init__(model,useriemann,useadam,**kw)
        self.leta=leta
    def orthoF_gradient(self,layerobj:Normallayer):
        layerweight=layerobj.kernel
        cooresI=tf.eye(layerweight.shape[0])
        dW=4*self.leta*tf.matmul((tf.matmul(layerweight,layerweight,transpose_b=True)-cooresI),layerweight)
        return dW
    @abstractmethod
    def orthS_gradient(self,*args,**kwrags):
        pass
    def runall(self,y_hat:tf.Tensor,target:tf.Tensor,learning_rate,
               losstype='categorycrossentropy',useadam=True,e=0):
        self.lt=losstype
        layerparams:dict=self.model.Layers
        all_layers=[x[-1] for x in layerparams.values()]#+[self.model.outlayer]
        losstart=self.loss_gradient(y_hat,target)
        for idx,layer in enumerate(reversed(all_layers)):
            output,x_in,layerobj=layerparams[layer.name]#[output,input,layetobj]
            #distinguish the last layer
            if layer.name == self.model.outlayer.name:
                dLdz=losstart
                dLdW=tf.matmul(tf.transpose(x_in),dLdz)
                dLdb=tf.reduce_sum(dLdz,axis=0)
                losstart=tf.matmul(dLdz,tf.transpose(layerobj.kernel))
            else:
                losstart,dLdW,dLdb=self.dense_gradient(losstart,x_in,output,layer)
                dOdW=self.orthoF_gradient(layer)
                if not useadam:
                    #calculate the weighted dW
                    W_all=(1-self.leta)*dLdW+dOdW
                    # W_all=dLdW
                    layerobj.kernel.assign_sub(learning_rate*W_all)
                    layerobj.bias.assign_sub(learning_rate*dLdb)
                    continue
                dw,db=self.momentumcore(layerobj,dLdW,dLdb)
                dodW=self.orthoF_gradient(layerobj)
                W_all=(1-self.leta)*dw+dodW
                layerobj.kernel.assign_sub(learning_rate*W_all)
                layerobj.bias.assign_sub(learning_rate*db)
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
def train(datasets,epoches,adaming,reiman,learning_rate,leta=0.3,onehot_process=False,**kw):
    lt=kw.get('losstype','categorycrossentropy')
    warnings.warn('if loss type is categorycrossentropy, please notice that labels must be one_hot codes,' \
    'if not,use onehot_process=True and pass the number of categories in kw')
    model=Forward(4,[(12,'relu'),(36,'relu'),(18,'relu')],useadam=adaming)
    betas={}
    if kw.get('beta1') is not None:
        betas['beta1']=kw.get('beta1')
    if kw.get('beta2') is not None:
        betas['beta2']=kw.get('beta2')
    backward=orthPropogation(model,reiman,leta,adaming,**betas)
    lossins=cOrthoLoss()
    losses=[]
    '''
    we decide to run for one epoch ,during this epoch we register layers params in model.Layers
    '''
    OrhLoss=[]
    for epoch in range(epoches):
        epoch_loss=[]
        for batchx,batchy in datasets:
            batchint=tf.cast(batchy,tf.int32)
            labels=tf.one_hot(batchint,depth=kw.get('categories')) if onehot_process else batchy
            labels=tf.squeeze(labels)
            yhat=model(batchx,epoch=epoch)
            # put ortho loss in overall loss
            ortho_loss=0.
            if epoch >= 1:
                for layername,layerparams in otrhofilter:
                    layer=layerparams[-1]
                    ortho_loss+=lossins.registry['Frobenius'](layer.kernel)
            else:
                otrhofilter=filter(lambda item:item[0] != model.outlayer.name,model.Layers.items())
                otrhofilter=list(otrhofilter)
            all_loss=lossins.registry[lt](labels,yhat)*(1-leta)
            OrhLoss.append(ortho_loss)
            epoch_loss.append(all_loss)
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
    return model,losses,OrhLoss
datas=xlsx_tf("D:/data_try.xlsx",[('feature1','feature2',True),('target',False)],batch=180)
model,alloss,Orloss=train(datas,2000,True,True,0.0001,leta=0.5,onehot_process=True,categories=4)
plt.plot(alloss,label='normal loss')
plt.plot(Orloss,label='ortho loss')
plt.show()
#try to predict
# x=tf.linspace(0.,10.,10)
# y=tf.linspace(0.,10.,10)
# X,Y=tf.meshgrid(x,y)
# points=tf.reshape(tf.stack([X,Y],axis=-1),[-1,2])
# pre=predicting(model,points,True,True)
# print(y.shape,'shape')
# result = pd.DataFrame({
#     "x": tf.reshape(X,[-1]).numpy(),
#     "y": tf.reshape(Y,[-1]).numpy(),
#     "pred": tf.reshape(pre,[-1]).numpy()
# })
# path=os.getcwd()
# result.to_excel(os.path.dirname(path)+'/resul.xlsx',index=False)
# print('pk')
# plt.scatter(
#     points[:,0],
#     points[:,1],
#     c=pre,
#     cmap='viridis' 
# )
# plt.colorbar()
# plt.show()