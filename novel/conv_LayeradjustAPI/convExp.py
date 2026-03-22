import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random,os,time
from typing import Literal
from tensorflow.keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D

def register(name):
    def decorator(func):
        def wrapper(*args,**kwargs):
            return func(*args,**kwargs)
        wrapper._register_name=name
        wrapper._register_func=func
        return wrapper
    return decorator
def dataset_select(select_label,select_number,path=None,df=None):
    if df is None and path is not None:
        df=pd.read_csv(path)
    mask=df.iloc[:,-1] == select_label
    nonmask=df.iloc[:,-1] != select_label
    featuresdf=df[mask].iloc[:,0:]
    print(featuresdf.shape)
    another=df[nonmask].iloc[:,0:]
    num,_=featuresdf.shape
    sands=random.sample(range(0,num),select_number)
    selected=featuresdf.iloc[sands]
    new_df=pd.concat([selected,another],axis=0,ignore_index=True)
    return new_df
def datasetcsv(filepath:str=None,df=None):
    assert not (filepath is not None and df is not None),'only one of the params filepath and df can pass a not None value'
    if df is None and filepath is not None:
        df=pd.read_csv(filepath)
    num_cols=len(df.columns)
    lom=[0.0]*num_cols
    if df is not None:
        # Convert DataFrame into TensorFlow Dataset
        features = df.iloc[:, :-1].values  #All columns except the last one (features)
        labels = df.iloc[:, -1].values.astype(int)  # Last column (labels)
        # Create a TensorFlow dataset from features and labels
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        ds=tf.data.experimental.CsvDataset(filepath,lom)
        ds=ds.map(lambda *rows:(tf.stack(rows[:-1]),tf.cast(rows[-1],tf.int32)))
    ds=ds.shuffle(100000)
    # ds=ds.map(lambda feat,label:(tf.expand_dims(feat,-1),label))
    ds=ds.map(lambda feat,label:(tf.expand_dims(feat,-1),tf.one_hot(label,depth=len(np.unique(df.iloc[:, -1].to_numpy())))))#x is already a tensor
    ds=ds.batch(10000)
    return ds,df.shape[1],len(np.unique(df.iloc[:,-1].to_numpy()))
class Adamparameters:
    def __init__(self,var):
        # one order adam
        self.V=tf.Variable(initial_value=tf.zeros_like(var),trainable=False,name='_v')
        # two order adam
        self.M=tf.Variable(initial_value=tf.zeros_like(var),trainable=False,name='_M')

class Adamregistry:
    registries={}
    @staticmethod
    def register(layerobj:tf.keras.layers.Layer):
        '''
        one layer always has several parameters,we need to match a adamparam for each parameter in this layer
        and we use a dict to contain such a adamparam-originalparam pairs
        '''
        state={}
        for param in layerobj.trainable_variables:
            #use ref() as key because common param cannot be key
            state[id(param)]=Adamparameters(param)
        Adamregistry.registries[id(layerobj)]=state#equivalent to a layers dict including a detailed parameters' dict
    @staticmethod
    def get_adam(layerobj,var):
        return Adamregistry.registries[id(layerobj)][id(var)]
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
    def __init__(self,units,activate,**kw):
        super().__init__()
        self.units=units
        self.activation=activate
        self.act=tf.keras.activations.get(activate)
        self.kw=kw
    def build(self,input_shape):
        self.kernel=self.add_weight(name='kernel',shape=(input_shape[-1],self.units),trainable=True,initializer='random_normal')
        self.bias=self.add_weight(name='bias',shape=(self.units,),trainable=True,initializer='zeros')
    def call(self,inputs):
        return self.act(tf.matmul(inputs,self.kernel)+self.bias)

@with_adam
class CustconvolutionLayer(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides=1,padding='same'):
        super().__init__()
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
    def build(self,input_shape):
        self.kernel=self.add_weight(name='kernel',shape=(self.kernel_size,input_shape[-1],
                                        self.filters),initializer='glorot_uniform',trainable=True)
        self.bias=self.add_weight(name='bias',shape=(self.filters,),initializer='zeros',
                                  trainable=True)
    def call(self,inputs):
        conv_output=tf.nn.conv1d(inputs,self.kernel,stride=self.strides,padding=self.padding.upper())
        conv_output=tf.nn.bias_add(conv_output,self.bias)
        return tf.nn.relu(conv_output)
class cosconvModel(tf.keras.Model):
    def __init__(self,layerlist,conv_size,pooltype:Literal['maxing','avging'],useadam=False,**kw):
        super().__init__()
        self.layerDense1=tf.keras.models.Sequential([
            Normallayer(units=x,activate=y) for x,y in layerlist
        ])
        self.layerDense2=tf.keras.models.Sequential([
            Normallayer(units=x,activate=y) for x,y in layerlist
        ])
        if useadam:
            self.t_wb=0
        #remember the input shape:batch,timespan,channel
        self.layers_cache={}#design it as :name of the layer:[output,input,layerobj]
        self.convLayer=CustconvolutionLayer(filters=conv_size,**kw)
        self.convLayer2=CustconvolutionLayer(filters=conv_size,**kw)
        self.poollayer=tf.keras.layers.GlobalMaxPooling1D() if pooltype == 'maxing' else tf.keras.layers.GlobalAveragePooling1D()
        self.poollayer2=tf.keras.layers.GlobalMaxPooling1D() if pooltype == 'maxing' else tf.keras.layers.GlobalAveragePooling1D()
        self.outputlayer=tf.keras.layers.Dense(units=5,activation='softmax')
    def call(self,input:tf.Tensor,epoch):
        if epoch >= 1:
            layers=[layerparams[-1] for layerparams in self.layers_cache.values()]
            for layerobj in layers:
                if isinstance(layerobj,CustconvolutionLayer) and input.ndim == 2:
                    input=tf.expand_dims(input,axis=-1)
                elif isinstance(layerobj,Normallayer) and input.ndim == 3:
                    input= tf.keras.layers.Flatten()(input)
                self.layers_cache[layerobj.name][1]=input
                input=layerobj(input)
                self.layers_cache[layerobj.name][0]=input
          
            return input
        assert input.ndim == 3,'conv layer needs 3d input,which contains batch_size for the first dimension,timespan for the second dimension,'
        'and channel for the last dimension'
        self.layers_cache[self.convLayer.name]=[input,self.convLayer]
        outconv=self.convLayer(input)
        self.layers_cache[self.convLayer.name].insert(0,outconv)
        self.layers_cache[self.poollayer.name]=[outconv,self.poollayer]
        poolout=self.poollayer(outconv)
        self.layers_cache[self.poollayer.name].insert(0,poolout)
        flatten_out = tf.keras.layers.Flatten()(poolout)
        for layer in self.layerDense1.layers:
            self.layers_cache[layer.name]=[flatten_out,layer]
            flatten_out=layer(flatten_out)
            self.layers_cache[layer.name].insert(0,flatten_out)
        print(flatten_out[34:39,:6],'flat')
        aftflaten=tf.expand_dims(flatten_out,axis=-1)
        self.layers_cache[self.convLayer2.name]=[aftflaten,self.convLayer2]
        convout=self.convLayer2(aftflaten)
        self.layers_cache[self.convLayer2.name].insert(0,convout)

        self.layers_cache[self.poollayer2.name]=[outconv,self.poollayer2]
        poolout=self.poollayer(convout)
        self.layers_cache[self.poollayer2.name].insert(0,poolout)

        flatten_out = tf.keras.layers.Flatten()(poolout)
        for layer in self.layerDense2.layers:
            self.layers_cache[layer.name]=[flatten_out,layer]
            flatten_out=layer(flatten_out)
            self.layers_cache[layer.name].insert(0,flatten_out)

        self.layers_cache[self.outputlayer.name]=[flatten_out,self.outputlayer]
        output=self.outputlayer(flatten_out)
        self.layers_cache[self.outputlayer.name].insert(0,output)
        print(output[:10,:],'ouy')
        return output
class propagation:
    def __init__(self,model:tf.keras.Model,beta1=0.9,beta2=0.99):
        self.m=model
        self.beta1,self.beta2=beta1,beta2
    def dense_gradient(self,lastgradient,inputx:tf.Tensor,output:tf.Tensor,layer):
        if layer.activation is not None:
            actype=layer.activation
            dactivate=output*(1-output) if actype == 'sigmoid' else tf.cast(output > 0,tf.float32)
            # print(dactivate.shape,'shape---------------------')
            dLdz=lastgradient*dactivate
        else:
            dLdz=lastgradient
        W=layer.kernel
        b=layer.bias
        dLdx=tf.matmul(dLdz,tf.transpose(W))
        # print(dLdx.shape,'/x')
        dLdW=tf.matmul(tf.transpose(inputx),dLdz)
        # print(dLdW.shape,'/w')
        dLdb=tf.reduce_sum(dLdz,axis=0)
        # print(dLdb.shape,'?b')
        return dLdx,dLdW,dLdb
    def loss_gradient(self,output:tf.Tensor,target:tf.Tensor,e):
        if getattr(self,'lt') == 'binarycrossentropy':
            #this way we use the default value of from_logist:0
            #and it is the start of back propagation
            eps=1e-7
            output=tf.clip_by_value(output,eps,1-eps)
            self.dLdyhat=(output-target)/(output*(1-output))
        elif getattr(self,'lt') == 'mseloss':
            self.dLdyhat=output-target
        elif getattr(self,'lt') == 'sparse_categorical_crossentropy':
            num_classes = output.shape[-1]
            target_onehot = tf.one_hot(tf.cast(target, tf.int32), depth=num_classes)
            target_onehot = tf.cast(target_onehot, tf.float32)
            
            self.dLdyhat = output-target_onehot  # (batch, 5)
        else:
            self.dLdyhat=output-target
        return self.dLdyhat
    def MAXpool_gradient(self, lastgradient, layerobj):
        _, inputdata, layer = self.m.layers_cache[layerobj.name]
        max_vals = tf.reduce_max(inputdata, axis=1, keepdims=True)  # (batch, 1, channels)
        mask = tf.cast(tf.equal(inputdata, max_vals), tf.float32)   # (batch, time, channels)
        mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)  # (batch, 1, channels)
        mask = mask / mask_sum
        dx = mask * lastgradient[:, tf.newaxis, :]
        return dx
    def AVGpool_gradient(self,lastgradient,layerobj):
        _,inputdata,layer=self.m.layers_cache[layerobj.name]
        if isinstance(layerobj, tf.keras.layers.GlobalAveragePooling1D):
            # For GlobalAveragePooling1D, the gradient is simply the mean of the gradients across the time axis
            dx = tf.broadcast_to(lastgradient[:,tf.newaxis,:], inputdata.shape)
            dx=dx/tf.cast(inputdata.shape[1],tf.float32)
            return dx
        pool_size,stride=layer.pool_size,layer.strides
        dx=tf.constant(0.,shape=inputdata)
        # print(dx.shape,'dxxxx')
        for i in range(0,inputdata.shape[0]-pool_size+1,stride):
            window=inputdata[i:i+pool_size]
            avg=np.mean(window)
            dx[i:i+pool_size]+=lastgradient[i//stride]/pool_size
        return dx
    def conv_gradient(self, lastgradient, layerobj):
        '''use einsum to vectorize the code ,make it more simplistic'''
        _, inputs, layer = self.m.layers_cache[layerobj.name]
        batch, timespan, channels = inputs.shape
        kernelsize, _, output_channels = layer.kernel.shape
        _, outputtime, _ = lastgradient.shape
        if layer.padding == 'same':
            output_time = (timespan + layer.strides - 1) // layer.strides
            pad_total = max((output_time - 1)*layer.strides + kernelsize - timespan, 0)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            input_padded = tf.pad(inputs, [[0,0], [pad_left, pad_right], [0,0]])
            dx_padded = tf.Variable(tf.zeros_like(input_padded))
            padded_len = timespan + pad_left + pad_right
        else:
            input_padded = inputs
            dx_padded = tf.zeros_like(inputs)
            pad_left = 0
            padded_len = timespan
        db=tf.reduce_sum(lastgradient,axis=[0,1])  # (output_channels,)
        starts=tf.range(outputtime)*layer.strides
        window_indices=starts[:,None]+tf.range(kernelsize)  # (outputtime, kernelsize)
   
        window_indices = tf.clip_by_value(window_indices, 0, padded_len - 1)

        windows=tf.gather(input_padded,window_indices,axis=1)
        # windows: (B, T_out, K, C_in), lastgradient: (B, T_out, C_out)
        # dw: (K, C_in, C_out)
        dw = tf.einsum('btkc,btq->kcq',windows,lastgradient)
        print("windows shape:", windows.shape)
        print("lastgradient shape:", lastgradient.shape)
        for i in range(outputtime):
            start = i * layer.strides
            t_indices = tf.range(start, start + kernelsize)
            # grad: (batch, output_channels), kernel: (kernelsize, channels, output_channels)
            grad = lastgradient[:, i, :]  # (batch, output_channels)
            # kernel: (K, C_in, C_out), grad: (B, C_out) -> (B, K, C_in)
            print(layer.kernel.shape,grad.shape,'shappppp')
            contribution = tf.einsum('kco,bo->bkc', layer.kernel, grad)
            for j, t_idx in enumerate(t_indices):
                if t_idx < padded_len:
                    dx_padded[:, t_idx, :].assign(dx_padded[:, t_idx, :]+contribution[:, j, :])
        if layer.padding == 'same':
            dx = dx_padded[:, pad_left:pad_left+timespan, :]
        else:
            dx = dx_padded
        return dx,dw,db
    def runall(self,y_hat:tf.Tensor,target:tf.Tensor,learning_rate,
               losstype:Literal['mseloss','binarycrossentropy','categorycrossentropy'],useadam=True,e=0):
        self.m.t_wb+=1
        self.lt=losstype
        layerparams:dict=self.m.layers_cache
        all_layers=[x[-1] for x in layerparams.values()]#+[self.model.outlayer]
        losstart=self.loss_gradient(y_hat,target,e)
        for idx,layer in enumerate(reversed(all_layers)):
            output,x_in,layerobj=layerparams[layer.name]#[output,input,layetobj]
            #distinguish the last layer
            if layer.name == self.m.outputlayer.name:
                dLdz=losstart
                dLdW=tf.matmul(tf.transpose(x_in),dLdz)
                dLdb=tf.reduce_sum(dLdz,axis=0)
                losstart=tf.matmul(dLdz,tf.transpose(layerobj.kernel))
            else:
                if isinstance(layerobj,Normallayer) and len(losstart.shape) == 3:
                    losstart=tf.squeeze(losstart,axis=-1)
                if isinstance(layerobj,GlobalAveragePooling1D):
                    losstart=self.AVGpool_gradient(losstart,layerobj)
                elif isinstance(layerobj,GlobalMaxPooling1D):
                    losstart=self.MAXpool_gradient(losstart,layerobj)
                elif isinstance(layerobj,CustconvolutionLayer):
                    losstart,dLdW,dLdb=self.conv_gradient(losstart,layerobj)
                else:
                    losstart,dLdW,dLdb=self.dense_gradient(losstart,x_in,output,layer)
                if not useadam and not isinstance(layerobj,(GlobalAveragePooling1D,GlobalMaxPooling1D)):
                    #these parameters' updating need to be superseded by rieman way,before it we will complete a adam optimizer first
                    layerobj.kernel.assign_sub(learning_rate*dLdW)
                    layerobj.bias.assign_sub(learning_rate*dLdb)
                    continue
                # if isinstance(layerobj,CustconvolutionLayer):
                #     layerobj.kernel.assign_sub(learning_rate*dLdW)
                #     layerobj.bias.assign_sub(learning_rate*dLdb)
                #     continue
                if not isinstance(layerobj,(GlobalAveragePooling1D,GlobalMaxPooling1D)):
                    dw,db=self.momentumcore(layerobj,dLdW,dLdb)
                    layerobj.kernel.assign_sub(learning_rate*dw)
                    layerobj.bias.assign_sub(learning_rate*db)

    def momentumcore(self,layer,dW,db,eps=1e-6):
        #this is weight update
        Wparams=Adamregistry.get_adam(layer,layer.kernel)
        V_w,M_w=Wparams.V,Wparams.M
        V_w.assign(self.beta1*V_w+(1-self.beta1)*dW)
        M_w.assign(self.beta2*M_w+(1-self.beta2)*tf.square(dW))
        V_hatw=V_w/(1-self.beta1**self.m.t_wb)
        B_hatw=M_w/(1-self.beta2**self.m.t_wb)
        new_dW=V_hatw/(tf.sqrt(B_hatw)+eps)
        #bias update
        bparams=Adamregistry.get_adam(layer,layer.bias)
        V_b,M_b=bparams.V,bparams.M
        V_b.assign(self.beta1*V_b+(1-self.beta1)*db)
        M_b.assign(self.beta2*M_b+(1-self.beta2)*tf.square(db))
        V_hatb=V_b/(1-self.beta1**self.m.t_wb)
        B_hatb=M_b/(1-self.beta2**self.m.t_wb)
        new_db=V_hatb/(tf.sqrt(B_hatb)+eps)
        return new_dW,new_db
def train(dataset:tf.data.Dataset,model,epoch,useadam,lr,losstype,preview_n):
    optimizer=propagation(model)
    loss_fn = tf.keras.losses.MeanSquaredError()
    for epoc in range(epoch):
        print(f"Epoch {epoc+1} of {epoch}")
        epochLoss=[]
        batchloss=[]
        for eback in dataset:
            features,label=eback
            # encode labels first
            print(epoc,'?')
            yhat=model(features,epoch=epoc)
            print(epoc,'?')
            loss=loss_fn(label,yhat)
            optimizer.runall(yhat,label,lr,losstype,useadam,epoc)
            batchloss.append(loss.numpy())
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
            print(f"  sample{preview_count+1}: True value={label[i].numpy()}, predict value={yhat[i].numpy()}")
            preview_count += 1
            if preview_count >= preview_n:
                break
        
        if preview_count >= preview_n:
            break
    print("-" * 40)
    print()

newdf=dataset_select(0.,600,path="D:/networks_basic/heartbeat/mitbih_train.csv")
newdf=dataset_select(1.,600,df=newdf)
newdf=dataset_select(2.,600,df=newdf)
newdf=dataset_select(3.,600,df=newdf)
newdf=dataset_select(4.,600,df=newdf)
d,featuresnumber,categories=datasetcsv(df=newdf)
model=cosconvModel([(32,'relu'),(64,'relu'),(64,'relu')],86,'maxing',kernel_size=5,useadam=True)
train(d,model,1,True,0.001,'mseloss',5)