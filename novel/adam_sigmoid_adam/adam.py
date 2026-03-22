'''
in this chapter we also use two category instance to show how momentum algorithm works
'''
import pandas as pd
from dataset_processing import Nanrots
import numpy as np
import tensorflow as tf
import warnings
from typing import Literal
nonan=Nanrots('penguins/Penguindata.csv')
nparray=nonan._deletenan()
def category(array:np.ndarray | tf.Tensor | pd.DataFrame,targethead:list[tuple]):
    '''
    list[tuple] means in this parameter, you can cluster same kind of labels into one tuple,and put them into a list
    then the output will be a series of tensors,every tensor encompasses one tuple's all datas
    '''
    for headcluster in targethead:
        son_df=array[list(headcluster)] if isinstance(array,pd.DataFrame) else array[:,list(headcluster)]
        if isinstance(son_df,pd.DataFrame):
            sonaray=son_df
            looups,encoding={},{}
            processed=[]
            for name,col in sonaray.items():#head and values
                if col.dtype == float:
                    continue
                col=col.values
                # print(col,'col')
                # print(np.unique(son_df[name]),'unqiue')
                looup=tf.keras.layers.StringLookup(vocabulary=np.unique(son_df[name]))
                looups[name]=looup
                onehot_coding=tf.keras.layers.CategoryEncoding(num_tokens=looup.vocabulary_size(),
                                                               output_mode='one_hot')
                encoding[name]=onehot_coding
                x=looup(col)
                codx=onehot_coding(x)
                processed.append(codx)
        elif isinstance(son_df,np.ndarray):
            sonaray=tf.convert_to_tensor(son_df)
        #we transfer input into tf tensor or df(with heads)
        # print(processed,'process')
        tens=tf.concat(processed,axis=1)
        yield tens
def batchcontached(tensor:list[tf.Tensor],batchsize:int):
    '''
    tensor is a list of tftensor,we will cut or divide them into same batch then return a generator
    '''
    firsttensor=tensor[0].shape[0]
    iall=(lambda lis:[x for x in lis if x.shape[0] != firsttensor])(tensor)
    assert len(iall) == 0,'input tensors must have the same 0 dimension'
    # if shuffle:
    indices = tf.random.shuffle(tf.range(firsttensor))
    tensor = [tf.gather(t, indices) for t in tensor]
    countline=1
    cut=lambda lispa,endidx,startidx:[matrix[startidx:endidx,:] for matrix in lispa]
    while (countline-1)*batchsize <= firsttensor:
        start=(countline-1)*batchsize
        last=countline*batchsize
        newlispa=cut(tensor,last,start)
        if newlispa[0].shape[0] != 0:
            yield newlispa
        countline+=1
class CommonL(tf.keras.layers.Layer):
    actdict={'sigmoid':tf.nn.sigmoid,'relu':tf.nn.relu,'linear':lambda x:x,'tanh':tf.nn.tanh}
    def __init__(self,units=32,activation='sigmoid',**kw):
        self.beta1=kw.get('beta1',0.9)
        self.beta2=kw.get('beta2',0.9)
        try:
            kw.pop('beta1')
            kw.pop('beta2')
        except KeyError:
            pass
        super().__init__(**kw)
        self.units=units
        self.activation=activation
    def build(self,input_shape):
        #input_shape :(batch,features)
        self.kernel=self.add_weight(shape=(input_shape[-1],self.units),initializer='glorot_uniform',trainable=True,name='kernel')
        self.bias=self.add_weight(shape=(self.units,),initializer='zeros',trainable=True,name='bias')
        #what have to memtion is that every w has a target V_w,every b has a corresbonding V_b
        #and V_w,V_b come from first gradient,we set the start value is 0
        self.V_w=tf.Variable(tf.zeros_like(self.kernel),trainable=False,name='V_w')
        self.V_b=tf.Variable(tf.zeros_like(self.bias),trainable=False,name='V_b')
        self.B_w=tf.Variable(tf.zeros_like(self.kernel),trainable=False,name='B_w')
        self.B_b=tf.Variable(tf.zeros_like(self.bias),trainable=False,name='B_b')
    def call(self,inputs):
        if self.activation == 'softmax':
            return tf.nn.softmax(tf.matmul(inputs,self.kernel)+self.bias)
        elif self.activation is None:
            return tf.matmul(inputs,self.kernel)+self.bias
        return CommonL.actdict[self.activation](tf.matmul(inputs,self.kernel)+self.bias)
class SigmoidTRAIN(tf.keras.layers.Layer):
    def __init__(self,mu=1,by=0,**kw):
        super().__init__()
        self.mustart,self.bystart=mu,by
        self.s_beta1=kw.get('beta1')
        self.s_beta2=kw.get('beta2')
    def build(self,input_shape):
        # define momentum linked V
        self.V_mu=tf.Variable(0.,trainable=False,name='m_amp')
        self.V_by=tf.Variable(0.,trainable=False,name='m_offset')
        self.B_mu=tf.Variable(0.,trainable=False,name='B_amp')
        self.B_by=tf.Variable(0.,trainable=False,name='B_offset')
        self.mu=tf.Variable(initial_value=self.mustart,trainable=True,dtype=tf.float32,name='amp')
        self.by=tf.Variable(initial_value=self.bystart,trainable=True,dtype=tf.float32,name='offset')
    def call(self,inputs):
        y=lambda x:self.mu/(1+tf.exp(-(x+self.by)))
        return y(inputs)
class ForwardModel(tf.keras.Model):
    def __init__(self,outputunits,units:list[int,str],usemomentum=True,sigmoidtrain=False,**kw):
        '''
        the last activation is suggested to be linear or relu,not sigmoid
        we will apply sigmoid to output layer as logistic function
        '''
        super().__init__()
        self.momentum=usemomentum
        self.sigtrain=sigmoidtrain
        assert all([isinstance(x[0],int) for x in units]),'input must be int'
        if units[-1][1] == 'sigmoid':
            warnings.warn('the last activation is suggested to be linear or relu,not sigmoid')
        mu=kw.pop('mu',None)
        by=kw.pop('by',None)
        self.models=tf.keras.Sequential([tf.keras.layers.Dense(number,activation=act)
                                        for number,act in units]) if not usemomentum else\
                                tf.keras.Sequential([CommonL(x[0],x[1],**kw) for x in units])
        if sigmoidtrain:
            layernum=len(units)
            sigkwargs={}
            if mu is not None:
                sigkwargs['mu']=mu
            if by is not None:
                sigkwargs['by']=by
            sigkwargs['beta1']=kw.get('beta1')
            sigkwargs['beta2']=kw.get('beta2')
            self.sigmoidmodels=tf.keras.Sequential([SigmoidTRAIN(**sigkwargs) for _ in range(layernum)])
        self.outlayer=CommonL(outputunits,activation='softmax',**kw)
        self.layerpacked={}
        self.t_wb=self.t_mb=0#wb is normal linear layer's weights and bias,mb is sigmoid layer's parameters
    def call(self,inputs):
        #name:[out,x_in,layer]
        if self.sigtrain:
            for layer,layersig in zip(self.models.layers,self.sigmoidmodels.layers):
                x_middle=layer(inputs)
                x_out=layersig(x_middle)
                self.layerpacked[layer.name]=[x_middle,inputs,layer]
                self.layerpacked[layersig.name]=[x_out,x_middle,layersig]
                inputs=x_out
        else:
            for layer in self.models.layers:
                x_out=layer(inputs)
                self.layerpacked[layer.name]=[x_out,inputs,layer]
                inputs=x_out
        output=self.outlayer(inputs)
        self.layerpacked[self.outlayer.name]=[output,inputs,self.outlayer]
        # print(output[:10,:],'outshape,很奇怪')
        return output
#this method is intended to show how the back propagation works and implement momentum algorithm
class propagation:
    def __init__(self,model:tf.keras.Model):
        self.model=model
    def sigmoid_gradient(self,output:tf.Tensor,lastgradient:tf.Tensor,layer:SigmoidTRAIN):
        mu,by=layer.mu,layer.by
        #standrad output
        sigma=output/mu
        dLdmu=tf.reduce_sum(lastgradient*sigma)
        dydz=output*(mu-output)/mu
        dLdby=tf.reduce_sum(lastgradient*dydz)
        #gradient to inputx, this will be passed to next layer
        dLdx=lastgradient*dydz
        return dLdx,dLdmu,dLdby
    def dense_gradient(self,output:tf.Tensor,inputx:tf.Tensor,lastgradient:tf.Tensor,
                       actype:Literal['sigmoid','relu'],layer):
        # outputcop=FNOtrainer.ndimtranform(output,2,'col') if not ndimagust else output
        # outputcop=tf.reshape(output,(-1,3))
        if layer.activation is not None:
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
    def loss_gradient(self,output:tf.Tensor,target:tf.Tensor):
        # outputcop=tf.reshape(output,(-1,3))
        # targetcop=tf.reshape(target,(-1,3))
        if getattr(self,'lt') == 'binarycrossentropy':
            #this way we use the default value of from_logist:0
            #and it is the start of back propagation
            eps=1e-7
            output=tf.clip_by_value(output,eps,1-eps)
            self.dLdyhat=(output-target)/(output*(1-output))
        elif getattr(self,'lt') == 'mseloss':
            self.dLdyhat=output-target
        else:
            self.dLdyhat=output-target
        return self.dLdyhat
    def runall(self,y_hat:tf.Tensor,target:tf.Tensor,learning_rate,
               losstype:Literal['mseloss','binarycrossentropy','categorycrossentropy'],usemomentum=True):
        self.lt=losstype
        Layers=self.model.models
        layerparams=self.model.layerpacked
        if hasattr(self.model,'outlayer'):
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
                # print(losstart.shape,'最后一层对输入的梯度')
            else:
                if layerobj.name.startswith('sigmoid') or isinstance(layerobj,SigmoidTRAIN):#means it is a sigmoid layer
                    losstart,dLdmu,dLdby=self.sigmoid_gradient(output,losstart,layerobj)
                    if not usemomentum:
                        layerobj.mu.assign_sub(learning_rate*dLdmu)
                        layerobj.by.assign_sub(learning_rate*dLdby)
                        continue
                    self.momentumcore_sig(layerobj,dLdmu,dLdby,learning_rate)
                else:
                    losstart,dLdW,dLdb=self.dense_gradient(output,x_in,losstart,layer.activation,layer)
                    if not usemomentum:
                        layerobj.kernel.assign_sub(learning_rate*dLdW)
                        layerobj.bias.assign_sub(learning_rate*dLdb)
                        continue
                    self.momentumcore(layerobj,dLdW,dLdb,learning_rate)
    def momentumcore_sig(self,layer,dmu,dby,lr,eps=1e-8):
        '''
        create a way to implement sigmoid layer's momentum algorithm
        '''
        # print('////////////////')
        self.model.t_mb+=1
        # print(layer.s_beta1,layer.V_mu,layer.by,'bybyby')
        V_mu=layer.s_beta1*layer.V_mu+(1-layer.s_beta1)*dmu
        V_by=layer.s_beta1*layer.V_by+(1-layer.s_beta1)*dby
        B_mu=layer.s_beta2*layer.B_mu+(1-layer.s_beta2)*dmu**2
        B_by=layer.s_beta2*layer.B_by+(1-layer.s_beta2)*dby**2
        layer.V_mu.assign(V_mu)
        layer.V_by.assign(V_by)
        layer.B_mu.assign(B_mu)
        layer.B_by.assign(B_by)
        # considerate time span
        V_mu_hat=layer.V_mu/(1-tf.pow(layer.s_beta1,self.model.t_mb))
        V_by_hat=layer.V_by/(1-tf.pow(layer.s_beta1,self.model.t_mb))
        B_mu_hat=layer.B_mu/(1-tf.pow(layer.s_beta2,self.model.t_mb))
        B_by_hat=layer.B_by/(1-tf.pow(layer.s_beta2,self.model.t_mb))
        layer.mu.assign_sub(lr*V_mu_hat/(tf.sqrt(B_mu_hat)+eps))
        layer.by.assign_sub(lr*V_by_hat/(tf.sqrt(B_by_hat)+eps))
    def momentumcore(self,layer,dW,db,lr,eps=1e-8):
        self.model.t_wb+=1#time detect
        #this is weight update
        layer.V_w.assign(layer.beta1*layer.V_w+(1-layer.beta1)*dW)
        layer.B_w.assign(layer.beta2*layer.B_w+(1-layer.beta2)*tf.square(dW))
        V_hatw=layer.V_w/(1-layer.beta1**self.model.t_wb)
        B_hatw=layer.B_w/(1-layer.beta2**self.model.t_wb)
        layer.kernel.assign_sub(lr*V_hatw/(tf.sqrt(B_hatw)+eps))
        #bias update
        layer.V_b.assign(layer.beta1*layer.V_b+(1-layer.beta1)*db)
        layer.B_b.assign(layer.beta2*layer.B_b+(1-layer.beta2)*tf.square(db))
        V_hatb=layer.V_b/(1-layer.beta1**self.model.t_wb)
        B_hatb=layer.B_b/(1-layer.beta2**self.model.t_wb)
        layer.bias.assign_sub(lr*V_hatb/(tf.sqrt(B_hatb)+eps))
def manual_mse_loss(y_true, y_pred):
    error = y_true-y_pred # (batch_size, output_dim)
    squared_error = tf.square(error)  # (batch_size, output_dim)
    sample_loss = tf.reduce_sum(squared_error, axis=1)  # (batch_size,)
    loss = tf.reduce_mean(sample_loss)
    
    return loss
def Arunstar(inputs,model,epoches,batch,learning_rate,losstype='categorycrossentropy'):
    con=category(inputs,[('Species',),('Region','Latitude ','Longitude')])
    lab_feat=[]
    for ir in con:
        lab_feat.append(ir)
    label,features=lab_feat
    # print(label.shape,'label')
    opt=propagation(model)
    numsaples=inputs.shape[0]
    history={'loss':[],'accuracy':[]}
    for epoch in range(epoches):
        epoch_loss=[]
        epoch_correct=0
        for batchidx,batchtensor in enumerate(batchcontached([features,label],batchsize=batch)):
            x,y=batchtensor[0],batchtensor[1][:,1:]
            ypred=model(x)
            if losstype == 'mseloss':
                loss=manual_mse_loss(y,ypred)
            elif losstype == 'categorycrossentropy':
                loss=tf.keras.losses.categorical_crossentropy(y,ypred)
                loss=tf.reduce_mean(loss)
            batchloss=loss
            #use backward by hand
            opt.runall(ypred,y,learning_rate,losstype,usemomentum=model.momentum)
            epoch_loss.append(float(batchloss))
            #calculate accurancy
            predictions = tf.argmax(ypred, axis=1)
            true_labels = tf.argmax(y, axis=1)
            correct = tf.reduce_sum(tf.cast(predictions == true_labels, tf.float32))
            epoch_correct += int(correct)
        #save exited figures
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy = epoch_correct / numsaples
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{epoches} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}")
    print(y[:5])
    print(ypred[:5])
# con=category(nparray,[('Species',),('Region','Latitude ','Longitude')])
# lab_feat=[]
model=ForwardModel(outputunits=3,units=[(64,None),(32,None),(16,None)],usemomentum=True,sigmoidtrain=1,beta1=0.1,beta2=0.8)

history=Arunstar(nparray,model,5000,nparray.shape[0],0.0009,losstype='categorycrossentropy')