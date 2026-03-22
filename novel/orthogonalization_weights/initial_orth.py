import numpy as np
import tensorflow as tf
from typing import Literal,Type
from abc import abstractmethod
class preL:
    registry={}
    def __init_subclass__(cls,**kw):
        super().__init_subclass__(**kw)
        for name,method in cls.__dict__.items():
            if hasattr(method,'_register_name'):
                cls.registry[method._register_name]=method._register_func
def register(name):
    def decorator(func):
        def wrapper(*args,**kwargs):
            return func(*args,**kwargs)
        wrapper._register_name=name
        wrapper._register_func=func
        return wrapper
    return decorator
class Orthogonalize:
    def __init__(self,way=Literal['qr','svd','gram']):
        self.choose=way
        self.seting={'qr':self._qr,'svd':self._svd}
    def __enter__(self):
        return self.seting[self.choose]
    def _qr(self,W:tf.Tensor | np.ndarray):
        d_out,d_in=W.shape
        '''
        we use the inner method to orthogonalize w
        '''
        if d_out >= d_in:
            #column orth
            Q,_=tf.linalg.qr(W)# if tf.keras.backend.is_keras_tensor(W) else np.linalg.qr(W)
            return Q
        else:
            #row orth
            print(type(W),'typw')
            Q,_=tf.linalg.qr(tf.transpose(W))# if tf.keras.backend.is_keras_tensor(W) else np.linalg.qr(W.T)
            return tf.transpose(Q)
    def _svd(self,W:tf.Tensor):
        #column orth
        U,_,Vh=tf.linalg.svd(W,full_matrices=False)
        return tf.matmul(U, Vh, transpose_b=True)
    def _gram(self,W,eps=1e-8):
        d_out,d_in=W.shape
        WT=tf.transpose(W)
        Q=[]
        for i in range(WT.shape[0]):
            v=WT[i]
            for q in Q:
                v=v-tf.reduce_sum(v*q)*q
            norm=tf.norm(v)
            v=tf.cond(norm>eps,lambda :v/norm,lambda :v)
            Q.append(v)
        Q=tf.stack(Q)
        return tf.transpose(Q)
    def __exit__(self,x0,x1,x2):
        pass
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
#we create back propagation in this single file
class Propagation:
    def __init__(self,model,useriemann,useadam=True,**kw):
        self.model=model
        self.rieman,self.adaming=useriemann,useadam
        if useadam:
            self.beta1=kw.get('beta1',0.3)
            self.beta2=kw.get('beta2',0.9)
    def dense_gradient(self,lastgradient,inputx:tf.Tensor,output:tf.Tensor,layer):
        if layer.activate is not None:
            actype=layer.activate
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
    @abstractmethod
    def runall(self,y_hat:tf.Tensor,target:tf.Tensor,learning_rate,
               losstype:Literal['mseloss','binarycrossentropy','categorycrossentropy'],useadam=True,e=0):
        pairfunc=lambda alp:0.7+(1-0.7)*alp/3.14159
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
                # print(losstart.shape,'最后一层对输入的梯度')
            else:
                losstart,dLdW,dLdb=self.dense_gradient(losstart,x_in,output,layer)
                if not useadam:
                    #these parameters' updating need to be superseded by rieman way,before it we will complete a adam optimizer first
                    W_rie= self.rieman_update(layerobj.kernel,dLdW)
                    angles=self.angle(W_rie,dLdW)
                    alpha=pairfunc(angles)
                    W_all=(1-alpha)*W_rie+alpha*dLdW
                    layerobj.kernel.assign_sub(learning_rate*W_all)
                    layerobj.bias.assign_sub(learning_rate*dLdb)
                    continue
                dw,db=self.momentumcore(layerobj,dLdW,dLdb)
                W_rie= self.rieman_update(layerobj.kernel,dw)
                angles=self.angle(W_rie,dw)
                alpha=pairfunc(angles)
                if e%100 == 0:
                    print(f'angle={angles:.4f},alpha={alpha:.4f}')
                W_all=(1-alpha)*W_rie+alpha*dw
                layerobj.kernel.assign_sub(learning_rate*W_all)
                layerobj.bias.assign_sub(learning_rate*db)
    def momentumcore(self,layer,dW,db,eps=1e-8):
        self.model.t_wb+=1#time detect
        #this is weight update
        Wparams:Type=Adamregistry.get_adam(layer,layer.kernel)
        V_w,M_w=Wparams.V,Wparams.M
        V_w.assign(self.beta1*V_w+(1-self.beta1)*dW)
        M_w.assign(self.beta2*M_w+(1-self.beta2)*tf.square(dW))
        V_hatw=V_w/(1-self.beta1**self.model.t_wb)
        B_hatw=M_w/(1-self.beta2**self.model.t_wb)
        new_dW=V_hatw/(tf.sqrt(B_hatw)+eps)
        #bias update
        bparams=Adamregistry.get_adam(layer,layer.bias)
        V_b,M_b=bparams.V,bparams.M
        V_b.assign(self.beta1*V_b+(1-self.beta1)*db)
        M_b.assign(self.beta2*M_b+(1-self.beta2)*tf.square(db))
        V_hatb=V_b/(1-self.beta1**self.model.t_wb)
        B_hatb=M_b/(1-self.beta2**self.model.t_wb)
        new_db=V_hatb/(tf.sqrt(B_hatb)+eps)
        return new_dW,new_db
    def rieman_update(self,W,dW):
        WTG=tf.matmul(W,dW,transpose_a=True)
        sym=0.5*(WTG+tf.transpose(WTG))
        GR=dW-tf.matmul(W,sym)
        return GR
    def angle(self,GR,dW):
        gr_vec=tf.reshape(GR,[-1])
        dw_vec=tf.reshape(dW,[-1])
        gr_norm=tf.nn.l2_normalize(gr_vec,axis=0)
        dw_norm=tf.nn.l2_normalize(dw_vec,axis=0)
        cos=tf.reduce_sum(gr_norm*dw_norm)
        angle_rad=tf.acos(tf.clip_by_value(cos,-1.0,1.0))
        return angle_rad
    # def riemannian_update(self, W, dW, lr,e):

    #     A = tf.matmul(dW, W, transpose_b=True) - tf.matmul(W, dW, transpose_b=True)

    #     I = tf.eye(W.shape[0], dtype=W.dtype)

    #     M1 = I - 0.5 * lr * A
    #     M2 = I + 0.5 * lr * A

    #     W_new = tf.linalg.solve(M1, tf.matmul(M2, W))
    #     if e == 1000:
    #         if W.shape[0] >= W.shape[1]:
    #             # column orth
    #             orth_err = tf.norm(
    #                 tf.matmul(W, W, transpose_a=True) - tf.eye(W.shape[1])
    #             )
    #         else:
    #             # row orth
    #             orth_err = tf.norm(
    #                 tf.matmul(W,W, transpose_b=True) - tf.eye(W.shape[0])
    #             )
    #         print("orth error:", orth_err)
    #     return W_new
    # def riemannian_grad(self,W,dW):

    #     WT_G = tf.matmul(W, dW, transpose_a=True)

    #     sym = 0.5*(WT_G + tf.transpose(WT_G))

    #     GR = dW - tf.matmul(W, sym)

    #     return GR