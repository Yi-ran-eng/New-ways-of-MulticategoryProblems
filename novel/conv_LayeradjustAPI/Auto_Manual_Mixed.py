import torch
from torch import nn
import torch.nn.functional as F
from typing import Type,Literal
import numpy as np
import inspect
class AdamParameters:
    def __init__(self, var):
        self.V = torch.zeros_like(var, requires_grad=False)
        self.M = torch.zeros_like(var, requires_grad=False)
class Adamregistry:
    registries={}
    @staticmethod
    def register(layerobj:nn.Module):
        '''
        one layer always has several parameters,we need to match a adamparam for each parameter in this layer
        and we use a dict to contain such a adamparam-originalparam pairs
        '''
        state={}
        for param in layerobj.parameters():
            #use ref() as key because common param cannot be key
            state[id(param)]=AdamParameters(param)
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
    Wrapper.__name__ = cls.__name__
    return Wrapper
class Propagation:
    def __init__(self, model: nn.Module, beta1=0.9, beta2=0.99):
        self.m = model
        self.beta1, self.beta2 = beta1, beta2
        self.lt = None  # loss type attribute
        
    def dense_gradient(self, lastgradient, inputx: torch.Tensor, output: torch.Tensor, layer):
        lastgradient=lastgradient.squeeze()
        print(lastgradient.shape,'lastshap')
        if hasattr(layer, 'activation') and layer.activation is not None:
            actype = layer.activation
            if actype == 'sigmoid':
                dactivate = output * (1 - output)
            elif actype == 'relu':
                dactivate = (output > 0).float()
            elif actype == '':
                dactivate=1
            else:
                # Default to sigmoid-like behavior if unknown
                dactivate = output * (1 - output) if actype == 'sigmoid' else (output > 0).float()
            dLdz = lastgradient * dactivate
        else:
            dLdz = lastgradient
        print(dLdz.shape,'last')
        W=layer.weight
        b = layer.bias
        print(layer.name,'------------========')
        print(dLdz.shape,'dldz',W.T.shape,'WTsha')
        dLdx = torch.matmul(dLdz, W.T)
        print(dLdx.shape,'dldzshape')
        dLdW = torch.matmul(inputx.T,dLdz)
        # dLdW=torch.matmul(dLdz.T,inputx)
        print(dLdW.shape,'dldw')
        dLdb = torch.sum(dLdz, dim=0)
        
        return dLdx, dLdW, dLdb
    
    def loss_gradient(self, output: torch.Tensor, target: torch.Tensor, e):
        if getattr(self, 'lt') == 'binarycrossentropy':
            eps = 1e-7
            output = torch.clamp(output, eps, 1 - eps)
            self.dLdyhat = (output - target) / (output * (1 - output))
        elif getattr(self, 'lt') == 'mseloss':
            self.dLdyhat = output - target
        elif getattr(self, 'lt') == 'sparse_categorical_crossentropy':
            num_classes = output.shape[-1]
            target_onehot = F.one_hot(target.long(), num_classes=num_classes).float()
            self.dLdyhat = output - target_onehot
        else:
            self.dLdyhat = output - target
        # print(self.dLdyhat[:10,:10],'dldyhat')
        return self.dLdyhat
    def MAXpool_gradient(self, lastgradient, layerobj):
        _, inputdata, layer = self.m.layers_cache[layerobj.name]
        # inputdata shape: (batch, channels, timespan)
        # lastgradient shape: (batch, channels, timespan_out) 或 (batch, channels)
        
        # (output shape: batch, channels)
        if lastgradient.dim() == 2:
            # lastgradient: (batch, channels)
            max_vals, max_indices = torch.max(inputdata, dim=2, keepdim=True)  # (batch, channels, 1)
            
            #build mask (batch, channels, timespan)
            mask = torch.zeros_like(inputdata)
            mask.scatter_(2, max_indices, 1.0)
            
            # the greatest value obtains gradient
            dx = mask * lastgradient.unsqueeze(2)  # (batch, channels, timespan)
            
        # avg (shape: batch, channels, timespan_out)
        else:
            # lastgradient: (batch, channels, timespan_out)
            kernel_size = layer.kernel_size if hasattr(layer, 'kernel_size') else layer.pool_size
            stride = layer.stride
            batch, channels, timespan = inputdata.shape
            _, timespan_out = lastgradient.shape[2], _
            
            unfolded = inputdata.unfold(2, kernel_size, stride)  # (batch, channels, windows, kernel_size)
            max_vals, max_indices = torch.max(unfolded, dim=3, keepdim=True)  # (batch, channels, windows, 1)
            dx = torch.zeros_like(inputdata)
            for i in range(batch):
                for c in range(channels):
                    for w in range(timespan_out):
                        pos = w * stride + max_indices[i, c, w, 0]
                        dx[i, c, pos] += lastgradient[i, c, w]
            
        return dx
    
    def AVGpool_gradient(self, lastgradient, layerobj):
        _, inputdata, layer = self.m.layers_cache[layerobj.name]
        
        # Check if it's GlobalAveragePooling1D by class name or attribute
        is_global = (hasattr(layerobj, 'global_pooling') and layerobj.global_pooling) or \
                    (hasattr(layerobj, '__class__') and 'GlobalAveragePooling1D' in layerobj.__class__.__name__)
        
        if is_global:
            # For GlobalAveragePooling1D, the gradient is simply the mean of the gradients across the time axis
            dx = lastgradient[:, None, :].expand_as(inputdata)
            dx = dx / inputdata.shape[1]
            return dx
        
        # For regular AvgPool1D
        pool_size = layer.pool_size if hasattr(layer, 'pool_size') else layer.kernel_size
        stride = layer.stride if hasattr(layer, 'stride') else layer.strides
        
        dx = torch.zeros_like(inputdata)
        
        for i in range(0, inputdata.shape[1] - pool_size + 1, stride):
            window = inputdata[:, i:i+pool_size, :]
            avg = torch.mean(window, dim=1, keepdim=True)
            dx[:, i:i+pool_size, :] += lastgradient[:, i//stride, None, :] / pool_size
        return dx
    def conv_gradient(self, lastgradient, layerobj):
        '''use einsum to vectorize the code, make it more simplistic'''
        _, inputs, layer = self.m.layers_cache[layerobj.name]
        batch, channels, timespan = inputs.shape
        out_channels, in_channels, kernelsize = layer.weight.shape
        # Get conv layer attributes
        stride = layer.stride if hasattr(layer, 'stride') else layer.strides
        padding = layer.padding if hasattr(layer, 'padding') else 'valid'
        
        _, out_channels, outputtime = lastgradient.shape
        
        if padding == 'same':
            output_time = (timespan + stride - 1) // stride
            pad_total = max((output_time - 1) * stride + kernelsize - timespan, 0)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            #padding at time dimension,dimension oder (batch, channels, timespan)
            input_padded = F.pad(inputs, (pad_left, pad_right)) 
            dx_padded = torch.zeros_like(input_padded)
            padded_len = timespan + pad_left + pad_right
        else:
            input_padded = inputs
            dx_padded = torch.zeros_like(inputs)
            pad_left = 0
            padded_len = timespan
        db = torch.sum(lastgradient, dim=[0, 2])  # (out_channels,)
        
        starts = torch.arange(outputtime) * stride
        window_indices = starts[:, None] + torch.arange(kernelsize)  # (outputtime, kernelsize)
        window_indices = torch.clamp(window_indices, 0, padded_len - 1)
        # input_padded: (batch, channels, padded_len)
        # window_indices: (outputtime, kernelsize)
        windows = input_padded[:, :, window_indices]  # (batch, channels, outputtime, kernelsize)
        # windows: (batch, channels, outputtime, kernelsize) -> (batch, outputtime, kernelsize, channels)
        windows = windows.permute(0, 2, 3, 1)  # (batch, outputtime, kernelsize, channels)
        # lastgradient: (batch, out_channels, outputtime) -> (batch, outputtime, out_channels)
        lastgradient_reshaped = lastgradient.permute(0, 2, 1)  # (batch, outputtime, out_channels)
        # windows: (batch, outputtime, kernelsize, channels)
        # lastgradient: (batch, outputtime, out_channels)
        # dw: (kernelsize, channels, out_channels)
        dw = torch.einsum('btkc,btq->kcq', windows, lastgradient_reshaped)
        for i in range(outputtime):
            start = i * stride
            t_indices = range(start, start + kernelsize)
            grad = lastgradient[:, :, i]  # (batch, out_channels)
            contribution = torch.einsum('ock,bo->bkc', layer.weight, grad)
            for j, t_idx in enumerate(t_indices):
                if t_idx < padded_len:
                    dx_padded[:, :, t_idx] += contribution[:, j, :]  # contribution[:, j, :] is (batch, channels)
        
        if padding == 'same':
            dx = dx_padded[:, :, pad_left:pad_left+timespan]  # clips gradients
        else:
            dx = dx_padded
        
        return dx, dw.permute(2,1,0), db
    
    def momentumcore(self, layer, dW, db, eps=1e-6):
        # this is weight update
        Wparams = Adamregistry.get_adam(layer, layer.weight)
        V_w, M_w = Wparams.V, Wparams.M
        V_w.data = self.beta1 * V_w + (1 - self.beta1) * dW
        M_w.data = self.beta2 * M_w + (1 - self.beta2) * torch.square(dW)
        V_hatw = V_w / (1 - self.beta1 ** self.m.t_wb)
        B_hatw = M_w / (1 - self.beta2 ** self.m.t_wb)
        new_dW = V_hatw / (torch.sqrt(B_hatw) + eps)
        
        # bias update
        bparams = Adamregistry.get_adam(layer, layer.bias)
        V_b, M_b = bparams.V, bparams.M
        V_b.data = self.beta1 * V_b + (1 - self.beta1) * db
        M_b.data = self.beta2 * M_b + (1 - self.beta2) * torch.square(db)
        V_hatb = V_b / (1 - self.beta1 ** self.m.t_wb)
        B_hatb = M_b / (1 - self.beta2 ** self.m.t_wb)
        new_db = V_hatb / (torch.sqrt(B_hatb) + eps)
        
        return new_dW, new_db
    
    def runall(self, y_hat: torch.Tensor, target: torch.Tensor, learning_rate,
               losstype: Literal['mseloss', 'binarycrossentropy', 'sparse_categorical_crossentropy'], 
               useadam=True, e=0):
        
        # Increment time step
        if not hasattr(self.m, 't_wb'):
            self.m.t_wb = 0
        self.m.t_wb += 1
        
        self.lt = losstype
        layerparams: dict = self.m.layers_cache
        all_layers = [x[-1] for x in layerparams.values()]
        '''
        all of these layers are a subclass of AtributeLayer,which contains an attribute:manual_back
        we use this attribute to decide to calculate the gradients
        '''
        meduim=list(reversed(all_layers))
        losstart = self.loss_gradient(y_hat, target, e)
        for idx, layer in enumerate(reversed(all_layers)):
            output, x_in, layerobj = layerparams[layer.name]
            layer_type = layerobj.__class__.__name__
            #this part inplements autograd works
            if hasattr(layerobj,'manual_back') and not layerobj.manual_back:
                k=idx
                current_layer=layerobj
                autolayers=[]
                while k < len(meduim) and not current_layer.manual_back:
                    print(current_layer.name,'currentname')
                    autolayers.append(meduim[k])
                    k+=1
                    try:
                        current_layer=meduim[k]
                    except IndexError:
                        pass
                else:
                    if current_layer.manual_back:
                        assert output.requires_grad,f"layer{layerobj.name}'s output should be allowed grad"
                        output=output.squeeze()
                        grads=torch.autograd.grad(output,[x_in]+list(layerobj.parameters()),
                                                grad_outputs=losstart,retain_graph=True,allow_unused=True)
                        losstart=grads[0]
                        params_grads=grads[1:]
                        #update by hand
                        with torch.no_grad():
                            for p,g in zip(layerobj.parameters(),params_grads):
                                if g is not None:
                                    p-=learning_rate*g
                        continue
                self.dependProp(autolayers,layerparams,losstart)
                break
            else: 
                # distinguish the last layer
                if layer.name == self.m.outputlayer.name:
                    dLdz = losstart
                    dLdW = torch.matmul(x_in.T,dLdz)
                    dLdb = torch.sum(dLdz, dim=0)
                    losstart = torch.matmul(dLdz, layerobj.weight.T)
                else:
                    
                    # Handle Normallayer shape conversion
                    if hasattr(layerobj, '__class__') and 'Normallayer' in layerobj.__class__.__name__ and len(losstart.shape) == 3:
                        losstart = losstart.squeeze(1)
                    
                    # Handle different layer types
                    
                    if 'AdaptiveAvgPool1d' in layer_type:
                        losstart=self.AVGpool_gradient(losstart, layerobj)
                    elif 'AdaptiveMaxPool1d' in layer_type:
                        losstart=self.MAXpool_gradient(losstart, layerobj)
                    elif 'CustconvolutionLayer' in layer_type:
                        losstart, dLdW, dLdb = self.conv_gradient(losstart, layerobj)
                    else:
                        losstart, dLdW, dLdb = self.dense_gradient(losstart, x_in, output, layer)
                with torch.no_grad():
                    if not useadam and not ('AdaptiveAvgPool1d' in layer_type or 'AdaptiveMaxPool1d' in layer_type):
                        # these parameters' updating need to be superseded by rieman way, before it we will complete a adam optimizer first
                        print(layerobj.name,'layername')
                        print(layerobj.weight.shape,dLdW.shape,'>')
                        layerobj.weight -= learning_rate * dLdW
                        if hasattr(layerobj, 'bias') and layerobj.bias is not None:
                            layerobj.bias -= learning_rate * dLdb
                        continue
                
                    if not ('AdaptiveAvgPool1d' in layer_type or 'AdaptiveMaxPool1d' in layer_type):
                        dw, db = self.momentumcore(layerobj, dLdW, dLdb)
                        layerobj.weight -= learning_rate * dw
                        if hasattr(layerobj, 'bias') and layerobj.bias is not None:
                            layerobj.bias -= learning_rate * db
    def dependProp(self,autolayers,layerparams,lastgradient):
        _,x_first,_=layerparams[autolayers[-1].name]# the first autolayer
        if x_first.grad_fn is not None:
            x_in_detached=x_first.detach().requires_grad(True)
            temp_out=x_in_detached
            for layer in reversed(autolayers):# do forward again
                temp_out=layer(temp_out)
        else:
            temp_out=layerparams[autolayers[0].name][0]# the last autolayer in forward
        assert temp_out.requires_grad,f'layer{autolayers[0].name} output should be allowed grad'
        temp_out=temp_out.squeeze()
        lastgradient=lastgradient.squeeze()
        torch.autograd.backward(temp_out,grad_tensors=lastgradient,retain_graph=True)
        for layer in autolayers:
            if hasattr(layer,'opt') and layer.opt is not None:
                layer.opt.step()
                layer.opt.zero_grad()

        if x_first.grad_fn is not None:
            losstart=x_in_detached.grad
        else:
            losstart=x_first.grad
        return losstart
class AtributeLayer(nn.Module):
    def __init__(self,*args,**kwrags):
        super().__init__(**kwrags)
        self.manual_back=False
        self._save_methods()
    @property
    def _manual_back(self):
        return self.manual_back
    @_manual_back.setter
    def _manual_back(self,newstates):
        self.manual_back=newstates
        return self.manual_back
    def _save_methods(self):
        methods = [method for method,value in inspect.getmembers(self) 
                   if callable(value) and not method.startswith('__')]
        if not hasattr(self, '_saved_methods'):
            self._saved_methods = []
        self._saved_methods.extend(methods)
    def _get_customizedLayers(self):
        return self._saved_methods
