import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from AdamReg import Adamregistry

class Propagation:
    def __init__(self, model: nn.Module, beta1=0.9, beta2=0.99):
        self.m = model
        self.beta1, self.beta2 = beta1, beta2
        self.lt = None  # loss type attribute

    def dense_gradient(self, lastgradient, inputx: torch.Tensor, output: torch.Tensor, layer):
        lastgradient=lastgradient.squeeze()
        # print(lastgradient.shape,'lastshap')
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
        # print(dLdz.shape,'last')
        W=layer.weight
        b = layer.bias
        # print(layer.name,'------------========')
        # print(dLdz.shape,'dldz',W.T.shape,'WTsha')
        dLdx = torch.matmul(dLdz,W)
        # print(dLdx.shape,'dldzshape')
        dLdW = torch.matmul(dLdz.T,inputx)
        # dLdW=torch.matmul(dLdz.T,inputx)
        # print(dLdW.shape,'dldw')
        dLdb = torch.sum(dLdz, dim=0)
        
        return dLdx, dLdW, dLdb
    
    def loss_gradient(self, output: torch.Tensor, target: torch.Tensor, e):
        if getattr(self, 'lt') == 'binarycrossentropy':
            eps = 1e-7
            output = torch.clamp(output, eps, 1 - eps)
            self.dLdyhat = (output - target) / (output * (1 - output))
        elif getattr(self, 'lt') == 'mseloss':
            self.dLdyhat = output - target
        elif getattr(self, 'lt') == 'crossentropy':
            # output is logits (pre-softmax)
            # print('>?')
            probs = F.softmax(output, dim=-1)
            # print(probs.shape,'probs')
            num_classes = output.shape[-1]
            # print(num_classes,'clas')
            target_onehot = F.one_hot(target.long(), num_classes=num_classes).float()
            # print(target_onehot.shape,'onehups')
            self.dLdyhat = probs - target_onehot.squeeze()
        else:
            self.dLdyhat = output - target
        return self.dLdyhat
    def MAXpool_gradient(self, lastgradient, layerobj):
        _, inputdata, layer = self.m.Layer_cache[layerobj.name]
        # inputdata shape: (batch, channels, height, width)
        # lastgradient shape: (batch, channels, out_h, out_w) 或 (batch, channels)
        # print(layerobj.name,lastgradient.shape,'pool-shape')
        # GlobalMaxPooling2D (output shape: batch, channels)
        if lastgradient.dim() == 2:
            print('jinle?')
            # lastgradient: (batch, channels)
            max_vals, max_indices = torch.max(inputdata.view(inputdata.shape[0], inputdata.shape[1], -1), dim=2, keepdim=True)  # (batch, channels, 1)
            max_indices_h = max_indices // inputdata.shape[3]
            max_indices_w = max_indices % inputdata.shape[3]
            
            # build mask (batch, channels, height, width)
            mask = torch.zeros_like(inputdata)
            for b in range(inputdata.shape[0]):
                for c in range(inputdata.shape[1]):
                    mask[b, c, max_indices_h[b, c, 0], max_indices_w[b, c, 0]] = 1.0
            
            # the greatest value obtains gradient
            dx = mask * lastgradient[:, :, None, None]  # (batch, channels, height, width)
            
        # regular MaxPool2D (shape: batch, channels, out_h, out_w)
        else:
            batch, channels, in_h, in_w = inputdata.shape
            _, _, out_h, out_w = lastgradient.shape
            
            dx = torch.zeros_like(inputdata)
            
            for i in range(batch):
                for c in range(channels):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            # if layerobj.name == 'pool1':
                                # print('*')
                            h_start = int(torch.floor(torch.tensor(oh * in_h / out_h)).item())
                            h_end = int(torch.ceil(torch.tensor((oh + 1) * in_h / out_h)).item())
                            w_start = int(torch.floor(torch.tensor(ow * in_w / out_w)).item())
                            w_end = int(torch.ceil(torch.tensor((ow + 1) * in_w / out_w)).item())
                            
                            h_end = min(h_end, in_h)
                            w_end = min(w_end, in_w)
                            
                            window = inputdata[i, c, h_start:h_end, w_start:w_end]
                            max_val = torch.max(window)
                            
                            found = False
                            # if layerobj.name == 'pool2':
                                # print(h_end,h_start)
                                # print(w_end,w_start,"?")
                            for kh in range(h_end - h_start):
                                for kw in range(w_end - w_start):
                                    if window[kh, kw] == max_val:
                                        dx[i, c, h_start + kh, w_start + kw] += lastgradient[i, c, oh, ow]
                                        found = True
                                        break
                                if found:
                                    break
            
        return dx

    def AVGpool_gradient(self, lastgradient, layerobj):
        _, inputdata, layer = self.m.layers_cache[layerobj.name]
        
        # Check if it's GlobalAveragePooling2D by class name or attribute
        is_global = (hasattr(layerobj, 'global_pooling') and layerobj.global_pooling) or \
                    (hasattr(layerobj, '__class__') and 'GlobalAveragePooling2D' in layerobj.__class__.__name__)
        
        if is_global:
            # For GlobalAveragePooling2D, the gradient is simply the mean of the gradients across the spatial axes
            # lastgradient: (batch, channels)
            # inputdata: (batch, channels, height, width)
            dx = lastgradient[:, :, None, None].expand_as(inputdata)
            dx = dx / (inputdata.shape[2] * inputdata.shape[3])
            return dx
        
        # For regular AvgPool2D
        pool_size_h = layer.pool_size if hasattr(layer, 'pool_size') else layer.kernel_size
        pool_size_w = pool_size_h if not isinstance(pool_size_h, (list, tuple)) else pool_size_h[1]
        pool_size_h = pool_size_h if not isinstance(pool_size_h, (list, tuple)) else pool_size_h[0]
        
        stride = layer.stride if hasattr(layer, 'stride') else layer.strides
        stride_h = stride if not isinstance(stride, (list, tuple)) else stride[0]
        stride_w = stride if not isinstance(stride, (list, tuple)) else stride[1]
        
        batch, channels, height, width = inputdata.shape
        
        dx = torch.zeros_like(inputdata)
        
        out_h = (height - pool_size_h) // stride_h + 1
        out_w = (width - pool_size_w) // stride_w + 1
        
        for i in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * stride_h
                        w_start = ow * stride_w
                        h_end = min(h_start + pool_size_h, height)
                        w_end = min(w_start + pool_size_w, width)
                        
                        actual_pool_size = (h_end - h_start) * (w_end - w_start)
                        dx[i, c, h_start:h_end, w_start:w_end] += lastgradient[i, c, oh, ow] / actual_pool_size
        
        return dx
    def conv_gradient(self, lastgradient, layerobj):
        ''' batch x channel x height x width 2D convolution'''
        _, inputs, layer = self.m.Layer_cache[layerobj.name]
        
        batch, in_channels, height, width = inputs.shape
        out_channels, _, kernel_h, kernel_w = layer.weight.shape
        
        def parse_param(p):
            return p if isinstance(p, int) else p[0]
        # print(layer.padding,'layer_pading')
        stride_h = parse_param(layer.stride)
        stride_w = parse_param(layer.stride) if isinstance(layer.stride, int) else layer.stride[1]
        padding = layer.padding if hasattr(layer, 'padding') else 0
        dilation_h = parse_param(layer.dilation)
        dilation_w = parse_param(layer.dilation) if isinstance(layer.dilation, (list, tuple)) else layer.dilation
        # print(padding,'padding>>>')
        _, _, out_h, out_w = lastgradient.shape
    
        if isinstance(padding, str):
            if padding == 'same':
                # 计算需要的padding
                pad_h = max((out_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - height, 0)
                pad_w = max((out_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - width, 0)
                pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
            else:  # valid
                pad_top = pad_bottom = pad_left = pad_right = 0
        else:
            pad_top = pad_bottom = pad_left = pad_right = padding[0]#need to be modified
        # print(pad_left,pad_right,'pading')
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            input_padded = F.pad(inputs, (pad_left, pad_right, pad_top, pad_bottom))
        else:
            input_padded = inputs
        
        padded_h, padded_w = input_padded.shape[2:]
        db = torch.sum(lastgradient, dim=[0, 2, 3])  # (out_channels,)
        # unfold: (batch, in_channels * kh * kw, out_h * out_w)
        windows = F.unfold(input_padded, (kernel_h, kernel_w), 
                        stride=(stride_h, stride_w),
                        dilation=(dilation_h, dilation_w))
        #reshape: (batch, out_h, out_w, in_channels, kernel_h, kernel_w)
        windows = windows.view(batch, in_channels, kernel_h, kernel_w, out_h, out_w)
        windows = windows.permute(0, 4, 5, 1, 2, 3)
        
        # einsum: (batch, out_h, out_w, in_channels, kh, kw) × (batch, out_h, out_w, out_channels)
        lastg = lastgradient.permute(0, 2, 3, 1)  # (batch, out_h, out_w, out_channels)
        dw = torch.einsum('bhwick,bhwo->icko', windows, lastg)
        dw = dw.permute(3, 0, 1, 2)  # (out_channels, in_channels, kernel_h, kernel_w)
        
        dx_padded = F.conv_transpose2d(
            lastgradient,
            layer.weight,
            stride=(stride_h, stride_w),
            padding=0, 
            dilation=(dilation_h, dilation_w)
        )
        
        # 裁剪padding
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            dx = dx_padded[:, :, pad_top:pad_top+height, pad_left:pad_left+width]
        else:
            dx = dx_padded
        
        return dx, dw, db
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
               losstype: Literal['mseloss', 'binarycrossentropy', 'crossentropy'], 
               useadam=True, e=0):
        
        # Increment time step
        if not hasattr(self.m, 't_wb'):
            self.m.t_wb = 0
        self.m.t_wb += 1
        
        self.lt = losstype
        layerparams: dict = self.m.Layer_cache
        all_layers = [x[-1] for x in layerparams.values()]
        '''
        all of these layers are a subclass of AtributeLayer,which contains an attribute:manual_back
        we use this attribute to decide to calculate the gradients
        '''
        meduim=list(reversed(all_layers))
        # print(y_hat.shape,target.shape,'yhat,target')
        losstart = self.loss_gradient(y_hat, target, e)
        if e == 1:
            print(losstart.shape,'losstartshape when e=1')
        for idx, layer in enumerate(reversed(all_layers)):
            output, x_in, layerobj = layerparams[layer.name]
            layer_type = layerobj.__class__.__name__
            #this part inplements autograd works
            if hasattr(layerobj,'manual_back') and not layerobj.manual_back:
                k=idx
                current_layer=layerobj
                autolayers=[]
                while k < len(meduim) and not current_layer.manual_back:
                    # print(current_layer.name,'currentname')
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
                        if losstart.shape != output.shape:
                            print('rel')
                            losstart=losstart.reshape(*output.shape)
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
                    print(losstart.shape,'nami')
                    dLdz = losstart
                    # print(dLdz,'dldz')
                    dLdW = torch.matmul(dLdz.T,x_in)
                    dLdb = torch.sum(dLdz, dim=0)
                    losstart = torch.matmul(dLdz,layerobj.weight)
                else:
                    
                    # Handle Normallayer shape conversion
                    if hasattr(layerobj, '__class__') and 'Linear' in layerobj.__class__.__name__ and len(losstart.shape) == 4:
                        losstart = losstart.squeeze(1)
                    if ('Conv2d' in layer_type or 'Pool' in layerobj.__class__.__name__):
                        if len(losstart.shape) == 2:
                            # print('has entered',layerobj.name)
                            targettype=output.shape
                            losstart=losstart.view(targettype)

                    # Handle different layer types
                    
                    if 'AdaptiveAvgPool1d' in layer_type:
                        # print(layer_type,"////????")
                        losstart=self.AVGpool_gradient(losstart, layerobj)
                    elif 'AdaptiveMaxPool2d' in layer_type:
                        losstart=self.MAXpool_gradient(losstart, layerobj)
                    elif 'Conv2d' in layer_type:
                        losstart, dLdW, dLdb = self.conv_gradient(losstart, layerobj)
                    else:
                        losstart, dLdW, dLdb = self.dense_gradient(losstart, x_in, output, layer)
                with torch.no_grad():
                    if not useadam and not ('AdaptiveMaxPool2d' in layer_type):
                        # these parameters' updating need to be superseded by rieman way, before it we will complete a adam optimizer first
                        # print(layerobj.name,'layername')
                        # print(layerobj.weight.shape,dLdW.shape,'>')
                        layerobj.weight -= learning_rate * dLdW
                        if hasattr(layerobj, 'bias') and layerobj.bias is not None:
                            layerobj.bias -= learning_rate * dLdb
                        continue
                
                    if not 'AdaptiveMaxPool2d' in layer_type:
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