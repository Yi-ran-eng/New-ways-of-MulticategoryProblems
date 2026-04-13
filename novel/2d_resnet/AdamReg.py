import torch
import torch.nn as nn
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
        return True
      
    @staticmethod
    def get_adam(layerobj,var):
        return Adamregistry.registries[id(layerobj)][id(var)]
def createAdam(layer_class,*args,**kwargs):
    getstatic=kwargs.pop('adam')
    layer=layer_class(*args,**kwargs)
    if getstatic:
        Adamregistry.register(layer)
    return layer
