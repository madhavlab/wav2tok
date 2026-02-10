import torch
import torch.nn as nn
from typing import List

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Conv1DFeatureExtractor(nn.Module):
    def __init__(self, layers_config:List[List[int]], dropout:float=0.0, activation:str="gelu", bias:bool=False)-> None: 
        super().__init__()
        self.bias = bias
        self.model = nn.Sequential()
        self.dropout=dropout
        assert activation in {"gelu", "relu"}, f"{activation} activation specified. Excepts either relu or gelu"
        self.activation = activation
        
        in_channels= 1
        for i, config in enumerate(layers_config):
            out_channels, kernel_size, stride = config
            layer = self._make_layer(in_channels, out_channels, kernel_size, stride) 
            self.model.add_module("conv_feat_extractor_layer"+str(i+1), nn.Sequential(*layer))
            in_channels = out_channels

    def _make_layer(self, in_channels: int, out_channels:int, kernel_size: int, stride: int):
        if self.activation == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.ReLU()
        layer = [nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=self.bias),
                 nn.Dropout(self.dropout),
                 TransposeLast(),
                 nn.LayerNorm(out_channels),  
                 TransposeLast(),
                 activation] 
        return layer

    def forward(self, x): 
        return self.model(x).permute(0,2,1) 

if __name__ == "__main__":

    layers_config = [[256,10,5], [256,7,4], [256,3,2], [512,3,2], [512,2,2] , [512,2,1]]
    nn = Conv1DFeatureExtractor(layers_config, bias=True)
    x = nn(torch.rand(10, 1, 16400))
    print(x.shape)
