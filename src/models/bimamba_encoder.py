import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


from mamba_ssm import Mamba2
@dataclass
class ModelArgs:
    d_model:int # Model dimension d_model
    d_project:int
    d_state:int  # SSM state expansion factor, typically 64 or 128
    d_conv:int    # Local convolution width
    expand:int    # Block expansion factor
    n_layer:int
    block_type:str
    headdim:int


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    
class ResidualBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
           
        self.mixer_forward = Mamba2(d_model=args.d_model,
                            d_state=args.d_state,
                            d_conv=args.d_conv,
                            expand=args.expand,
                            headdim=args.headdim
                            )
        self.mixer_backward = Mamba2(d_model=args.d_model,
                            d_state=args.d_state,
                            d_conv=args.d_conv,
                            expand=args.expand,
                            headdim=args.headdim
                            )
        self.norm1_forward = RMSNorm(args.d_model)
        self.norm2_forward = RMSNorm(args.d_model)
        self.norm1_backward = RMSNorm(args.d_model)
        self.norm2_backward = RMSNorm(args.d_model)
        self.feedforward = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_model, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output_forward = self.norm2_forward(self.mixer_forward(self.norm1_forward(x)) + x)
        output_backward = self.norm2_backward(torch.flip(self.mixer_backward(self.norm1_backward(torch.flip(x, dims=[1]))),dims=[1]) + x)
        output = output_backward+output_forward
        output = self.feedforward(output.permute(0,2,1)).permute(0,2,1) + output
        return output

class BiMambaEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.project = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_project, kernel_size=1, stride=1)


    def forward(self, x):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        for layer in self.layers:
            x = layer(x)
            
        x = F.normalize(self.project(x.permute(0,2,1)).permute(0,2,1), dim=-1)
        return x

if __name__ == "__main__":
    args =ModelArgs(
            d_model=96,
            d_project=512,
            d_state=128,
            d_conv=4,
            expand=4,
            n_layer=8,
            headdim=48,
            block_type="mamba2",
            )

    model = BiMambaEncoder(args).cuda()
    y = model(torch.rand(2,51,96).cuda())
    print(y.shape, torch.linalg.norm(y,dim=-1), torch.linalg.norm(y,dim=-1).shape)

