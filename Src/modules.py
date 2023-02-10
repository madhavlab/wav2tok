import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional





logger = logging.getLogger(__name__)


class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))



try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
          @torch.jit.unused
          def forward(self, x):
              if not x.is_cuda:
                  return super().forward(x)
              else:
                 with torch.cuda.device(x.device):
                       return super().forward(x)
except ImportError:
    has_fused_layernorm = False

def LayerNorm(normalized_shape, eps = 1e-5, elementwise_affine = True, export = False):
     if torch.jit.is_scripting():
           export = True
     if not export and torch.cuda.is_available() and has_fused_layernorm:
           return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
     return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)





class Fp32GroupNorm(nn.GroupNorm):
     def __init__(self, *args , **kwargs):
          super().__init__(*args, **kwargs)

     def forward(self, input):
          output = F.group_norm( input.float(),
                     self.num_groups,
                     self.weight.float() if self.weight is not None else None,
                     self.bias.float() if self.bias is not None else None,
                     self.eps,
                    )
          return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)

    def forward(self, input):
         output = F.layer_norm( input.float(),
                   self.normalized_shape,
                   self.weight.float() if self.weight is not None else None,
                   self.bias.float() if self.bias is not None else None,
                   self.eps,
               )
         return output.type_as(input)


class GradMultiply(torch.autograd.Function):
       @staticmethod
       def forward(ctx, x, scale):
            ctx.scale = scale
            res = x.new(x)
            return res

       @staticmethod
       def backward(ctx, grad):
            return grad* ctx.scale, None



class SamePad(nn.Module):
   def __init__(self, kernel_size, causal = False):
         super().__init__()
         if causal:
             self.remove = kernel_size - 1
         else:
            self.remove = 1 if kernel_size % 2 == 0 else 0
   def forward(self, x):
       if self.remove > 0 :
            x = x[:, :, : -self.remove]
       return x


class TransposeLast(nn.Module):
      def __init__(self, deconstruct_idx = None):
          super().__init__()
          self.deconstruct_idx = deconstruct_idx
      def forward(self, x):
           if self.deconstruct_idx is not None:
                x = x[self.deconstruct_idx]
           return x.transpose(-2,-1)



def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)



