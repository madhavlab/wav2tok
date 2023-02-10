

import math
#from dataclasses import dataclass, field
#from typing import List, Optional

#import torch
#import torch.nn.functional as F

#import utils
import metrics


#from dataclass import Dataclass
from meters import safe_round
from torch.nn.modules.loss import _Loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from wav2vec_criterion import Wav2VecCriterionConfig, Wav2vecCriterion
import math
from dataclasses import dataclass , field
from typing import List, Tuple, Optional
from data_utils import compute_mask_indices
from dataclass import ChoiceEnum, Dataclass
from modules import (
   Fp32GroupNorm,
   Fp32LayerNorm,
   GradMultiply,

   LayerNorm,

   SamePad,
   TransposeLast,
)

from utils import buffered_arange
import utils
#from gumble_vector_quantizer import GumbelVectorQuantizer
#from gumble_note_quantizer import GumbelNoteQuantizer
from multihead_attention import MultiheadAttention


#from function_library import WAV_DATASET, Wav_Dict_GEN, DATALOADER, load_weights, load

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])

class Emb(nn.Module):
       def __init__(
        self,
        input_dim ,
        emb_dim,
        num_layers ,

             ):

           super().__init__()

           self.input_dim = input_dim
           self.emb_dim = emb_dim
           self.lstm1 = nn.LSTM(input_dim,emb_dim //2,num_layers, dropout = 0.25, \
                             bidirectional = True,batch_first = True)

           #self.lstm2 = nn.LSTM(emb_dim,emb_dim //2,2, dropout = 0.25, \
            #                 bidirectional = True,batch_first = True)





       def forward(self, x1 ):

            x, _ = self.lstm1(x1)

            #x, _ = self.lstm2(x)
            #x = F.normalize(x, p=2, dim = -1)
            return x

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



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_dim =768):
        super().__init__()

        self.dropout = 0.1 #  args.dropout
        self.embedding_dim = encoder_dim#args.encoder_embed_dim
        self.conv_pos = 128
        self.conv_pos_groups = 16
        self.encoder_ffn_embed_dim = 3072
        self.encoder_attention_heads = 8
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.activation_fn = "gelu"
        self.layer_norm_first = False
        self.encoder_layers = 2
        self.encoder_layerdrop = 0.05

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=self.conv_pos,
            padding=self.conv_pos // 2,
            groups=self.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (self.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(self.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=self.encoder_ffn_embed_dim,
                    num_attention_heads=self.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation_dropout=self.activation_dropout,
                    activation_fn=self.activation_fn,
                    layer_norm_first=self.layer_norm_first,
                )
                for _ in range(self.encoder_layers)
            ]
        )

        self.layer_norm_first = self.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = self.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict



class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn



class Wav2Vec2Model(nn.Module):
     def __init__(self, input_dim , encoder_dim , number_of_groups, number_of_codewords,  ):
         super(Wav2Vec2Model,self).__init__()



         self.input_dim = input_dim
         self.mask_prob = 0.65 #cfg.mask_prob #mask_prob = 0.65
         self.mask_selection ='static' # cfg.mask_selection
         # mask_selection = 'static' <- mask distribution 
         self.embed = encoder_dim
         self.mask_other = 0 # cfg.mask_other # mask_other = 0
         self.mask_length = 10 #cfg.mask_length # mask_length = 10
         self.no_mask_overlap = False#cfg.no_mask_overlap # no_mask_overlap = False
         self.mask_min_space = 1#cfg.mask_min_space # mask_min_space = 1
         self.mask_channel_prob =0.0# cfg.mask_channel_prob # mask_channel_prob = 0.0
         self.mask_channel_selection  = 'static'#cfg.mask_channel_selection
         # mask_channel_selection = 'static'

         self.mask_channel_other = 0 #cfg.mask_channel_other # mask_channel_other = 0
         self.mask_channel_length = 10 #cfg.mask_channel_length # mask_channel_length = 10
         self.no_mask_channel_overlap = False #cfg.no_mask_channel_overlap
         # no_mask_channel_overlap = False

         self.mask_channel_min_space =1 # cfg.mask_channel_min_space
         # mask_channel_min_space = 1

         self.dropout_input = nn.Dropout(p = 0.1) # dropout_input = 0
         self.dropout_features = nn.Dropout(p = 0.1) # dropout_features = 0

         self.feature_grad_mult = 1#cfg.feature_grad_mult # feature_grad_mult = 1
         self.latent_temp = (0.5, 0.5, 0.9999)
         self.num_toks = number_of_codewords
         self.groups = number_of_groups

         self.quantizer =  GumbelVectorQuantizer(
                            dim = self.embed,
                            num_vars = self.num_toks, #lef.latent_vars,
                            temp = self.latent_temp,
                            groups = self.groups,#cfg.latent_groups,
                            combine_groups = False,
                            vq_dim = self.embed ,
                            time_first = True,
                            ) 
          
         self.encoder_embed_dim = self.embed

         self.mask_emb = nn.Parameter(
                 torch.FloatTensor(self.encoder_embed_dim).uniform_()
                     )
         self.rnn = Emb(self.input_dim, self.embed, 2)
         self.encoder = TransformerEncoder(self.embed)
         self.layer_norm = LayerNorm(self.embed)

         self.target_glu = None
         final_dim = self.embed
#         self.encoder_embed_dim = self.embed


         self.n_negatives = 100 #cfg.num_negatives # num_negatives = 100
         self.cross_sample_negatives = 0 #cfg.cross_sample_negatives # cross_sample_negatives = 0
         self.codebook_negatives =0 # cfg.codebook_negatives # codebook_negatives = 0
         self.negatives_from_everywhere = False#cfg.negatives_from_everywhere
         # negatives_from_everywhere = False


         if self.target_glu :
            self.target_glu = nn.Sequential(
                    nn.Linear(final_dim, final_dim * 2), nn.GLU()
                     )
         self.final_proj = nn.Linear(self.encoder_embed_dim, final_dim)

         self.logit_temp = 0.1#cfg.logit_temp



     def apply_mask(self, x , padding_mask, mask_indices_ = None):
           B, T, C = x.shape
           if self.mask_prob > 0 :


                mask_indices = compute_mask_indices(
                           (B,T),
                           padding_mask,
                           self.mask_prob,
                           self.mask_length,
                           self.mask_selection,
                           self.mask_other,
                           min_masks = 2,
                           no_overlap = self.no_mask_overlap,
                           min_space = self.mask_min_space,

                             )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
                if mask_indices_ not None:
                
                   x[mask_indices_] = self.mask_emb
                else:
                   x[mask_indices] = self.mask_emb
           else:
                mask_indices = None

           if self.mask_channel_prob > 0 :
                 mask_channel_indices = compute_mask_indices(
                         (B, C),
                         None,
                         self.mask_channel_prob,
                         self.mask_channel_length,
                         self.mask_channel_selection,
                         self.mask_channel_other,
                         no_overlap = self.no_mask_channel_overlap,
                         min_space = self.mask_channel_min_space,
                               )
                 mask_channel_indices = (
                         torch.from_numpy(mask_channel_indices).to(device).unsqueeze(1)
                              .expand(-1, T, -1)


                             )
                 x[mask_channel_indices]= 0
           return x, mask_indices

     def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs


     def compute_preds(self, x, y, negatives):
            neg_is_pos = (y == negatives).all(-1)
            y = y.unsqueeze(0)
            targets = torch.cat([y, negatives],dim = 0)

            logits = torch.cosine_similarity(x.float(), targets.float(), dim = -1).type_as(x)

            logits /= self.logit_temp
            if neg_is_pos.any():
                 logits[1:][neg_is_pos]= float("-inf")
            return logits

     def quantize(self, x):
           assert self.quantizer is not None
           x = self.rnn(x)
           #x = x.transpose(1,2)
           x = self.layer_norm(x)
           return self.quantizer.forward_idx(x)

     def extract_features(self, source, padding_mask, mask= False):
            res = self.forward(source, padding_mask, mask= mask, features_only = True)
            return res["x"], res["padding_mask"]

     def get_logits(self, net_output):
            logits = net_output["x"]

            logits = logits.transpose(0,2)
            logits = logits.reshape(-1, logits.size(-1))

            return logits

     def get_targets(self, net_output, expand_steps = True):
           x = net_output["x"]
           return x.new_zeros(x.size(1)* x.size(2), dtype = torch.long)


     def get_extra_losses(self, net_output):
         pen= []
         if "prob_perplexity" in net_output:
              pen.append(
                 (net_output["num_vars"] - net_output["prob_perplexity"])/ net_output["num_vars"]
                 )

         if "features_pen" in net_output:
             pen.append(net_output["features_pen"])

         return pen

     def remove_pretraining_modules(self):
            self.quantizer = None
            self.project_q = None
            self.target_glu = None
            self.final_proj = None
     def produce_code(self, source):

           x = self.quantize(source)
           return x
      

     def forward(self, source, source2, padding_mask  = None,  mask = True , features_only = False, steps = None):
         features = self.rnn(source)   # [self.rnn(i.unsqueeze(0).to(self.device)).squeeze() for i in source]
         features2 = self.rnn(source2)   # [self.rnn(i.unsqueeze(0).to(self.device)).squeeze() for i in source]
         features_pen = features.float().pow(2).mean()
#         features = features.transpose(1,2)
         features_pen2 = features2.float().pow(2).mean()

         features = self.layer_norm(features)

         features2 = self.layer_norm(features2)
         unmasked_features = features.clone()

         unmasked_features2 = features2.clone()

         features = self.dropout_input(features)

         features2 = self.dropout_input(features2)
         
         unmasked_features = self.dropout_features(unmasked_features) 

         unmasked_features2 = self.dropout_features(unmasked_features2) 

         num_vars = None
         code_ppl = None
         prob_ppl = None
         curr_temp = None

         num_vars2 = None
         code_ppl2 = None
         prob_ppl2 = None
         curr_temp2 = None


         if mask:

                x, mask_indices = self.apply_mask(features, padding_mask = None)
                x2, mask_indices_ = self.apply_mask(features2, padding_mask= None, mask_indices_= mask_indices)

                if mask_indices is not None:

                      y = unmasked_features[mask_indices].view(
                             unmasked_features.size(0), -1, unmasked_features.size(-1)
                                )

                      y2 = unmasked_features2[mask_indices].view(
                             unmasked_features2.size(0), -1, unmasked_features2.size(-1)
                                )


                else:
                     y = unmasked_features
                     y2 = unmasked_features2

         else:
                x = features
                y = unmasked_features
                mask_indices = None

                x2 = features2
                y2 = unmasked_features2
                #mask_indices = None

         
         x = self.encoder(x, padding_mask = None)

         x2 = self.encoder(x2, padding_mask = None)

         if features_only:
               return {"x": x, "x2": x2,"padding_mask": padding_mask}




         q = self.quantizer(y, produce_targets =True)


         q2 = self.quantizer(y2, produce_targets =True)

         if steps != None:
                      self.quantizer.set_num_updates(steps)


         y = q["x"]
         num_vars = q["num_vars"]
         code_ppl = q["code_perplexity"]
         prob_ppl = q["prob_perplexity"]
         curr_temp = q["temp"]




         y2 = q2["x"]
         num_vars2 = q2["num_vars"]
         code_ppl2 = q2["code_perplexity"]
         prob_ppl2 = q2["prob_perplexity"]
         curr_temp2 = q2["temp"]
         #print(y.shape)
         #y = self.project_q(y)
         ts2 = q["targets"]

         ts1 = q2["targets"] 
         if self.negatives_from_everywhere:



                    neg_cands, *_ = self.quantizer(unmasked_features,\
                                    produce_targets = False)
                    negs, _ = self.sample_negatives(neg_cands, y.size(1))
                    negs = self.project_q(negs)


                    neg_cands2, *_ = self.quantizer(unmasked_features2,\
                                    produce_targets = False)
                    negs2, _ = self.sample_negatives(neg_cands2, y2.size(1))
                    negs2 = self.project_q(negs2)

         else:

                    negs, _ = self.sample_negatives(y, y.size(1))


                    negs2, _ = self.sample_negatives(y2, y2.size(1))


         if self.codebook_negatives > 0:

                       cb_negs = self.quantizer.sample_from_codebook(
                            y.size(0) * y.size(1), self.codebook_negatives
                                  )
                       cb_negs = cb_negs.view(
                               self.codebook_negatives, y.size(0), y.size(1), -1
                               ) #order doesn't matter
                       cb_negs = self.project_q(cb_negs)

                       negs = torch.cat([negs , cb_negs], dim = 0)


                       cb_negs2 = self.quantizer.sample_from_codebook(
                            y2.size(0) * y2.size(1), self.codebook_negatives
                                  )
                       cb_negs2 = cb_negs2.view(
                               self.codebook_negatives, y2.size(0), y2.size(1), -1
                               ) #order doesn't matter
                       cb_negs2 = self.project_q(cb_negs2)

                       negs2 = torch.cat([negs2 , cb_negs2], dim = 0)


         x = x[mask_indices].view(x.size(0),-1,x.size(-1))



         x = self.final_proj(x)

         x = self.compute_preds(x,y2, negs2)


         x2 = x2[mask_indices].view(x2.size(0),-1,x2.size(-1))



         x2 = self.final_proj(x2)

         x2 = self.compute_preds(x2,y, negs)

         result = {"x": x, "x2": x2, "padding_mask": padding_mask, \
                                   "features_pen": features_pen, "features_pen2": features_pen2}

         if prob_ppl is not None:
                      result["prob_perplexity"] = (prob_ppl + prob_ppl2)/2
                      result["code_perplexity"] = (code_ppl + code_ppl2)/2
                      result["num_vars"] = num_vars
                      result["temp"] = curr_temp
                      result["targets2"] = torch.unique_consecutive(ts1.squeeze()).detach().cpu().numpy()

                      result["targets1"] = torch.unique_consecutive(ts2.squeeze()).detach().cpu().numpy()

         return result





class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result


class Criterion(_Loss):
    def __init__(self):
        super().__init__()
        #self.cfg = cfg 
        self.infonce = False
        #self.loss_weights = [0.1,0.1,10]
        self.loss_weights = [0.1, 10]


        self.log_keys = ["prob_perplexity", "code_perplexity", "temp"] 


    def forward(self, model, x, reduce=True, steps = None):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(x, steps = steps)


        logits  = model.get_logits(net_output).float()
        #print(logits.shape)
        target = model.get_targets(net_output)
        #print(target)
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        
        losses = []
        if not self.infonce:
            loss= F.cross_entropy(
                logits,
                target,
                reduction="sum"# if reduce else "none",
            )
            
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target.float(),
                weights,
                reduction="sum" if reduce else "none",
            )

        losses.append(loss)
        sample_size = target.numel()

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)

            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]

            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)

            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,

            "sample_size": sample_size,
        }
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    logging_output["target"] = target.cpu().numpy()
            elif lk in net_output:
                if isinstance(net_output[lk], list) :
                    logging_output[lk] = float(sum(net_output[lk]))
                else:
                    logging_output[lk] = float(net_output[lk])


        if len(losses) > 1:
            for i, l in enumerate(losses):
            #    print(i, l)
                logging_output[f"loss_{i}"] = l.item()

        if not self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count


        logging_output["Tokens"] = net_output["targets"]
        return loss, sample_size, logging_output



'''
if __name__ == "__main__" :

    model = Wav2Vec2Model(513,560,2,50).cuda()

    x = torch.randn(size = (1,251,513)).cuda()
    criterion = Criterion()

    loss, _, logs = criterion(model, x)

    print(logs)'''
