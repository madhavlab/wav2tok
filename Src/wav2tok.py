import torch
from itertools import islice
import torch.nn as nn
import torch.nn.functional as F
from new_function_library import load
from findpeaks import findpeaks

import numpy as np
from new_function_library import save
import random
import librosa

from sklearn.cluster import KMeans

import math




#@torch.jit.script
def ctc_loss(log_probs : torch.Tensor, targets : torch.Tensor, input_lengths : torch.Tensor, target_lengths : torch.Tensor, blank : int = 0, reduction : str = 'mean', finfo_min_fp32: float = torch.finfo(torch.float32).min, finfo_min_fp16: float = torch.finfo(torch.float16).min, alignment : bool = False):
	input_time_size, batch_size = log_probs.shape[:2]
	B = torch.arange(batch_size, device = input_lengths.device)
	
	_t_a_r_g_e_t_s_ = torch.cat([targets, targets[:, :1]], dim = -1)
	_t_a_r_g_e_t_s_ = torch.stack([torch.full_like(_t_a_r_g_e_t_s_, blank), _t_a_r_g_e_t_s_], dim = -1).flatten(start_dim = -2)
	
	diff_labels = torch.cat([torch.as_tensor([[False, False]], device = targets.device).expand(batch_size, -1), _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]], dim = 1)
	
	# if zero = float('-inf') is used as neutral element, custom logsumexp must be used to avoid nan grad in torch.logsumexp
	
	zero_padding, zero = 2, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype)
	log_probs_ = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(input_time_size, -1, -1))
	log_alpha = torch.full((input_time_size, batch_size, zero_padding + _t_a_r_g_e_t_s_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[0, :, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]
	# log_alpha[1:, :, zero_padding:] = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(len(log_probs), -1, -1))[1:]
	for t in range(1, input_time_size):
		log_alpha[t, :, 2:] = log_probs_[t] + logadd(log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1], torch.where(diff_labels, log_alpha[t - 1, :, :-2], zero))

	l1l2 = log_alpha[input_lengths - 1, B].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	loss = -torch.logsumexp(l1l2, dim = -1)
	return loss

	if not alignment:
		return loss
	
	# below is for debugging, for real alignment use more efficient the distinct ctc_alignment(...) method
	path = torch.zeros(len(log_alpha), len(B), device = log_alpha.device, dtype = torch.int64)
	path[input_lengths - 1, B] = zero_padding + 2 * target_lengths - 1 + l1l2.max(dim = -1).indices
	for t, indices in reversed(list(enumerate(path))[1:]):
		indices_ = torch.stack([(indices - 2) * diff_labels[B, (indices - zero_padding).clamp(min = 0)], (indices - 1).clamp(min = 0), indices], dim = -1)
		path[t - 1] += (indices - 2 + log_alpha[t - 1, B].gather(-1, indices_).max(dim = -1).indices).clamp(min = 0)
	return torch.zeros_like(log_alpha).scatter_(-1, path.unsqueeze(-1), 1.0)[..., (zero_padding + 1)::2]




def logadd(x0, x1, x2):
	# produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
	#return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)
	
	# use if -inf log-space zero element is used
	return LogsumexpFunction.apply(x0, x1, x2)
	
	# produces inplace modification error https://github.com/pytorch/pytorch/issues/31819
	#m = torch.max(torch.max(x0, x1), x2)
	#m = m.masked_fill(torch.isinf(m), 0)
	#res = (x0 - m).exp() + (x1 - m).exp() + (x2 - m).exp()
	#return res.log().add(m)

class LogsumexpFunction(torch.autograd.function.Function):
	@staticmethod
	def forward(self, x0, x1, x2):
		m = torch.max(torch.max(x0, x1), x2)
		m = m.masked_fill_(torch.isinf(m), 0)
		e0 = (x0 - m).exp_()
		e1 = (x1 - m).exp_()
		e2 = (x2 - m).exp_()
		e = (e0 + e1).add_(e2).clamp_(min = 1e-16)
		self.save_for_backward(e0, e1, e2, e)
		return e.log_().add_(m)

	@staticmethod
	def backward(self, grad_output):
		e0, e1, e2, e = self.saved_tensors
		g = grad_output / e
		return g * e0, g * e1, g * e2


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





       def forward(self, x1 ):

            x, _ = self.lstm1(x1)


            x = F.normalize(x, p=2, dim = -1)
            return x





class wav2tok(nn.Module):
  def __init__(self , input_dim , emb_dim, alpha = 0.01, beta = 0.01,temp = 0.1, dataset= 'MIR', iter_clust = 500, cluster_split = 0.1,  use_cosine = False,  use_transformer = False, \
                                       num_tokens=25, num_layers= 2, device = 'cuda:0'):
      super().__init__()

      self.input_dim = input_dim

      self.num_toks = num_tokens

      self.num_layers = num_layers

      self.emb_dim = emb_dim

      self.use_transformer = use_transformer
 
      self.cluster_split = cluster_split

      self.use_cosine = use_cosine
 
      self.no_sim = no_sim

      self.alpha = alpha
       
      self.beta = beta

      if not self.use_transformer: 
           self.embs  = Emb(self.input_dim, self.emb_dim, self.num_layers)
      else:
           self.embs= TransformerEncoder(self.input_dim, self.emb_dim)

      self.codebook = nn.Parameter(torch.FloatTensor(self.num_toks,self.emb_dim).uniform_())

      
      self.class_dis = nn.Linear(self.emb_dim ,1)
      self.loss = nn.CrossEntropyLoss()
      self.temp = temp
   
      self.dataset = dataset

      self.iter_clust = iter_clust
      self.device = device




  def initialize_classifier(self, cluster_centers):
        save(cluster_centers, 'Cluster_center_forModel')

        self.codebook = nn.Parameter(F.normalize(torch.Tensor(cluster_centers),\
                         p  =2 , dim= -1).to(self.device))

        print('Model Initialized')




  def get_feats(self, x, mfcc= True):
          
        if mfcc:

            mfccs = librosa.feature.mfcc(y=x, sr= 16000, n_mfcc= 13, n_fft= 368)


            deltas = librosa.feature.delta(mfccs, mode = 'constant')
            ddeltas = librosa.feature.delta(deltas, mode = 'constant')
            concat = np.concatenate([mfccs, deltas, ddeltas], axis = 0)
            concat = concat.T  

        else:
            stft = librosa.stft(y =x , sr = 16000, n_fft = 1024)
       
            stft = np.abs(stft)
            concat = stft.T
            
 
        return concat




  def cluster(self,dataset):
      
       tr = load(dataset)[0]

       X = []

       tr = random.sample(tr, int(self.cluster_split* len(tr)))
       if dataset == 'librispeech'


  
         for i in tr:
              a1, _  = librosa.load(i, sr = 16000)

              a_len = len(a1)//16000

              splits = a_len//3


              a1 = [a1[j*16000*3: (j+1)*16000*3] for j in range(splits)]
              
              a1 = [torch.tensor(self.get_feats(j)) for j in a1]
              
              a1 = self.get_embs(a1)

              X.extend(a1)

       else:

          for i in tr.keys():

              el = tr[i]

              for j in el:

                 a1, _  = librosa.load(j, sr = 16000)                   

                 a1 = [torch.tensor(self.get_feats(k)) for k in a1]
              
                 a1 = self.get_embs(a1)

                 X.extend(a1)



     
       X = np.concatenate(X)
       print(X.shape)
       clusterer = KMeans(n_clusters = self.num_toks).fit(X)

       cluster_centroids = clusterer.cluster_centers_





       print(cluster_centroids.shape)
       print('INITIALIZING CLASSIFIER')
       self.initialize_classifier(cluster_centroids)   
       
       return      

  def gen_prototype(self,unique1,feats1,tokens1):

      dict1 = {}

      for i in unique1:

         label1 = i.repeat(tokens1.shape[0])

         match1 = torch.where(tokens1.cpu() == label1.cpu(),torch.tensor( 1.0),torch.tensor(0.0))

         inds = torch.nonzero(match1, as_tuple = False).squeeze()

         if inds.dim() != 0:

              dict1[i.item()] = F.normalize(feats1[inds].mean(0), p =2 , dim = -1)

         else:

              dict1[i.item()] = feats1[inds]

      return dict1


  def matching_loss_cal(self, dict1): 





       x = []
       lab = []





       x, lab = self.inter_dict_weights(x, lab , dict1)

       x= torch.cat(x, 0)
       lab = torch.tensor(lab).to(self.device)

       if not self.use_cosine:
             x= self.class_dis(x)
       x = x/ self.temp
       loss = self.loss(x,lab)
       return loss





  def inter_dict_weights(self, x, lab, dict1):
       codes =  F.normalize(self.codebook, p= 2, dim = -1)
       #print(weights.shape)
       for i in dict1.keys():
           el = dict1[i]

		
           if self.use_cosine:
	      diff = torch.cosine_similarity(el.float(), codes.float(), dim=-1).type_as(
                                codes
                                  )
      
	
	     
		
           else: 		
              el = el.repeat(self.num_toks, 1)

           
              diff = torch.abs(el - codes)   
  
              diff = diff.unsqueeze(0)


            x.append(diff)
  
            lab.append(i)
  


       return x, lab



  def ctc_loss_cal(self, t, tok) :
         t = t.unsqueeze(1)


         inps =  torch.full(size =(1,) , fill_value = t.size(0), dtype= torch.int32)
         targs = torch.full(size =(1,) , fill_value = tok.size(0), dtype= torch.int32)


 

         ctc_loss = ctc_loss(t, tok,inps, targs)



      
         return ctc_loss

  def get_embs(self, x):


         z = [self.embs(i.unsqueeze(0).to(self.device)).detach().squeeze().cpu().numpy() for i in x]  
         return z




  def get_tokens(self,x, mfcc = False):
      x = [torch.tensor(self.get_feats(i.cpu().numpy(), mfcc = mfcc)) for i in x]
      z = [self.embs(i.unsqueeze(0).to(self.device)).squeeze().detach() for i in x]

      t = []
       
      for i in range(len(z)):

            t_feats = z[i]


       

            ts = len(t_feats)
            codes = F.normalize(self.codebook.unsqueeze(0).repeat(ts,1,1).permute(1,0,2), p =2 , dim = -1)





            logits = torch.cosine_similarity(t_feats.float(), codes.float(), dim=-1).type_as(
                                 codes
                                    ).T
            logits = logits/ self.temp

            #print(logits.shape)
            t.append(logits.argmax(-1)) 

      
      return z , t






  def forward(self, x1 , x2, steps= None):
      loss = []
      logs = {}


      if steps is not None:
         if steps % self.iter_clust == 0:
            print('Clustering')
            self.cluster(self.dataset)
   
      z1 , t1 = self.get_tokens(x1)
      z2 , t2 = self.get_tokens(x2)
      bs = len(x1)

      loss_ctc= [] 
      loss_m = []
      for i in range(bs):



        tks1 = torch.cat([t1[i], t2[i]], 0 )
 
        tokens1= tks1.argmax(-1)
        unique1 = torch.unique(tokens1)
       

### ENFORCE BLANK log softmax prob as  -negative infinity 
   
        logits1 = F.log_softmax(t1[i], dim = -1)

        logits1  = torch.cat([torch.full((len(logits1),1), -float("inf")).to(self.device), logits1], dim = -1)

        targs1 = torch.unique_consecutive(t2[i].argmax(-1)) 



        logits2 = F.log_softmax(t2[i], dim = -1)

        logits2  = torch.cat([torch.full((len(logits2),1), -float("inf")).to(self.device), logits2], dim = -1)

        targs2 = torch.unique_consecutive(t1[i].argmax(-1)) 

        l_ctc1 = 0
 
        l_ctc2 = 0

        if len(z1[i]) >= len(t2[i]):

             l_ctc1 = self.ctc_loss_cal(logits1,targs1)

        if len(z2[i]) >= len(t1[i]):
             l_ctc2 = self.ctc_loss_cal(logits2,targs2)  

        l_ctc = self.alpha*l_ctc1 + self.beta * l_ctc2




        loss_ = self.matching_loss_cal(dict1) 

        loss_m.append(loss_) 
        loss_ctc.append(l_ctc)   


      loss_m = sum(loss_m)/len(loss_m)
      loss_ctc = sum(loss_ctc)/len(loss_ctc)
      loss = loss_m + loss_ctc
      loss =  sum(loss)/len(loss)
      logs['loss'] = loss.item()
      logs['matching'] = loss_m.item()
      logs['ctc'] = loss_ctc.item()

      logs['unique x1 '] = unique1.detach().cpu().numpy()





      return loss, logs







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
    def __init__(self, input_dim, encoder_dim):
        super().__init__()

        self.dropout = 0.1 #  args.dropout
        self.embedding_dim = encoder_dim#args.encoder_embed_dim
        self.conv_pos = 128
        self.conv_pos_groups = 16
        self.encoder_ffn_embed_dim = 2272
        self.encoder_attention_heads = 2
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.activation_fn = "gelu"
        self.layer_norm_first = False
        self.encoder_layers = 2
        self.encoder_layerdrop = 0.05
        self.input_dim = input_dim
        self.pos_conv = nn.Conv1d(
            self.input_dim,
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
        x = F.normalize(x, p = 2 , dim = -1)
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




