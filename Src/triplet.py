
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np





class Emb(nn.Module):
       def __init__(
        self,
        input_dim ,
        emb_dim,

             ):

           super().__init__()

           self.input_dim = input_dim
           self.emb_dim = emb_dim
           self.lstm1 = nn.LSTM(input_dim,emb_dim //2,2, dropout = 0.25, \
                             bidirectional = True,batch_first = True)




       def forward(self, x1 ):

            x, _ = self.lstm1(x1)
            x = F.normalize(x, p=2, dim = -1)
            return x




class Triplet(nn.Module):
   def __init__(self , input_dim , emb_dim, window_size = 5):
      super().__init__()
      self.input_dim = input_dim
      self.emb_dim  = emb_dim
      self.embs = Emb(self.input_dim, self.emb_dim)
      self.temp = 0.05
      self.loss = nn.TripletMarginLoss(margin= 0.4, p =2.0, eps = 1e-8)
      self.device = 'cuda:0'
      self.window_length = window_size
   def sim_matrix(self,  a, b, eps=1e-8):

      a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
      a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
      b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
      sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
      return sim_mt

   def get_logits(self, x):
       diags = list(torch.diag(x))
       flat = torch.flatten(x).unsqueeze(1)

       logits = []
       for i in range(len(diags)):
          el = [diags[i].unsqueeze(0)]

          others = list(flat)

          others.pop(x.shape[0] + i)

          list_val = el + others

          logit = torch.cat(list_val).unsqueeze(0)

          logits.append(logit)
       logits = torch.cat(logits, 0)

       return logits
   def get_embs(self,x):
         z = [self.embs(i.unsqueeze(0).to(self.device)).squeeze() for i in x]
         return z
   def forward(self, x1, x2):
         z1 = self.get_embs(x1)
         z2 = self.get_embs(x2)
         #bs , ts , f = z1.shape

         anc = []
         p = []
         n = []
         for (i,j) in zip(z1, z2):
             z1_ = i
             z2_ = j
             len1_ , _ = i.shape
             len2_, _ = j.shape
             inds1 = torch.arange(min(len1_, len2_)).numpy().tolist()
             inds2 = torch.arange(max(len1_,len2_)).numpy().tolist()

             len1 = len(inds1)
             len2 = len(inds2)
             s = (len2 - len1+1)/(len1-1)
             neg = []
             pos = []
             for k in range(len1):

                   positive =inds2[max(k + int(k*s) -self.window_length ,0): min(k+ int(k*s)  + self.window_length, len2+1)]
                   negative= [m for m in inds2 if m not in positive]
                   positive =inds2[max(k + int(k*s) - 1,0): min(k+ int(k*s)  + 1, len2+1)]
                   #print(el2)
                   el2= np.random.choice(positive)
                   el1 = np.random.choice(negative)

                   #print(k)
                   #print(el2)

                   neg.append(el1)
                   pos.append(el2)


            # print(len(i2),len1, len2, len1_,len2_)
             if len1 == len1_:
                  anchor = z1_

                  positive = z2_[pos]
                  negative = z2_[neg]
             else:
                  anchor = z2_
                  positive = z1_[pos]
                  negative = z1_[neg]
             #print('len',len(inds1))
             anc.append(anchor)
             p.append(positive)
             n.append(negative)


         anc = torch.cat(anc, 0)
         p = torch.cat(p, 0)
         n = torch.cat(n,0)

         loss = self.loss(anc,p,n)
         logs= {'loss': 10* loss.detach().cpu().item()}

         return loss, logs
