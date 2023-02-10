
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



class MIPS(nn.Module):
   def __init__(self , input_dim , emb_dim, window_size = 3):
      super().__init__()
      self.input_dim = input_dim
      self.emb_dim  = emb_dim
      self.embs = Emb(self.input_dim, self.emb_dim)
      self.temp = 0.05
      self.loss = nn.CrossEntropyLoss()
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

         z_1 = []
         z_2 = []
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
             #i1 = []
             i2 = []
             for k in range(len1):

                   el2 =inds2[max(k + int(k*s) -self.window_length ,0): min(k+ int(k*s)  + self.window_length, len2+1)]


                   #print(el2)
                   el2= np.random.choice(el2)
                   i2.append(el2)


            # print(len(i2),len1, len2, len1_,len2_)
             if len1 == len1_:
                  zs_1 = z1_

                  zs_2 = z2_[i2]
             else:
                  zs_1 = z2_
                  zs_2 = z1_[i2]
             #print('len',len(inds1))
             z_1.append(zs_1)
             z_2.append(zs_2)


         z1_s = torch.cat(z_1, 0)
         z2_s = torch.cat(z_2, 0)

         sim_mat =self.sim_matrix(z1_s, z2_s)
         #print(sim_mat.shape)
         logits = sim_mat / self.temp

         labs = torch.arange(logits.shape[0]).to(self.device)

         labs2 = torch.arange(logits.shape[1]).to(self.device)
         loss = self.loss(logits,labs) + self.loss(logits.T, labs2)
         logs= {'loss': loss.detach().cpu().item()}

         return loss, logs
