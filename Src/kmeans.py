
import torch
import torch.nn as nn
#from modules import Fp32GroupNorm

import torch.nn.functional as F

#0.25

class KmeansVectorQuantizer(nn.Module):
    def __init__(
        self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.0  
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.embedding =nn.Parameter(
       torch.randn(num_vars, num_groups, self.var_dim)
        )
       # self.projection = nn.Sequential(
       #     nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
       #     Fp32GroupNorm(groups, dim),
       # )
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction="mean")

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """

        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return F.normalize(self.embedding, p =2 , dim = -1).expand(self.num_vars, self.groups, self.var_dim)
        return F.normalize(self.embedding, p =2 , dim =-1)

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["targets"]

    def forward(self, x, produce_targets=True):

        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape
        #print(bsz,fsz, tsz)
        #ze = F.normalize(self.projection(x), p =2 , dim = 1)
        ze = x
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        #print(ze_.shape,  self.expand_embedding.unsqueeze(1).unsqueeze(1).shape)
        d = (
            (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1))
            .view(self.num_vars, bsz, tsz, self.groups, -1)
            .norm(dim=-1, p=2)
        )
        idx = d.argmin(dim=0)
        zq = (
            torch.stack(
                [
                    self.expand_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
            .view(bsz, tsz, self.groups * self.var_dim)
            .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)

        hard_x = (
            idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
            .scatter_(-1, idx.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
        #result["x"] = x

        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        result["kmeans_loss"] = latent_loss + self.gamma * commitment_loss

        return result


class KmeansVectorQuantizer2(nn.Module):
    def __init__(
        self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.25
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.embedding =nn.Parameter(
       torch.randn(num_vars, num_groups, self.var_dim)
        )
       # self.projection = nn.Sequential(
       #     nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
       #     Fp32GroupNorm(groups, dim),
       # )
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction="mean")

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """

        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["targets"]

    def forward(self, x, produce_targets=True):

        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape
        
        #ze = F.normalize(self.projection(x), p =2 , dim = 1)
        ze = x
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (
            (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1))
            .view(self.num_vars, bsz, tsz, self.groups, -1)
            .norm(dim=-1, p=2)
        )
        idx = d.argmin(dim=0)
        zq = (
            torch.stack(
                [
                    self.expand_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
            .view(bsz, tsz, self.groups * self.var_dim)
            .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)

        hard_x = (
            idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
            .scatter_(-1, idx.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
        #result["x"] = x

        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        result["kmeans_loss"] = latent_loss + self.gamma * commitment_loss

        return result


class SymbolEncoder(nn.Module):
    def __init__(
        self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.0
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.embedding =nn.Parameter(
       torch.randn(num_vars, num_groups, self.var_dim)
        )
       # self.projection = nn.Sequential(
       #     nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
       #     Fp32GroupNorm(groups, dim),
       # )
        self.gamma = gamma
        self.bce_mean = nn.BCELoss(reduction="mean")


    def sim_matrix(self,  a, b, eps=1e-8):

      a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
      a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
      b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
      sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
      return sim_mt



    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["targets"]

    def forward(self, x, produce_targets=True):

        result = {"num_vars": self.num_vars}


        bsz, tsz, fsz = x.shape
        
        #ze = F.normalize(self.projection(x), p =2 , dim = 1)
        ze = x.squeeze()
        protos = F.normalize( self.embedding.squeeze(), p =2 , dim = -1 )
        d = self.sim_matrix(ze, protos)
       
        

        idx = d.argmax(-1)
   
        zq = protos[idx]

         

        if produce_targets:
            result["targets"] = idx

     
        result["x"] = d

        ze = ze.float()
        zq = zq.float()
        sim1 = nn.CosineSimilarity(dim = 1)(ze.detach(), zq)
        #print(sim1.shape)
        sim2 = nn.CosineSimilarity(dim=1)(ze, zq.detach())
        labs = torch.ones(size = (tsz,)).cuda()
         
        latent_loss = self.bce_mean(sim1, labs)
        comm_loss = self.bce_mean(sim2,labs)
        result["kmeans_loss"] = latent_loss + self.gamma * comm_loss
        return result
