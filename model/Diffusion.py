# Copyright (c) 2021 Yusuke Tashiro
# Licensed under the MIT License
# All rights reserved.
# Modified by XXX on 2024-10-14
# --------------------------------------------------------
# References:
# CSDI: https://github.com/ermongroup/CSDI
# --------------------------------------------------------


import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NPDiff import Denoising_network








class  Diff_base(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.model_name=args.model
        self.device = args.device
        self.target_dim = args.target_dim
        self.emb_time_dim = 128
        self.emb_feature_dim = 16
        self.is_unconditional = 0
        self.target_strategy = "test"
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        self.diffmodel = Denoising_network().get_model(args)
        # self.side_dim = self.emb_total_dim
         


        self.num_steps = 50

        self.beta = np.linspace(
                0.0001 ** 0.5, 0.5 ** 0.5, self.num_steps
            ) ** 2


        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe



    def calc_loss_valid(
        self, prior,observed_data, cond_mask,  side_info, is_train,
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
               prior,observed_data, cond_mask, side_info, is_train,set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps


    def calc_loss(
        self,prior,observed_data, cond_mask, side_info, is_train,set_t=-1
    ):
        if(self.model_name=="CSDI") or (self.model_name=="STID"):
            B, _, _ = observed_data.shape # B,H*W,L
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(self.device)
            else:
                t = torch.randint(0, self.num_steps, [B]).to(self.device)
            current_alpha = self.alpha_torch[t]  # (B,1,1)
            noise = torch.randn_like(observed_data)
        elif self.model_name=="ConvLSTM":
            B, _, _ ,_= observed_data.shape # B,H*W,L
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(self.device)
            else:
                t = torch.randint(0, self.num_steps, [B]).to(self.device)
            current_alpha = self.alpha_torch[t].unsqueeze(3)  # (B,1,1)
            noise = torch.randn_like(observed_data)            

        # B,H*W,L
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        # (B,2,H*W,L) history pure data + noisy predict data
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask) 

        predicted = self.diffmodel(prior,total_input, side_info, t,current_alpha)  # (B,H*W,L)

        target_mask = torch.ones_like(cond_mask) - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,H*W,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1) 
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,H*W,L)

        return total_input

    def impute(self,prior, observed_data, cond_mask, side_info, n_samples):
        if (self.model_name=="CSDI") or (self.model_name=="STID"):
            B, K, L = observed_data.shape
            imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        elif self.model_name=="ConvLSTM":
            B, L, H, W = observed_data.shape
            imputed_samples = torch.zeros(B, n_samples, H, W, L).to(self.device)
        else:
            raise ValueError(f"Model {self.model_name} not found")

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                
                predicted = self.diffmodel(prior,diff_input, side_info, torch.tensor([t]).to(self.device),self.alpha_torch[t])

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            
            if self.model_name=="ConvLSTM":
                imputed_samples[:, i] = current_sample.permute(0,2,3,1).detach()
            else:
                imputed_samples[:, i] = current_sample.detach()

        return imputed_samples












class Diff_Forecasting( Diff_base):
    def __init__(self,args):
        super(Diff_Forecasting, self).__init__(args)
        self.target_dim_base = args.target_dim
        self.data_name=args.data_name


    def process_data(self, batch):

        observed_data = batch["observed_data"].to(self.device).float() # B,L,H,W
        observed_tp = batch["timepoints"].to(self.device).float() # B,L,2
        gt_mask = batch["gt_mask"].to(self.device).float() # B,L,H,W
        prior= batch["prior"].to(self.device).float() # B,L,H,W

        B,L,H,W=gt_mask.shape
        if (self.model_name=="CSDI") or (self.model_name=="STID"):
            observed_data = rearrange(observed_data,'b l h w -> b (h w) l')
            gt_mask = rearrange(gt_mask,'b l h w -> b (h w) l')
            prior=rearrange(prior,'b l h w -> b (h w) l')
        return (
            observed_data,
            observed_tp,
            gt_mask,
            prior
        )        



    def get_side_info(self, observed_tp, cond_mask):
        if self.model_name=="CSDI":
            B, K, L = cond_mask.shape
            time_embed_hour = self.time_embedding(observed_tp[:,:,1], self.emb_time_dim-self.emb_time_dim//4)  # (B,L,emb)
            time_embed_hour = time_embed_hour.unsqueeze(2).expand(-1, -1, self.target_dim, -1) # (B,L,H*W,emb)
        
            time_embed_day = self.time_embedding(observed_tp[:,:,0], self.emb_time_dim//4)  # (B,L,emb)
            time_embed_day = time_embed_day.unsqueeze(2).expand(-1, -1, self.target_dim, -1) # (B,L,H*W,emb)
        
            feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # (H*W,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) #(B,L,H*W,emb)

            side_info = torch.cat([time_embed_hour,time_embed_day, feature_embed], dim=-1)  # (B,L,H*W,*)
            side_info = side_info.permute(0, 3, 2, 1)  # (B,*,H*W,L)
            if self.is_unconditional == False:
                side_mask = cond_mask.unsqueeze(1)  # (B,1,H*W,L)
                side_info = torch.cat([side_info, side_mask], dim=1)
        elif self.model_name=="STID":
            B, K, L = cond_mask.shape
            cond_mask=cond_mask.permute(0,2,1).unsqueeze(-1)  # (B,L,N,1)
            Points=24 if self.data_name!="MobileSH16" else 96
            time_embed_hour = observed_tp[:,:,1].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.target_dim, -1)/Points
            time_embed_day = observed_tp[:,:,0].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.target_dim, -1)/7 # (B,L,H*W,1)            
            side_info = torch.cat([time_embed_hour,time_embed_day], dim=-1)  # (B,L,H*W,2)
            if self.is_unconditional == False:
                side_mask = cond_mask #.permute(0,3,2,1)  # (B,L,H*W,1)
                side_info = torch.cat([side_info, side_mask], dim=-1)  # (B,L,H*W,3) 
        elif self.model_name=="ConvLSTM":
            B, L,H,W = cond_mask.shape

            time_embed_hour = self.time_embedding(observed_tp[:,:,1], self.emb_time_dim-self.emb_time_dim//4)  # (B,L,emb)
            time_embed_hour = time_embed_hour.unsqueeze(2).unsqueeze(2).expand(-1, -1, H,W, -1) # (B,L,H*W,emb)
        
            time_embed_day = self.time_embedding(observed_tp[:,:,0], self.emb_time_dim//4)  # (B,L,emb)
            time_embed_day = time_embed_day.unsqueeze(2).unsqueeze(2).expand(-1, -1, H,W, -1) # (B,L,H*W,emb)
        
            feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # (H*W,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).reshape(B,L,H,W,-1) #(B,L,H,W,emb)

            side_info = torch.cat([time_embed_hour,time_embed_day, feature_embed], dim=-1)  # (B,L,H,W,*)
            side_info = side_info.permute(0,4,1,2, 3)  # (B,*,H,W,L)

            if self.is_unconditional == False:
                side_mask = cond_mask.unsqueeze(1)  # (B,1,H*W,L)
                side_info = torch.cat([side_info, side_mask], dim=1)                       
        return side_info






    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_tp,
            gt_mask,
            prior
        ) = self.process_data(batch)

    
        cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask)  # (B,C_info,H*W,L)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(prior,observed_data, cond_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_tp,
            gt_mask,
            prior
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)
            # B, n_samples, K, L
            samples = self.impute(prior,observed_data, cond_mask, side_info, n_samples)

        return samples
