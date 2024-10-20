# Copyright (c) 2021 Yusuke Tashiro
# Licensed under the MIT License
# All rights reserved.
# Modified by XXX on 2024-10-14
# Changes: 
# - Changed the model architecture to a diffusion denoising network for NPDiff.
# --------------------------------------------------------
# References:
# CSDI: https://github.com/ermongroup/CSDI
# --------------------------------------------------------


from linear_attention_transformer import LinearAttentionTransformer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer




class CSDI(nn.Module):
    def __init__(self, args, inputdim=2):
        super().__init__()
        self.channels = 64

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=50,
            embedding_dim=128,
        )
        self.side_dim=145
        # channels: 2->64
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # channels: 64->1
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.Lambda = args.Lambda
        self.his_len=args.history_len

        self.local_prior = args.local_prior

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=self.side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=128,
                    nheads=8,
                    is_linear=True,
                )
                for _ in range(4)
            ]
        )



    def forward(self,prior,x, cond_info, diffusion_step,alpha_torch):
        
        # noise prior
        B,K,L=prior.shape
        future_prior=prior[:,:,self.his_len:]
        if self.local_prior & (L-self.his_len==1):
            future_prior=x[:,0,:,self.his_len-1:self.his_len]
        noise_prior= (x[:,1,:,self.his_len:]-torch.sqrt(alpha_torch)*future_prior)/torch.sqrt(1-alpha_torch) #B,K,L
        noise_prior=torch.cat([torch.zeros(B,K,self.his_len).to(noise_prior.device),noise_prior],dim=-1)



        B, inputdim, K, L = x.shape # B,2,H*W,24
        x = x.reshape(B, inputdim, K * L) # B,2,H*W*24
        x = self.input_projection(x) # B,64,H*W*24
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L) # B,64,H*W,24

        diffusion_emb = self.diffusion_embedding(diffusion_step) # B,128   128为扩散步的向量编码

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb) #x [B, 64, H*W, 24] cond_info:[B, C_info, H*W, 24]
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,H*W*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,H*W*L)
        x = x.reshape(B, K, L) 

        x=(1.0-self.Lambda)*x+self.Lambda*noise_prior #B,K,L
        

        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


