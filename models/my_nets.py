import sys, os
sys.path.insert(0, '/data/lxy/sqj/code/InteractionVideo')

import torch
import torch.nn as nn
from models.embedder import FourierEmbedder


class FlowNet(nn.Module):
    def __init__(self, entity_dim=226, hidden_dim=256, out_dim=512, fourier_freqs=8):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        #
        self.input_dim = fourier_freqs*2 # 2 is sin&cos
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim 
        #   
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
    
    def forward(self, inputs):
        flow_embedding = self.fourier_embedder(inputs) # [B, F, N, C]
        # breakpoint()
        out = self.mlp(flow_embedding) # [B, F, N, C]
        B, F, N, C = out.shape
        # breakpoint()
        # 压缩
        res = out[:, 1:, :, :].reshape(B, -1, 4, N, C).mean(dim=2)
        out = torch.cat([out[:, :1, :, :], res], dim=1)
        # breakpoint()
        return out


if __name__ == '__main__':
    flow_net = FlowNet()
    B, F, dim = 2, 49, 3
    data = torch.randn(B, F, dim)
    y = flow_net(data)
    # y.shape ---> torch.Size([N, dim * 2 * num_freqs])
    breakpoint()