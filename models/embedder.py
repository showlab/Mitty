
import torch


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x):
        "x: torch.Size([2, 81, 3]). dim: cat dim"
        x = x.unsqueeze(-1)
        #
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )

        return torch.cat(out, dim=-1)


if __name__  == '__main__':
    embedder = FourierEmbedder()
    B, F, dim = 2, 49, 3
    data = torch.randn(B, F, dim)
    y = embedder(data)
    # y.shape ---> torch.Size([N, dim * 2 * num_freqs])
    breakpoint()