import torch, math
import torch.nn as nn
from einops import rearrange
    

# part code is from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer_Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.Sequential(
            Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
            FeedForward(dim, mlp_dim, dropout = dropout)
        )

    def forward(self, x):
        return self.layers(x)

class Cross_Attention_Fourier(nn.Module):
    '''
    MSA with FFT.
    param dim:      num of input feature channels
    param heads:    num of heads
    param dim_head: dimension of processing features' channels for per head
    param dropout:  set the prob for dropout (default: 0.)
    param query:    must be "condition" or "diffusion", determine which kind of feature is query

    code is adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    '''
    def __init__(self, dim, time_emd_dim=512, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_head = self.dim // heads if heads else dim_head
        project_out = True # whenever we have a MLP in the last
        self.heads = heads

        self.attend = nn.Softmax(dim = -1)
        self.linear_trans = nn.Linear(self.dim, self.dim)
        self.to_q =  nn.Linear(dim, dim, bias = False)
        self.to_k =  nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.emd_fc1 = nn.Linear(time_emd_dim, dim*2)
        self.emd_fc_mean_std = nn.Linear(dim*2, dim*2)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = FeedForward(dim, dim, dropout=dropout)
        self.layernorm_con = nn.LayerNorm(dim)
        self.layernorm_diff = nn.LayerNorm(dim)


    def forward(self, con_features, diff_features, time_emb):
        fea_q, fea_kv = (diff_features, con_features)
        fea_q = self.layernorm_diff(fea_q)
        fea_kv = self.layernorm_con(fea_kv)

        # fea_ff_q = torch.fft.fft2(fea_q, dim=(-2, -1), norm='ortho')
        # fea_ff_k = torch.fft.fft2(fea_kv, dim=(-2, -1), norm='ortho')

        q = self.to_q(fea_q)
        k = self.to_k(fea_kv)
        v = self.to_v(fea_kv)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))

        q = torch.fft.fft2(q, dim=(-2, -1), norm='ortho')
        k = torch.fft.fft2(k, dim=(-2, -1), norm='ortho')

        dots = torch.matmul(q, k.adjoint())
        # dots = torch.matmul(q.real, k.real) + torch.matmul(q.imag, k.imag)
        attn1 = dots
        attn2 = torch.fft.ifft2(attn1, dim=(-2, -1), norm='ortho')
        attn3 = torch.sqrt(attn2.real**2 + attn2.imag**2)
        attn4 = torch.matmul(self.softmax(attn3) / math.sqrt(self.dim_head), v)
        attn5 = rearrange(attn4, 'b h n d -> b n (h d)', h=self.heads)

        out1 = (attn5 - torch.mean(attn5, dim=(-2, -1), keepdim=True)) / (torch.std(attn5, dim=(-2, -1), keepdim=True))
        time_emb_att = nn.functional.silu(self.emd_fc1(time_emb))
        time_emb_att = self.emd_fc_mean_std(time_emb_att)
        mean, std = time_emb_att.chunk(2, dim=-1)
        # std = std + self.std_eps
        mean, std = (mean.unsqueeze(1), std.unsqueeze(1))

        out = out1 * std + mean

        out = self.to_out(out)
        out = self.mlp(out)

        return out


class SSformer(nn.Module):
    '''
    SSformer: process the deepest features of UNET
    param: 
    '''
    def __init__(self, dim, image_size, time_emd_dim=512, heads = 8, dim_head = 64, dropout = 0.1,  std_eps = 1e-3, num_block=4):
        super().__init__()
        self.layer_norm_con = nn.LayerNorm(dim)
        self.layer_norm_diff = nn.LayerNorm(dim)
        self.c_lproj = nn.Linear(dim, dim)
        self.d_lproj = nn.Linear(dim, dim)
        self.image_size = image_size
        self.c_pos = nn.Parameter(torch.rand(self.image_size ** 2, dim))
        self.d_pos = nn.Parameter(torch.rand(self.image_size ** 2, dim))

        self.layers = nn.ModuleList([])
        for _ in range(num_block):
            self.layers.append(nn.ModuleList([
                Transformer_Block(dim=dim, heads=heads, dim_head=dim_head, mlp_dim=dim, dropout=dropout),
                Cross_Attention_Fourier(dim=dim, time_emd_dim=time_emd_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ]))
        for att_con, att_diff in self.layers:
            for module in att_diff.children():
                module.register_full_backward_hook(ssformer_backward_hook_fn)
            for module in att_con.children():
                module.register_full_backward_hook(ssformer_backward_hook_fn)
            
    def forward(self, con_features, diff_features, time_emb):
        c_features = con_features
        d_features = diff_features
        c_features = rearrange(c_features, 'b c h w -> b (h w) c', h=self.image_size, w=self.image_size)
        d_features = rearrange(d_features, 'b c h w -> b (h w) c', h=self.image_size, w=self.image_size)
        c_features = self.c_lproj(c_features)
        d_features = self.d_lproj(d_features)
        c_features += self.c_pos
        d_features += self.d_pos

        for att_con, att_diff in self.layers:
            c_features = att_con(c_features)
            d_features = att_diff(c_features, d_features, time_emb)

        d_features = rearrange(d_features, 'b (h w) c -> b c h w', h=self.image_size, w=self.image_size)

        return d_features
    
# def ssformer_hook_fn(module, grad):
#     if grad > 1e3:
#         print(f"Exploding gradient in {grad.__class__.__name__}: {grad_norm}")

def ssformer_backward_hook_fn(module, grad_input, grad_output):
    for grad in grad_input:
        if not grad is None:
            if grad.norm().item() > 1e3:
                print(f"Exploding gradient in {module.__class__.__name__}")
                breakpoint()

if __name__ == "__main__":
    model = SSformer(dim=256, image_size=8, time_emd_dim=512, heads=8, dim_head=64, dropout=0.1, num_block=4)
    con_features = torch.randn(1, 256, 8, 8)
    diff_features = torch.randn(1, 256, 8, 8)
    time_emb = torch.randn(1, 512)
    out = model(con_features, diff_features, time_emb)
    print(out.shape)