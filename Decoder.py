import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class _HeadAttention(nn.Module):
    def __init__(self,n_embed, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        block_size = int(block_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weight = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        weight = weight.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        value = self.value(x)
        out = weight @ value
        return out


class _MultiHeadAttention(nn.Module):
    def __init__(self,  n_embed, n_head, block_size, dropout, head_size):
        super().__init__()
        self.heads = nn.ModuleList([_HeadAttention(n_embed, block_size, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        out = self.dropout(x)
        return out


class _FeedForward(nn.Module):
    def __init__(self, n_embed,  dropout ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class _Block(nn.Module):
    def __init__(self, n_embed, n_head,block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        
        self.multiHead = _MultiHeadAttention(n_embed,n_head,block_size, dropout,head_size)
        self.ffwd = _FeedForward(n_embed,dropout)
        self.layrNorm1 = nn.LayerNorm(n_embed)
        self.layerNorm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multiHead(self.layrNorm1(x))
        x = x + self.ffwd(self.layerNorm2(x))
        return x
    

class BigramLanguage(nn.Module):
    
    def __init__(self, vocab_size, n_embed=384, n_head=6, block_size=256, n_layer=6, dropout = 0.2,device='cpu'):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[_Block(n_embed, n_head,block_size, dropout) for _ in range(n_layer)])
        self.layrNorm = nn.LayerNorm(n_embed)
        self.output_layer = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding(idx)
        position_embedding = self.position_embedding(torch.arange(T, device = device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.layrNorm(x)
        logits = self.output_layer(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx