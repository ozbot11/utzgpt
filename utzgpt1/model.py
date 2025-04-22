import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
          super().__init__()
          self.heads = nn.ModuleList([
               Head(head_size, n_embd, block_size) for _ in range(num_heads)
          ])
          self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
          super().__init__()
          self.net = nn.Sequential(
               nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd)
          )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumes Block, MultiHeadAttention, FeedForward, and Head classes already defined

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=4):
        super().__init__()
        self.block_size = block_size  # <- ✅ Add this line!

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + positional embeddings
        tok_emb = self.token_embedding_table(idx)                      # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb                                          # (B, T, C)

        # Transformer blocks
        x = self.blocks(x)                                             # (B, T, C)

        # Final normalization and projection
        x = self.ln_f(x)                                               # (B, T, C)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)

        # If no targets provided, skip loss computation
        if targets is None:
            return logits, None

        # Compute cross-entropy loss
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)              # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature # scale logits for randomness
            probs = F.softmax(logits, dim=-1)       # convert to probabilities
            next_token = torch.multinomial(probs, num_samples=1)  # sample from distribution
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    # def generate(self, idx, max_new_tokens):
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -self.block_size:]
    #         logits, _ = self(idx_cond)                # (B, T, vocab_size)
    #         logits = logits[:, -1, :]                 # focus on last time step
    #         probs = F.softmax(logits, dim=-1)         # (B, vocab_size)
    #         next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
    #         idx = torch.cat((idx, next_token), dim=1) # append to sequence
    #     return idx