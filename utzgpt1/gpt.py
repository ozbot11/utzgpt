import torch
import torch.nn.functional as F
import pandas as pd
from model import GPTLanguageModel

# -------------------- Load CSV and Extract Text --------------------
def load_csv_text(path, column='text'):
    df = pd.read_csv(path)
    return '\n'.join(df[column].astype(str).tolist())

train_text = load_csv_text('your_dataset/train.csv')
val_text   = load_csv_text('your_dataset/val.csv')
test_text  = load_csv_text('your_dataset/test.csv')

# Combine all text to build the vocabulary
all_text = train_text + val_text + test_text
chars = sorted(list(set(all_text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode datasets
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data   = torch.tensor(encode(val_text), dtype=torch.long)
test_data  = torch.tensor(encode(test_text), dtype=torch.long)

# -------------------- Hyperparameters --------------------
block_size = 512
batch_size = 64
max_iters = 100_000
eval_interval = 1000
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 768
n_head = 12
n_layer = 12

# -------------------- Batch Loader --------------------
def get_batch(split):
    data_split = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }[split]

    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -------------------- Model Setup --------------------
model = GPTLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -------------------- Eval Helper --------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -------------------- Training Loop --------------------
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, test loss {losses['test']:.4f}")
        torch.save(model.state_dict(), f'checkpoints/model_step{iter}.pt')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -------------------- Text Generation --------------------
context = torch.zeros((1, 1), dtype=torch.long).to(device)
generated = model.generate(context, max_new_tokens=500, temperature=0.9)
print("\n--- Generated Text ---\n")
print(decode(generated[0].tolist()))