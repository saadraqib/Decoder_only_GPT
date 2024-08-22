import torch
import torch.nn as nn
from torch.nn import functional as F
import Decoder

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def read_plainText(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def vocab(txt_path, show_chars = False):

    text = read_plainText(txt_path)

    chars = sorted(list(set(text)))
    return chars

def text_encode(text):
    map_char2indx = {index:char for char,index in enumerate(chars)}
    encode = [map_char2indx[c] for c in text]
    return encode

def text_decode(array):
    map_indx2char = {char:index for char,index in enumerate(chars)}
    decode = ''.join([map_indx2char[val] for val in array])
    return decode

def batch_retrieval(data):
    processing_unit = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = train if data == "train" else val
    # print("compare: ",len(data),block_size )
    random_batch = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in random_batch])
    y = torch.stack([data[i+1: i+block_size+1] for i in random_batch])
    x, y = x.to(processing_unit), y.to(processing_unit)
    return x, y

def data_split(data, train_p=0.9):
    n = int(train_p*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data




print(device)
path = r"poem.txt"
plain_text = read_plainText(path)
chars = vocab(path)
vocab_size = len(chars)

data = torch.tensor(text_encode(plain_text), dtype=torch.long, device = device)

train, val = data_split(data, 0.9)

x, y = batch_retrieval(train)

model = Decoder.BigramLanguage(vocab_size = vocab_size, n_embed=n_embd)
model = model.to(device=device)
logits, loss = model(x, y)


print(logits.shape)
print(logits)



optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for data in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = batch_retrieval(data)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[data] = losses.mean()
    model.train()
    return out

for step in range(5):
    if step % eval_interval==0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = batch_retrieval('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)   
    loss.backward()
    optimizer.step()

input_text = 'Pacific'
encoded_input = torch.tensor(text_encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
generated_text = model.generate(encoded_input, max_new_tokens=500)[0].tolist()
print(text_decode(generated_text))

# idx = torch.zeros((1,1), dtype=torch.long, device= device)
# generated_text = model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()
# print(text_decode(generated_text))