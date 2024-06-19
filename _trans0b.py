

# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

# pip3 install torch torchvision torchaudio




import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

if not torch.cuda.is_available():
    device="cpu"
    d_model = 64
    num_heads = 8
    num_layers = 6
    d_ff = 128
else:
    device='cuda:0'
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048


class MultiHeadAttention(nn.Module):
    def __init__(__, d_model, num_heads):
        super(MultiHeadAttention, __).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        __.d_model = d_model
        __.num_heads = num_heads
        __.d_k = d_model // num_heads
        
        __.W_q = nn.Linear(d_model, d_model) # Query transformation
        __.W_k = nn.Linear(d_model, d_model) # Key transformation
        __.W_v = nn.Linear(d_model, d_model) # Value transformation
        __.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(__, wQ, wK, wV, mask=None):

        attn_scores = torch.matmul(wQ, wK.transpose(-2, -1)) / math.sqrt(__.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_probs, wV)

        return output
        
    def split_heads(__, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, __.num_heads, __.d_k).transpose(1, 2)
        
    def combine_heads(__, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, __.d_model)
        
    def forward(__, Q, K, V, mask=None):
        # d_model 512
        # Q,K,V ([64, 100, 512]
        wQ = __.split_heads(__.W_q(Q)) # [64, 8, 100, 64]
        wK = __.split_heads(__.W_k(K))
        wV = __.split_heads(__.W_v(V))
        
        attn_output = __.scaled_dot_product_attention(wQ, wK, wV, mask) # [64, 8, 100, 64]

        output = __.W_o(__.combine_heads(attn_output)) # [64, 100, 512]

        return output






class PositionWiseFeedForward(nn.Module):
    def __init__(__, d_model, d_ff):
        super(PositionWiseFeedForward, __).__init__()
        __.fc1 = nn.Linear(d_model, d_ff)
        __.fc2 = nn.Linear(d_ff, d_model)
        __.relu = nn.ReLU()

    def forward(__, x):
        return __.fc2(__.relu(__.fc1(x)))







class PositionalEncoding(nn.Module):
    def __init__(__, d_model, max_seq_length):
        super(PositionalEncoding, __).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        __.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(__, x):
        return x + __.pe[:, :x.size(1)]






class EncoderLayer(nn.Module):
    def __init__(__, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, __).__init__()
        __.self_attn = MultiHeadAttention(d_model, num_heads)
        __.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        __.norm1 = nn.LayerNorm(d_model)
        __.norm2 = nn.LayerNorm(d_model)
        __.dropout = nn.Dropout(dropout)
        
    def forward(__, x, mask):
        attn_output = __.self_attn(x, x, x, mask)
        x = __.norm1(x + __.dropout(attn_output))
        ff_output = __.feed_forward(x)
        x = __.norm2(x + __.dropout(ff_output))
        return x





class DecoderLayer(nn.Module):
    def __init__(__, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, __).__init__()
        __.self_attn = MultiHeadAttention(d_model, num_heads)
        __.cross_attn = MultiHeadAttention(d_model, num_heads)
        __.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        __.norm1 = nn.LayerNorm(d_model)
        __.norm2 = nn.LayerNorm(d_model)
        __.norm3 = nn.LayerNorm(d_model)
        __.dropout = nn.Dropout(dropout)
        
    def forward(__, x, enc_output, src_mask, tgt_mask):
        attn_output = __.self_attn(x, x, x, tgt_mask)
        x = __.norm1(x + __.dropout(attn_output))
        attn_output = __.cross_attn(x, enc_output, enc_output, src_mask)
        x = __.norm2(x + __.dropout(attn_output))
        ff_output = __.feed_forward(x)
        x = __.norm3(x + __.dropout(ff_output))
        return x




class Transformer(nn.Module):
    def __init__(__, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, __).__init__()
        __.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        __.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        __.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        __.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        __.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        __.fc = nn.Linear(d_model, tgt_vocab_size)
        __.dropout = nn.Dropout(dropout)

    def generate_mask(__, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(__, src, tgt):
        src_mask, tgt_mask = __.generate_mask(src, tgt)
        src_embedded = __.dropout(__.positional_encoding(__.encoder_embedding(src)))
        tgt_embedded = __.dropout(__.positional_encoding(__.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in __.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in __.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = __.fc(dec_output)
        return output











max_seq_length = 10
dropout = 0.1
batch_size=16
##################
#
from kllm.preprocess import *
#
###################


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)


if False:
    cb('\t',transformer.load_state_dict(torch.load(opjD('transformer.pth')),strict=False))
# Generate random sample data
#src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
#tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)





criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()



display_timer=Timer(30)
display_timer.trigger()
save_timer=Timer(300)
train_epochs=[]
train_losses=[]
val_epochs=[]
val_losses=[]

epoch=0


start,stop=0,len(x)//2

for epoch in range(epoch,100000000):
    printr(epoch)
    optimizer.zero_grad()

    src_data,tgt_data=get_src_tgt_data(start,stop,max_seq_length,batch_size)
    src_data=src_data.to(device)
    tgt_data=tgt_data.to(device)

    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    


    if display_timer.rcheck():#not epoch%1000:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        train_epochs.append(epoch)
        train_losses.append(loss.item())

        transformer.eval()

        # Generate random sample validation data
        #val_src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
        #val_tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
        
        val_src_data,val_tgt_data=get_src_tgt_data(stop,stop+1000,max_seq_length,batch_size)
        val_src_data=val_src_data.to(device)
        val_tgt_data=val_tgt_data.to(device)

        with torch.no_grad():

            val_output = transformer(val_src_data, val_tgt_data[:, :-1])

        
            print(40*'=')
            for b in range(batch_size):
                ws=[]
                for i in range(max_seq_length):
                    j=int(val_src_data[b,i].detach().cpu().numpy())
                    ws.append(v2w[j])
                ws0=[' '.join(ws),'`g--']
                ws=[]
                for i in range(max_seq_length-1):
                    a=val_output[b,i,:].detach().cpu().numpy()
                    j=np.argmax(a)
                    ws.append(v2w[j])
                ws=ws0+[' '.join(ws)]
                clp(*ws)
            print(40*'=')
            val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
            print(f"Validation Loss: {val_loss.item()}")
            val_epochs.append(epoch)
            val_losses.append(val_loss.item())

        figure(1)
        clf()
        plot(train_epochs,train_losses,'b')
        plot(val_epochs,val_losses,'r')
        spause()

        transformer.train()

        if save_timer.rcheck():
            torch.save(transformer.state_dict(),opjD('transformer.pth'))











#EOF




