## 79 ########################################################################


# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

# pip3 install torch torchvision torchaudio




import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
max_seq_length = 10
dropout = 0.1
batch_size=64
if not torch.cuda.is_available():
    device='cpu'
    d_model = 64
    num_heads = 8
    num_layers = 6
    d_ff = 256
else:
    device='cuda:1'
    d_model = 64
    num_heads = 8
    num_layers = 6
    d_ff = 256

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
       
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # d_model 512
        # Q,K,V ([64, 100, 512]
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q)) # [64, 8, 100, 64]
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask) # [64, 8, 100, 64]

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output)) # [64, 100, 512]

        return output






class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))







class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]






class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x





class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x




class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output












##################
#
from kllm.preprocess import *
#
###################

device = torch.device(device if torch.cuda.is_available() else 'cpu')
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)


try:
    cb('\t',transformer.load_state_dict(torch.load(opjD('transformer_a4.pth'),map_location=torch.device(device)),strict=False))
except:
    print('unable to load model')


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()



display_timer=Timer(30)
display_timer.trigger()
save_timer=Timer(300)
save_timer.trigger()
train_epochs=[]
train_losses=[]
val_epochs=[]
val_losses=[]
train_index_tracker={}
val_index_tracker={}

epoch=0


start,stop=0,len(x)//2

for epoch in range(epoch,100000000):
    printr(epoch)
    optimizer.zero_grad()

    src_data,tgt_data=get_src_tgt_data(start,stop,max_seq_length,batch_size,train_index_tracker)
    src_data=src_data.to(device)
    tgt_data=tgt_data.to(device)

    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    


    if display_timer.rcheck():
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        train_epochs.append(epoch)
        train_losses.append(loss.item())

        transformer.eval()
        
        val_src_data,val_tgt_data=get_src_tgt_data(stop,len(x),max_seq_length,batch_size,val_index_tracker)
        val_src_data=val_src_data.to(device)
        val_tgt_data=val_tgt_data.to(device)


        with torch.no_grad():

            val_output = transformer(val_src_data, val_tgt_data[:, :-1])
            train_output = transformer(src_data, tgt_data[:, :-1])
        
            print(40*'_')
            for b in range(batch_size):
                if b<batch_size//2:
                    c='g'
                else:
                    val_output=train_output
                    c='b'
                ws=[]
                for i in range(max_seq_length):
                    j=int(val_src_data[b,i].detach().cpu().numpy())
                    ws.append(v2w[j])
                ws0=[' '.join(ws),'`'+c+'--']
                ws=[]
                for i in range(max_seq_length-1):
                    a=val_output[b,i,:].detach().cpu().numpy()
                    j=np.argmax(a)
                    if j in v2w:
                        ws.append(v2w[j])
                    else:
                        cE('something went wrong, key',j,'not in v2w')
                        ws.append('<unknown word>')
                ws=ws0+[' '.join(ws)]
                clp(*ws)

            print(40*'_')
            val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
            print(f"Validation Loss: {val_loss.item()}")
            val_epochs.append(epoch)
            val_losses.append(val_loss.item())

        figure(1)
        clf()
        plot(train_epochs,train_losses,'b')
        plot(val_epochs,val_losses,'r')
        figure(2)
        clf()
        plot(kys(train_index_tracker),list(train_index_tracker.values()),'b.')
        plot(kys(val_index_tracker),list(val_index_tracker.values()),'r.')
        spause()

        transformer.train()

        if save_timer.rcheck():
            print('saving weights')
            torch.save(transformer.state_dict(),opjD('transformer_a4.pth'))










## 79 ########################################################################

#EOF




