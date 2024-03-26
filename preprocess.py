
from utilz2 import *

x=file_to_text(opjD('War_and_Peace.txt')).lower() #(opjh('kllm/raw.txt')).lower()
for t in ['• • •','\t','\n','.',',','√','"',';']:
	x=x.replace(t,' ')
x=x.split(' ')
x=remove_empty(x)
#print(' '.join(x))

v={}
index=1
for w in x:
	if w not in v:
		v[w]=index
		index+=1
#kprint(v)
#soD('war-and-peace-data.pkl',dict(x=x,v=v))

src_vocab_size=len(x)
tgt_vocab_size=src_vocab_size
cm(src_vocab_size)

def get_src_tgt_data(max_seq_length,batch_size):
	src_data=[]
	tgt_data=[]
	for b in range(batch_size):
		i=randint(0,len(x)-2*max_seq_length)
		src=x[i:i+max_seq_length]
		tgt=x[i+max_seq_length:i+2*max_seq_length]
		#cm(src)
		#cr(tgt)
		for j in rlen(src):

			src[j]=v[src[j]]
			tgt[j]=v[tgt[j]]
		#cb(src)
		#cg(tgt)
		src_data.append(src)
		tgt_data.append(tgt)
	src_data=torch.from_numpy(na(src_data))
	tgt_data=torch.from_numpy(na(tgt_data))
	return src_data,tgt_data


