
from utilz2 import *

def strip(text):
	return re.sub(r'[^a-zA-Z\']',' ',text)

x=file_to_text(opjD('War_and_Peace.txt')).lower() #(opjh('kllm/raw.txt')).lower()
x=x.replace('- \n','')
x=x.replace('-\n','')
x=strip(x)
x=x.split(' ')
x=remove_empty(x)
print(' '.join(x))

v={}
index=1
for w in x:
	if w not in v:
		v[w]=index
		index+=1

v2w={}
for k in v:
	w=v[k]
	v2w[w]=k
#kprint(v)
#soD('war-and-peace-data.pkl',dict(x=x,v=v))

src_vocab_size=len(v)+100
tgt_vocab_size=src_vocab_size
cm(src_vocab_size)



def get_src_tgt_data(start,stop,max_seq_length,batch_size):
	src_data=[]
	tgt_data=[]
	for b in range(batch_size):
		i=randint(start,stop-2*max_seq_length)
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


a="""
COUNT  LEO  NIKOLAYEVICH  TOLSTOY  was  born 
August  28,  1828,  at  the  family  estate  of  Yasna- 
ya  Polyana,  in  the  province  of  Tula.  His  moth- 
er died  when  he  was  three  and  his  father  six 
years  later.  Placed  in  the  care  of  his  aunts,  he 
passed  many  of  his  early  years  at  Kazan,  where, 
in  1844,  after  a  preliminary  training  by  French 
tutors,  he  entered  the  university.  He  cared  lit- 
tle for  the  university  and  in  1847  withdrew  be- 
cause of  "ill-health  and  domestic  circum- 
stances." He  had,  however,  done  a  great  deal 
of  reading,  of  French,  English,  and  Russian 
novels,  the  New  Testament,  Voltaire,  and 
Hegel.  The  author  exercising  the  greatest  in- 
fluence upon  him  at  this  time  was  Rousseau; 
he  read  his  complete  works  and  for  sometime 
wore  about  his  neck  a  medallion  of  Rousseau. 
"""