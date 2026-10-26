"""ProstT5+official CNN predictions, full-length training sequences only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from transformers import T5EncoderModel,T5Tokenizer
from .settings import BASE,LOCAL_ONLY,setup,write_json

class ProstCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier=nn.Sequential(nn.Conv2d(1024,32,kernel_size=(7,1),padding=(3,0)),nn.ReLU(),nn.Dropout(0.),nn.Conv2d(32,20,kernel_size=(7,1),padding=(3,0)))
    def forward(self,x):return self.classifier(x.permute(0,2,1).unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--model',default='Rostlab/ProstT5');parser.add_argument('--cnn-checkpoint',required=True);parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args();rank,world,device=setup()
    final=BASE/'structures/train.npz'
    if final.exists():dist.destroy_process_group();return
    frame=pd.read_csv(BASE/'data/train.csv');sequences=frame.sequence.tolist()
    tokenizer=T5Tokenizer.from_pretrained(args.model,do_lower_case=False,legacy=True,local_files_only=LOCAL_ONLY)
    encoder=T5EncoderModel.from_pretrained(args.model,torch_dtype=torch.float16,local_files_only=LOCAL_ONLY).to(device).eval()
    cnn=ProstCNN();cnn.load_state_dict(torch.load(args.cnn_checkpoint,map_location='cpu',weights_only=True)['state_dict']);cnn.to(device=device,dtype=torch.float16).eval()
    indices=sorted(range(rank,len(sequences),world),key=lambda i:-len(sequences[i]))
    width=max(map(len,sequences));labels=np.full((len(indices),width),-100,dtype=np.int16);confidence=np.zeros_like(labels,dtype=np.float16)
    start=0
    while start<len(indices):
        size=max(1,min(args.batch_size,4096//(len(sequences[indices[start]])+2)));rows=indices[start:start+size];texts=[sequences[i] for i in rows]
        batch=tokenizer(['<AA2fold> '+' '.join(s) for s in texts],return_tensors='pt',padding=True,truncation=False).to(device)
        if batch.attention_mask.sum(1).tolist()!=[len(s)+2 for s in texts]:raise ValueError('ProstT5 token/residue alignment failure')
        hidden=encoder(**batch).last_hidden_state;keep=batch.attention_mask.clone()
        for j,s in enumerate(texts):keep[j,len(s)+1]=0
        probability=cnn((hidden*keep.unsqueeze(-1))[:,1:]).float().softmax(1)
        if not torch.isfinite(probability).all():raise FloatingPointError('Nonfinite3Di prediction')
        cert,pred=probability.max(1)
        for j,s in enumerate(texts):labels[start+j,:len(s)]=pred[j,:len(s)].cpu().numpy();confidence[start+j,:len(s)]=cert[j,:len(s)].cpu().numpy()
        print('3Di rank',rank,'rows',start,'/',len(indices),flush=True);start+=len(rows)
    np.savez(BASE/'structures'/f'train_rank{rank}.npz',ids=np.asarray(indices),labels=labels,confidence=confidence)
    dist.barrier()
    if rank==0:
        pieces=[np.load(BASE/'structures'/f'train_rank{r}.npz') for r in range(world)]
        ids=np.concatenate([p['ids'] for p in pieces]);order=np.argsort(ids)
        if not np.array_equal(ids[order],np.arange(len(sequences))):raise ValueError('3Di row coverage failure')
        np.savez_compressed(final,labels=np.concatenate([p['labels'] for p in pieces])[order],confidence=np.concatenate([p['confidence'] for p in pieces])[order],lengths=np.asarray(list(map(len,sequences))),sequences=np.asarray(sequences))
        for p in pieces:p.close()
        write_json(BASE/'structures/SOURCE.json',dict(model=args.model,cnn_checkpoint=str(Path(args.cnn_checkpoint).resolve()),scope='training rows only; full length; predicted structure, not coordinates'))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
