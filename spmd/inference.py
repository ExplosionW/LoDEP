"""Sequence-only inference with an SPMD student and valid-residue pooling."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .data import ProteinDataset,move
from .models import Student
from .settings import STUDENT_BASE,LOCAL_ONLY

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-csv',required=True);parser.add_argument('--output-csv',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--student-base',default=STUDENT_BASE)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args();frame=pd.read_csv(args.input_csv)
    column='sequence' if 'sequence' in frame else 'seq'
    if column not in frame:raise ValueError('Input CSV requires seq or sequence')
    inputs=pd.DataFrame({'sequence':frame[column].map(lambda s:''.join(a if a in 'ACDEFGHIKLMNPQRSTVWYX' else 'X' for a in ''.join(str(s).upper().split()))),'label':0})
    if (inputs.sequence.str.len()==0).any():raise ValueError('Empty protein')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(args.student_base,local_files_only=LOCAL_ONLY)
    model=Student(args.student_base,args.checkpoint).to(device).eval()
    dataset=ProteinDataset('input',tokenizer,frame=inputs);probabilities=[]
    for batch in DataLoader(dataset,batch_size=args.batch_size,collate_fn=dataset.collate):
        batch=move(batch,device)
        with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
            logits,_=model(batch['input_ids'],batch['attention_mask'],batch['residue_mask'])
        probabilities.extend(logits.float().softmax(-1)[:,1].cpu().tolist())
    frame['probability_1']=probabilities;frame['predicted_label']=[int(p>=.5) for p in probabilities]
    path=Path(args.output_csv);path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False)

if __name__=='__main__':main()
