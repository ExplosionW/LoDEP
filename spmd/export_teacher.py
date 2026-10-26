"""Export original LoDEP teacher targets with valid-residue feature pooling."""
import torch.distributed as dist
from transformers import AutoTokenizer
from .evaluation import extract,summarize_binary,family_probes
from .models import OriginalTeacher
from .settings import BASE,TEACHER_BASE,LOCAL_ONLY,setup

def main():
    rank,world,device=setup()
    tokenizer=AutoTokenizer.from_pretrained(TEACHER_BASE,local_files_only=LOCAL_ONLY)
    model=OriginalTeacher().to(device).eval()
    for split in ('train','val','test','external','pace','family'):
        if (BASE/'data'/f'{split}.csv').exists():extract(model,tokenizer,'lodep_650m_teacher',split,device,rank,world,batch_size=8)
    if rank==0:summarize_binary('lodep_650m_teacher');family_probes('lodep_650m_teacher')
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
