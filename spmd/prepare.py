"""Fix CSV row order and build structure groups using fitting rows only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .settings import BASE,CONFIG,REPO,write_json

def read_frame(path,label_column='label'):
    data=pd.read_csv(path)
    source='sequence' if 'sequence' in data else 'seq'
    if source not in data:raise ValueError('CSV requires seq or sequence')
    data=data.copy();data['sequence']=data[source].map(lambda s:''.join(a if a in 'ACDEFGHIKLMNPQRSTVWYX' else 'X' for a in ''.join(str(s).upper().split())))
    if (data.sequence.str.len()==0).any():raise ValueError('Empty protein sequence')
    if label_column not in data:raise ValueError('Missing column: '+label_column)
    if 'row_id' not in data:data.insert(0,'row_id',np.arange(len(data)))
    return data

def prepare_data(args):
    for name in ('data','structures','runs','features','evaluation'): (BASE/name).mkdir(parents=True,exist_ok=True)
    paths=dict(train=args.train_csv,val=args.val_csv,test=args.test_csv,external=args.external_csv,pace=args.pace_csv,family=args.family_csv)
    membership={}
    for split,path in paths.items():
        if path is None:continue
        data=read_frame(path,'family_id' if split=='family' else 'label')
        if split!='family' and not set(data.label).issubset({0,1}):raise ValueError('Binary0/1 labels required')
        destination=BASE/'data'/f'{split}.csv'
        if destination.exists():
            old=pd.read_csv(destination)
            if not old.sequence.equals(data.sequence) or not old[['family_id' if split=='family' else 'label']].equals(data[['family_id' if split=='family' else 'label']]):
                raise ValueError('Existing work directory uses different data; choose a new SPMD_WORK_DIR')
        data.to_csv(destination,index=False);membership[split]=dict(source=str(Path(path).resolve()),n=len(data))
    if bool(args.family_csv)!=bool(args.family_splits):raise ValueError('Provide both --family-csv and --family-splits')
    if args.family_splits:
        import json
        write_json(BASE/'data/family_splits.json',json.loads(Path(args.family_splits).read_text()))
    write_json(BASE/'INPUTS.json',membership)

def structural_groups(states,confidence,lengths,labels,per_class=32):
    descriptors=np.zeros((len(labels),420),dtype=np.float32);quality=np.zeros(len(labels),dtype=np.float32)
    for i,length in enumerate(lengths):
        s=states[i,:length].astype(int);c=confidence[i,:length].astype(np.float32)
        if np.any((s<0)|(s>19)) or not np.isfinite(c).all():raise ValueError('Invalid3Di states/confidence')
        weights=c*(c>=CONFIG['structure_min_confidence']);quality[i]=weights.sum()/max(int(length),1)
        hist=np.bincount(s,weights=weights,minlength=20)
        transitions=np.bincount(s[:-1]*20+s[1:],weights=weights[:-1]*weights[1:],minlength=400)
        descriptors[i,:20]=np.sqrt(hist/max(hist.sum(),1e-8));descriptors[i,20:]=np.sqrt(transitions/max(transitions.sum(),1e-8))
    assignment=np.full(len(labels),-1,dtype=np.int64);centers=[];offset=0
    for label in sorted(np.unique(labels)):
        rows=np.flatnonzero((np.asarray(labels)==label)&(quality>0))
        if not len(rows):raise ValueError('Class has no confident structure rows')
        count=min(per_class,len(rows),len(np.unique(descriptors[rows],axis=0)))
        km=KMeans(n_clusters=count,random_state=42,n_init=1,max_iter=100)
        local=km.fit_predict(descriptors[rows]);assignment[rows]=local+offset
        centers.append(km.cluster_centers_);offset+=count
    counts=np.bincount(assignment[assignment>=0],minlength=offset)
    if not (counts>0).all():raise ValueError('Empty prototype')
    return dict(assignments=assignment,quality=quality,centers=np.concatenate(centers),counts=counts)

def groups(path):
    train=pd.read_csv(BASE/'data/train.csv')
    with np.load(path,allow_pickle=False) as archive:
        lengths=archive['lengths']
        if not np.array_equal(lengths,train.sequence.str.len()):raise ValueError('3Di length alignment failure')
        if 'sequences' not in archive or not np.array_equal(archive['sequences'],train.sequence):
            raise ValueError('3Di cache must include matching sequences in exact training CSV order; generate with spmd.predict_3di')
        result=structural_groups(archive['labels'],archive['confidence'],lengths,train.label.to_numpy(),CONFIG['prototypes_per_class'])
    np.savez_compressed(BASE/'structural_groups.npz',**result)
    write_json(BASE/'STRUCTURAL_MEMORY.json',dict(source_split='train',n_rows=len(train),
        n_valid=int((result['assignments']>=0).sum()),prototypes=len(result['counts']),
        counts=result['counts'].tolist(),fit_inputs='training labels and predicted3Di only; no validation/test/family inputs',seed=42))

def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('data')
    for name,default in [('train','train.csv'),('val','validation.csv'),('test','test.csv')]:p.add_argument('--'+name+'-csv',default=str(REPO/'dataset'/default))
    for name in ('external','pace','family'):p.add_argument('--'+name+'-csv')
    p.add_argument('--family-splits')
    p=sub.add_parser('groups');p.add_argument('--structure-npz',required=True)
    args=parser.parse_args()
    prepare_data(args) if args.stage=='data' else groups(args.structure_npz)

if __name__=='__main__':main()
