"""Deterministically split admitted trajectories by task, across all teachers."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-modulus',type=int,default=10)
    a=p.parse_args();assert a.validation_modulus>1
    a.output.mkdir(parents=True,exist_ok=False)
    counts={'train':0,'validation':0}
    with (a.output/'train.jsonl').open('x') as train,(a.output/'validation.jsonl').open('x') as val,a.input.open() as source:
        for line in source:
            row=json.loads(line)
            if row['tier'] not in {'strict','hybrid'} or row.get('control_only'):continue
            bucket=int(hashlib.sha256(row['task_id'].encode()).hexdigest()[:8],16)%a.validation_modulus
            split='validation' if bucket==0 else 'train'
            (val if split=='validation' else train).write(json.dumps(row,ensure_ascii=False)+'\n');counts[split]+=1
    if not all(counts.values()):raise ValueError('Not enough tasks for both nonempty splits; supply your own task-disjoint split')
    print(json.dumps(counts))


if __name__=='__main__':main()
