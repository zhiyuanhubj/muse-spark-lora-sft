"""Convert audited token windows to Megatron arrays, with exactly one label shift."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pad-token-id',type=int,default=200018,help='Original Muse Spark tokenizer pad ID')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ready=json.loads((a.tokens/'READY.json').read_text());length=ready['max_length'];reports={}
    for split in ['positive_train','positive_validation']:
        count=ready['totals'][split+'_windows']
        ids=np.lib.format.open_memmap(a.output/(split+'-ids.npy'),mode='w+',dtype=np.int32,shape=(count,length+1))
        labels=np.lib.format.open_memmap(a.output/(split+'-labels.npy'),mode='w+',dtype=np.int32,shape=(count,length+1))
        ids[:]=a.pad_token_id;labels[:]=-100;total=0;rows=0
        with (a.tokens/(split+'.jsonl')).open() as source:
            for i,line in enumerate(source):
                row=json.loads(line);n=len(row['input_ids'])
                assert n<=length and n==len(row['labels']) and row['labels'][0]==-100
                assert row['tier'] in {'strict','hybrid'} and row['native_rerender_equal'] and row['tool_dependencies_closed']
                ids[i,:n]=row['input_ids'];labels[i,:n]=row['labels']
                total+=int((labels[i,1:]!=-100).sum());rows+=1
        assert rows==count and total==ready['totals'][split+'_targets']
        ids.flush();labels.flush();del ids,labels
        reports[split]={'rows':count,'max_sequence_length':length,'supervised_tokens_after_single_shift':total}
    (a.output/'READY.json').write_text(json.dumps({'state':'ready','sequence_length':length,
        'source_template_sha256':ready['native_template_sha256'],'splits':reports,
        'shift_policy':'tokens=ids[:L], labels=original_labels[1:L+1]; Megatron performs no extra shift',
        'no_new_truncation_or_packing':True,'control_excluded':True},indent=2))


if __name__=='__main__':main()
