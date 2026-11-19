"""Normalize already-admitted messages or ATIF + original request records.

No admission is inferred. Missing tool results are restored only from the exact
provided original request, and otherwise fail. Thinking is never fabricated.
"""
import argparse
import hashlib
import json
from pathlib import Path


def canonical_messages(messages):
    result=[]
    for item in messages:
        m=dict(item)
        if m['role'] not in {'system','developer','user','assistant','tool'}:
            raise ValueError('Unsupported message role')
        if m.get('content') is None:m['content']=''
        if not isinstance(m['content'],str):
            raise ValueError('This text-only recipe requires string message content')
        if m.get('reasoning_content') is not None and not isinstance(m['reasoning_content'],str):
            raise ValueError('reasoning_content must be text when present')
        if m.get('tool_calls'):
            calls=[]
            for original in m['tool_calls']:
                call=dict(original);call['function']=dict(call['function'])
                args=call['function'].get('arguments',{})
                if isinstance(args,str):args=json.loads(args)
                if not isinstance(args,dict):raise ValueError('Tool arguments must be JSON objects')
                call['function']['arguments']=args;calls.append(call)
            m['tool_calls']=calls
        result.append(m)
    return result


def from_atif(trajectory,request):
    if not request.get('tools'):raise ValueError('Original tool schemas are required for ATIF tool traces')
    source_tools={m['tool_call_id']:m for m in request['messages']
        if m['role']=='tool' and m.get('tool_call_id')}
    messages=[{'role':m['role'],'content':m['content']} for m in request['messages']
        if m['role'] in {'system','developer'}]
    restored=0;names={}
    for step in trajectory['steps']:
        role=step['source']
        if role in {'user','system'}:
            messages.append({'role':role,'content':step.get('message') or ''})
        elif role=='agent':
            m={'role':'assistant','content':step.get('message') or ''}
            if step.get('reasoning_content'):m['reasoning_content']=step['reasoning_content']
            calls=step.get('tool_calls') or []
            if calls:
                m['tool_calls']=[]
                for call in calls:
                    cid,name=call['tool_call_id'],call['function_name'];names[cid]=name
                    args=call.get('arguments') or {}
                    if isinstance(args,str):args=json.loads(args)
                    if not isinstance(args,dict):raise ValueError('Tool arguments must be objects')
                    m['tool_calls'].append({'id':cid,'type':'function',
                        'function':{'name':name,'arguments':args}})
            if m['content'] or m.get('reasoning_content') or calls:messages.append(m)
        else:raise ValueError('Unsupported ATIF step source: '+role)
        observations={o['source_call_id']:o for o in (step.get('observation') or {}).get('results',[])}
        for call in step.get('tool_calls') or []:
            cid=call['tool_call_id']
            if cid in observations:content=observations[cid].get('content')
            elif cid in source_tools:
                content=source_tools[cid].get('content');restored+=1
            else:raise ValueError('Missing original tool observation: '+cid)
            if not isinstance(content,str):content=json.dumps(content,ensure_ascii=False)
            messages.append({'role':'tool','name':names[cid],'tool_call_id':cid,'content':content})
    return canonical_messages(messages),request['tools'],restored


def normalize(row,base):
    for field in ['task_id','model_family','tier','audit_source']:
        if not row.get(field):raise ValueError('Missing supplied admission metadata: '+field)
    if 'messages' in row:
        messages=canonical_messages(row['messages']);tools=row.get('tools',[]);restored=0
    else:
        trace=json.loads((base/row['trajectory_path']).read_text())
        request=json.loads((base/row['request_path']).read_text())
        messages,tools,restored=from_atif(trace.get('trajectory',trace),request)
    # Import only the structural validator, not any model weights.
    from prepare_data import transactions
    transactions(messages)
    return {**{k:row[k] for k in ['task_id','model_family','tier','audit_source']},
        'control_only':bool(row.get('control_only',False) or row['tier'] in {'control','unsuccessful_access_control'}),
        'messages':messages,'tools':tools,'restored_tool_observations':restored,
        'messages_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Refusing to replace an existing export')
    with a.input.open() as source,a.output.open('x') as out:
        for line in source:
            row=normalize(json.loads(line),a.input.parent)
            out.write(json.dumps(row,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
