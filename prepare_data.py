"""Whole native assistant/tool transactions; no token-level truncation.

Structural dependencies are checked exactly. Omitted older *semantic* state
cannot be proven harmless automatically and is explicitly recorded per window.
"""
import argparse
import bisect
import collections
import hashlib
import json
import os
import time
from pathlib import Path

TEMPLATE_DATE='2026-11-19'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def transactions(messages):
    first = next(i for i, m in enumerate(messages) if m['role'] == 'assistant')
    assert all(m['role'] in ('system', 'developer', 'user') for m in messages[:first])
    groups = []
    i = first
    while i < len(messages):
        start = i
        pending = {}
        # Consecutive assistant messages can be parallel calls in one decision.
        while i < len(messages) and messages[i]['role'] == 'assistant':
            for call in messages[i].get('tool_calls', []):
                assert call['id'] not in pending, 'Duplicate tool call id'
                pending[call['id']] = call['function']['name']
            i += 1
        assert i > start
        while i < len(messages) and messages[i]['role'] != 'assistant':
            m = messages[i]
            if m['role'] == 'tool':
                cid = m['tool_call_id']
                assert cid in pending, 'Orphan, duplicate or future tool observation'
                assert m.get('name', pending[cid]) == pending[cid], 'Tool name mismatch'
                pending.pop(cid)
            else:
                assert not pending, 'Unanswered tools before user/system message'
                assert m['role'] in ('user', 'system', 'developer')
            i += 1
        assert not pending, 'Missing corresponding tool observation'
        groups.append((start, i))
    return first, groups


def plan_windows(lengths, prefix, limit):
    """Greedy complete transactions plus at least the immediate previous one.

    Then fill spare space with additional contiguous older transactions.
    An oversized transaction is withheld, not shortened. If even the target
    and immediate predecessor cannot coexist, the target is withheld too.
    """
    budget = limit - prefix
    plans, withheld = [], []
    target = 0
    while target < len(lengths):
        if lengths[target] > budget:
            withheld.append((target, 'complete_transaction_exceeds_budget'))
            target += 1
            continue
        history = target - 1 if target else 0
        if target and lengths[history] + lengths[target] > budget:
            withheld.append((target, 'immediate_complete_history_does_not_fit'))
            target += 1
            continue
        end = target + 1
        used = sum(lengths[history:end])
        while end < len(lengths) and used + lengths[end] <= budget:
            used += lengths[end]
            end += 1
        while history > 0 and used + lengths[history - 1] <= budget:
            history -= 1
            used += lengths[history]
        plans.append((history, target, end))
        target = end
    return plans, withheld


def encode_source(tok, row):
    from transformers.utils.chat_template_utils import render_jinja_template
    messages = row['messages']
    first, groups = transactions(messages)
    rendered, generation = render_jinja_template(
        conversations=[messages], tools=row['tools'], chat_template=tok.chat_template,
        return_assistant_tokens_mask=True, add_generation_prompt=False,
        **tok.special_tokens_map, reasoning_effort='high', current_date=TEMPLATE_DATE)
    text, spans = rendered[0], generation[0]
    assistants = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    assert len(spans) == len(assistants), 'Native generation span per source assistant required'
    encoded = tok(text, add_special_tokens=False)
    ids = encoded['input_ids']
    labels = [-100] * len(ids)
    message_starts = {}
    for message_index, (a, b) in zip(assistants, spans):
        begin = encoded.char_to_token(a)
        end = encoded.char_to_token(b - 1) + 1
        assert begin is not None and ids[begin] == tok.convert_tokens_to_ids('<|start|>')
        message_starts[message_index] = begin
        labels[begin:end] = ids[begin:end]
    prefix = message_starts[first]
    bounds = [message_starts[a] for a, _ in groups] + [len(ids)]
    return ids, labels, groups, bounds, prefix


def prepare(tok, row, limit):
    ids, labels, groups, bounds, prefix = encode_source(tok, row)
    lengths = [b - a for a, b in zip(bounds, bounds[1:])]
    plans, withheld = plan_windows(lengths, prefix, limit)
    records, coverage = [], collections.Counter()
    for number, (history, target, end) in enumerate(plans):
        h, t, e = bounds[history], bounds[target], bounds[end]
        x = ids[:prefix] + ids[h:e]
        y = [-100] * (prefix + t - h) + labels[t:e]
        assert len(x) == len(y) <= limit
        assert x[prefix] == tok.convert_tokens_to_ids('<|start|>')
        assert all(v == -100 for v in y[:prefix + t - h])
        source_messages = row['messages']
        subset = source_messages[:groups[0][0]] + source_messages[groups[history][0]:groups[end - 1][1]]
        transactions(subset)  # All call/result IDs close within the window.
        # Re-render every window, not just a fixture: source-fragment stitching
        # must be byte-for-token equivalent to an intact native conversation.
        native = tok.apply_chat_template(subset, tools=row['tools'], tokenize=True,
            add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True,
            reasoning_effort='high', current_date=TEMPLATE_DATE)
        assert x == native['input_ids'], 'Window differs from whole-message native re-render'
        assert all(v == -100 or (m and v == token)
                   for v, m, token in zip(y, native['assistant_masks'], x))
        for pos in range(t, e):
            if labels[pos] != -100:
                coverage[pos] += 1
        target_tokens = sum(v != -100 for v in y)
        assert target_tokens
        records.append({'input_ids': x, 'labels': y,
            **{k:row[k] for k in ('task_id', 'model_family', 'tier', 'audit_source')},
            'window': number, 'history_groups': [history, target], 'target_groups': [target, end],
            'source_message_range': [groups[history][0], groups[end - 1][1]],
            'omitted_older_groups': history, 'full_history': history == 0,
            'tool_dependencies_closed': True, 'native_rerender_equal': True,
            'semantic_dependencies_verified': history == 0,
            'supervised_tokens': target_tokens})
    excluded_tokens = 0
    exclusions = []
    for index, reason in withheld:
        n = sum(v != -100 for v in labels[bounds[index]:bounds[index + 1]])
        excluded_tokens += n
        exclusions.append({'group': index, 'source_message_range': list(groups[index]),
            'reason': reason, 'transaction_tokens': lengths[index], 'supervised_tokens': n})
    assert all(v == 1 for v in coverage.values()), 'Duplicate supervision'
    original_targets = sum(v != -100 for v in labels)
    assert len(coverage) + excluded_tokens == original_targets, 'Unaccounted source targets'
    return records, {'task_id':row['task_id'], 'model_family':row['model_family'],
        'prefix_tokens':prefix, 'transactions':len(groups), 'windows':len(records),
        'source_targets':original_targets, 'trained_targets':len(coverage),
        'withheld_targets':excluded_tokens, 'exclusions':exclusions}



def main():
    global TEMPLATE_DATE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True,help='Normalized, already-admitted training messages JSONL')
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--model',required=True,help='Local original BF16 model/tokenizer directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--template-date',default='2026-11-19',help='Freeze the native template date for reproducibility')
    p.add_argument('--smoke-world-size',type=int,default=32)
    a=p.parse_args();TEMPLATE_DATE=a.template_date
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    if a.output.exists():raise FileExistsError('Prepared data must have a new output directory')
    a.output.mkdir(parents=True)
    totals=collections.Counter();cases=[];keys={};hashes={};candidates=[]
    for split,source in [('positive_train',a.train),('positive_validation',a.validation)]:
        keys[split]=set();hashes[split]=hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open() as inp,(a.output/(split+'.jsonl')).open('x') as dest:
            for line in inp:
                row=json.loads(line)
                if row['tier'] not in {'strict','hybrid'} or row.get('control_only'):
                    raise ValueError('Only already-admitted strict/hybrid positives may enter this recipe')
                records,audit=prepare(tok,row,a.max_length)
                keys[split].add(row['task_id']);cases.append({'split':split,**audit})
                totals[split+'_source_trajectories']+=1
                totals[split+'_trajectories']+=bool(records)
                totals[split+'_windows']+=len(records)
                totals[split+'_targets']+=audit['trained_targets']
                totals[split+'_withheld_targets']+=audit['withheld_targets']
                for record in records:
                    totals[split+'_full_history_windows']+=record['full_history']
                    totals[split+'_history_omitted_windows']+=not record['full_history']
                    if split=='positive_train':
                        candidates.append((len(candidates),len(record['input_ids']),record['supervised_tokens']))
                    dest.write(json.dumps(record,separators=(',',':'))+'\n')
    if keys['positive_train'] & keys['positive_validation']:
        raise ValueError('Task leakage: all teachers/trajectories for a task must share one split')
    if not totals['positive_train_windows'] or not totals['positive_validation_windows']:
        raise ValueError('Both task-disjoint splits need at least one non-withheld window')
    # Stress the longest contexts and largest target spans, not only easy rows.
    n=a.smoke_world_size;chosen=[]
    for index,_,_ in sorted(candidates,key=lambda x:x[2],reverse=True)[:n//2]+sorted(candidates,key=lambda x:x[1],reverse=True):
        if index not in chosen:chosen.append(index)
        if len(chosen)==n:break
    (a.output/'smoke-indices.json').write_text(json.dumps(chosen))
    summary={'time':time.time(),'max_length':a.max_length,'format_version':'whole_transactions_v1',
        'template_date':TEMPLATE_DATE,'totals':dict(totals),'quarantined':[],
        'source_hashes':hashes,'native_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        'all_windows_native_rerender_equal':True,'all_tool_dependencies_closed':True,
        'no_split_messages_or_tool_transactions':True,'target_coverage_accounted_exactly_once':True,
        'older_semantic_dependencies_not_proven_when_history_omitted':True,
        'original_admission_and_task_split_unchanged':True,'control_excluded':True}
    (a.output/'summary.json').write_text(json.dumps({**summary,'cases':cases},indent=2))
    (a.output/'READY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
