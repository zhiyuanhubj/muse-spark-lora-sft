"""Native prompt construction with an explicit integer-token interface."""
import torch


def encode_probe(tokenizer, messages, device):
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,
        add_generation_prompt=True,reasoning_effort='low',current_date='2026-11-19')
    ids=encoded['input_ids']
    assert isinstance(ids,list) and ids and all(isinstance(x,int) for x in ids)
    return torch.tensor([ids],dtype=torch.long,device=device)
