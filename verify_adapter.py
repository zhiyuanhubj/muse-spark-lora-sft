"""Verify a real smoke adapter and its existing distributed reload receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open

torch.set_num_threads(2)

p = argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
args = p.parse_args()
run=args.run.resolve()
root=run
final = json.loads((run / 'FINAL.json').read_text())
if final.get('state')=='smoke_complete':
    assert final['state'] == 'smoke_complete' and final['step'] == 1
    assert final['adapter_reloaded_and_forward_verified']
else:
    assert final['state'] in {'complete', 'lease_bounded_partial'} and final['step'] > 0
assert final['adapter_rank_hashes_identical']
adapter = run / 'adapter-final'
config = json.loads((adapter / 'adapter_config.json').read_text())
assert config['peft_type'] == 'LORA' and config['r'] == 16
assert config['base_model_name_or_path'] == 'meta-models/Muse-Spark-Open-816B-A42B'
assert config['revision'] == '0fb02478b02565b2ce7fc57e6051c91172774ac9'
weights = adapter / 'adapter_model.safetensors'
count = parameters = 0
nonzero_b = 0
parameter_hash = hashlib.sha256()
with safe_open(weights, framework='pt', device='cpu') as tensors:
    for key in sorted(tensors.keys()):
        assert 'lora_' in key and 'language_model' in key
        value = tensors.get_tensor(key)
        canonical = key.replace('.lora_A.', '.lora_A.default.').replace('.lora_B.', '.lora_B.default.')
        parameter_hash.update(canonical.encode())
        parameter_hash.update(value.float().numpy().tobytes())
        assert torch.isfinite(value).all()
        count += 1
        parameters += value.numel()
        if 'lora_B' in key:
            nonzero_b += int(torch.count_nonzero(value))
assert count == 62 * 4 * 2 and nonzero_b > 0
assert parameter_hash.hexdigest() == final['adapter_parameters_sha256'], 'Saved tensor values differ from trained values'
sha = hashlib.sha256()
with weights.open('rb') as handle:
    while block := handle.read(8 * 1024 * 1024):
        sha.update(block)
proof = {'time': time.time(), 'run': str(run), 'adapter': str(adapter),
         'tensors': count, 'parameters': parameters, 'nonzero_lora_b_values': nonzero_b,
         'bytes': weights.stat().st_size, 'sha256': sha.hexdigest(), 'all_weights_finite': True,
         'parameter_sha256_matches_trained': True,
         'adapter_restored_for_sft_forward_verified': final['adapter_reloaded_and_forward_verified'],
         'training': final}
(root/'ADAPTER_VERIFIED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof),flush=True)
