# Muse Spark Open LoRA SFT

Portable text/agent-trajectory SFT for the original BF16
`meta-models/Muse-Spark-Open-816B-A42B` checkpoint. Includes complete-turn data
preparation, native Transformers FSDP2/expert parallelism + TRL/PEFT training,
adapter checks, and an experimental NeMo / Megatron Bridge recipe.

No model weights, private trajectories, credentials, scheduler scripts, machine
addresses or inference-server configuration are included.

## What has been verified

- HF/TRL: actual 16K long-window forward/backward, parameter update, save/reload,
  task-disjoint held-out evaluation, and native short-prompt generation.
- The 16K reference full epoch completed **41/41 optimizer steps** over 1,312
  training windows from 278 trajectories, with mean training loss **0.9361**.
  All 496 adapter tensors were finite and matched the trained parameter hashes.
  A fresh process restored the final adapter, evaluated **135/135 held-out
  windows** (token-weighted NLL **1.0577**), and generated the final answer `OK`
  for the fixed short instruction probe. These checks do not establish benchmark
  improvement or serving-mode correctness.
- NeMo / Megatron Bridge: the full-model 32-H200 recipe completed **3/3 real
  optimizer updates**, with finite adapter weights changed on every rank and
  only adapters trainable. Native checkpoint save and fresh-process restoration
  passed: each of the **32 rank-local parameter hashes matched** its trained
  counterpart. Held-out subset forwards completed before and after restoration.
  This is a functional framework check, **not a complete epoch**, HF export,
  generation test or serving certification.

The reference HF profile uses **32 H200 GPUs**, a **16,384-token combined
input/output window**, attention-only LoRA **r=16, alpha=32, dropout=0.05**, BF16
frozen base/experts, activation checkpointing and CPU activation offload. It needs
substantial host RAM. Other hardware layouts have not been qualified here.
Tested HF stack: PyTorch 2.14.1+cu130, Transformers 5.20.0, TRL 1.14.0, PEFT 0.21.0.

## Prepare data

Input is JSONL with `task_id`, `model_family`, `tier`, `audit_source`, `messages`
and `tools`. `reasoning_content`, assistant content/tool calls, exact tool results
and final answers are retained. Labels and audit provenance are supplied by the
caller; these scripts do not independently approve trajectories.

`export_messages.py` accepts either normalized OpenAI-style messages or manifest
rows with `trajectory_path` (ATIF export) and `request_path` (the exact original
SDK request). Missing observations are restored only from that request. Otherwise
conversion fails. This recipe is text-only.

```bash
python -m pip install -r requirements-hf.txt
python export_messages.py --input admitted-input.jsonl --output admitted.jsonl
python split_tasks.py --input admitted.jsonl --output splits
python prepare_data.py --train splits/train.jsonl \
  --validation splits/validation.jsonl --model "$MODEL_DIR" \
  --output prepared --max-length 16384 --template-date 2026-11-19
```

Use an original local BF16 model/tokenizer directory for `MODEL_DIR`. All
trajectories/teachers for one task must share a split. Only already-admitted
`strict`/`hybrid` positives enter this recipe; control examples are excluded.
`examples/messages.jsonl` is a **synthetic format fixture**, not real audited data.

Windows retain the original task prefix and complete assistant/tool transactions,
including matching tool results and at least the immediate previous transaction.
They are re-rendered with the native template and checked token-for-token.
Only newly targeted assistant tokens receive loss; overlapping history and tool
observations are masked with `-100`. Oversized transactions are withheld rather
than cut. `summary.json` reports withheld target tokens and omitted old history.
Tool-call closure does **not** prove that omitted history is semantically harmless.

## Train and verify with HF/TRL

Run the following command on each of four hosts, with distinct `NODE_RANK=0..3`
and a common rendezvous address. Model/data/output paths must be available on
all hosts. Configure your transport outside this repository.

```bash
DISTRIBUTED=(python -m torch.distributed.run --nnodes=4 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500)
"${DISTRIBUTED[@]}" train_hf.py --model "$MODEL_DIR" --tokens prepared --output-dir runs/smoke \
  --smoke --runtime-seconds 1800
```

After the smoke passes, run the same distributed command without `--smoke`,
using `--output-dir runs/train --runtime-seconds 14400`. It starts from original
BF16 weights and trains one epoch; it does not continue the smoke adapter.

Then use the same distributed launcher for a **new process**:

```bash
"${DISTRIBUTED[@]}" train_hf.py --model "$MODEL_DIR" --tokens prepared --output-dir runs/verify \
  --adapter-input runs/train/adapter-final --runtime-seconds 3600
# CPU file/hash check:
python verify_adapter.py --run runs/train
```

Read `status.json`, `metrics.jsonl`, `FINAL.json`, `ADAPTER_RELOADED.json`,
`VALIDATION.json` and `GENERATION.json`. Require the full planned step count,
finite/changed parameters, matching all-rank hashes, a new-process adapter hash
match, complete held-out coverage, and an actual generated final answer.

Adapters are backed up every five steps (two retained). The pinned Transformers
native `DistributedConfig` path does not support optimizer-state resumption;
these are **model-only** checkpoints. Replicated LoRA gradients are averaged only
after EP backward completes, before clipping/update, to preserve collective order.

Training/long validation disable cuDNN and dense-math SDPA, permit Flash/efficient
SDPA, reserve allocator headroom, and clear unused cache each step. The short
functional generation probe (at most 512 tokens) uses math SDPA, disables
checkpoint/offload and synchronizes before expert communication. It keeps
autograd enabled without backward or KV cache. This is **not** a serving-mode
or throughput certification.

## NeMo / Megatron Bridge (experimental)

Use a compatible NeMo runtime, reference image `nvcr.io/nvidia/nemo:26.10.00`,
and [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) pinned to
`ed919b61f6fa5c55051ed9d4500a51a510399a8b` with its matching MCore submodule.
Keep the runtime's PyTorch/Transformer Engine stack; do not install the HF Torch
requirements over it. If necessary, overlay only Transformers 5.20.0,
Hugging Face Hub 1.32.0, Tokenizers 0.23.2 and Safetensors 0.8.0.

```bash
python prepare_nemo_data.py --tokens prepared --output nemo-arrays
export PYTHONPATH="$PWD:$BRIDGE_ROOT/src:$BRIDGE_ROOT/3rdparty/Megatron-LM:$PYTHONPATH"
NEMO_DISTRIBUTED=(uv run --no-project --no-sync python -m torch.distributed.run \
  --nnodes=4 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500)
"${NEMO_DISTRIBUTED[@]}" train_nemo.py --model "$MODEL_DIR" --data nemo-arrays --stage train \
  --output nemo-run/train --checkpoint-dir nemo-run/checkpoints --steps 3
"${NEMO_DISTRIBUTED[@]}" train_nemo.py --model "$MODEL_DIR" --data nemo-arrays --stage reload \
  --output nemo-run/reload --checkpoint-dir nemo-run/checkpoints \
  --trained-run nemo-run/train --steps 3
```

Both commands require the distributed launcher. This is a **full-model 32-H200
functional test**, TP4/CP2/EP32/ETP1/PP1, ordinary all-to-all and full recomputation,
not the official 64-GB200/NVL72 recipe. It requires enough real windows for the
configured global batches and validation. The original BF16 base loads before
native adapter restoration. Three updates are not a full data epoch.

NeMo's fused QKV adapter differs from HF separate q/k/v adapters; this is not an
apples-to-apples convergence comparison. Experts/base/perception stay frozen;
routing remains natural. Arrays preserve the native windows, shift labels exactly
once and never introduce new truncation/packing. A local config subclass omits
PEFT's nonpersistent runtime name set from YAML; upstream sources are unchanged.
Success requires finite actual updates, changed adapter hashes, native checkpoint
save, matching per-rank hashes in a new process, and held-out loss. Native HF
adapter export and serving generation are not certified by this test.

## Metrics and structural tests

- Training loss: the trainer's assistant-only causal cross entropy.
- Held-out NLL: `sum(window_loss * supervised_next_token_count) / sum(count)`;
  report coverage and omitted-history groups separately. Perplexity is `exp(NLL)`.
- Generation checks: finite logits, consistent replicas, and the actual native
  final answer; no benchmark score is inferred from an `OK` probe.

```bash
python -m unittest discover -s tests -v
```
