"""Native Transformers FSDP2/EP + TRL LoRA; consumes frozen Muse tokens."""
import argparse
import hashlib
import json
import math
import os
import re
import time
import threading
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from transformers.distributed import DistributedConfig
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from probe import encode_probe


class AdapterSyncTrainer(SFTTrainer):
    """Synchronize replicated LoRA gradients after EP backward has completed.

    Native per-parameter post-accumulate hooks enqueue default-group
    collectives while checkpoint recomputation is using the EP and FSDP
    groups. Deferring the same gradient average keeps the collective order
    deterministic and avoids interleaving those communication streams.
    """
    def _sync_replicated_trainable_parameters(self, model):
        from torch.distributed.tensor import DTensor
        self._adapter_parameters = [(n, p) for n, p in sorted(model.named_parameters()) if p.requires_grad]
        assert self._adapter_parameters
        assert all('lora_' in n and not isinstance(p, DTensor) for n, p in self._adapter_parameters)
        with torch.no_grad():
            for _, parameter in self._adapter_parameters:
                torch.distributed.broadcast(parameter, src=0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            gradients = [p.grad for _, p in self._adapter_parameters]
            assert all(g is not None and not g.is_sparse for g in gradients)
            flat = torch._utils._flatten_dense_tensors(gradients)
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.AVG)
            with torch.no_grad():
                for gradient, averaged in zip(gradients, torch._utils._unflatten_dense_tensors(flat, gradients)):
                    gradient.copy_(averaged)
        return loss



class Progress(TrainerCallback):
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            event = {'time': time.time(), 'step': state.global_step, 'epoch': state.epoch,
                     'max_steps': state.max_steps, 'logs': logs or {}}
            with (self.root / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(event) + '\n')
            tmp=self.root/'status.json.tmp'
            tmp.write_text(json.dumps({'state':'training',**event},indent=2))
            tmp.replace(self.root/'status.json')

    def on_step_end(self, args, state, control, **kwargs):
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        assert not torch.backends.cuda.math_sdp_enabled()
        # All ranks receive the same wall-clock deadline; preserve an adapter
        # if it expires. Native sharding does not support optimizer resumption.
        stop = torch.tensor(int(time.time() >= self.deadline), device=args.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item():
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self,args,state,control,**kwargs):
        # Bind each saved adapter to the in-memory values on every EP rank.
        h=hashlib.sha256()
        for name,p in sorted(kwargs['model'].named_parameters()):
            if not p.requires_grad:continue
            assert 'lora_' in name and torch.isfinite(p).all()
            h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes())
        fingerprints=[None]*torch.distributed.get_world_size()
        torch.distributed.all_gather_object(fingerprints,h.hexdigest())
        assert len(set(fingerprints))==1,'Checkpoint adapters diverged'
        if state.is_world_process_zero:
            checkpoint=self.root/('checkpoint-'+str(state.global_step))
            (checkpoint/'PARAMETERS_VERIFIED.json').write_text(json.dumps({
                'step':state.global_step,'adapter_parameters_sha256':fingerprints[0],
                'all_rank_parameter_hashes_equal':True,'model_only':True,'optimizer_exact_resume_supported':False},indent=2))
        torch.distributed.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tokens', required=True)
    parser.add_argument('--deadline', type=float, help='Optional common Unix deadline for all ranks')
    parser.add_argument('--runtime-seconds',type=int,default=14400)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--adapter-input', help='Fresh-process validation of a saved adapter, no optimizer updates')
    args = parser.parse_args()
    args.deadline=args.deadline or time.time()+args.runtime_seconds
    world = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    # Avoid cuDNN's shape-specialized SDPA JIT allocations, the observed cause
    # of the previous late-epoch failure. Keep only memory-efficient kernels;
    # a dense math fallback would require quadratic attention memory at 16K.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.cuda.memory.set_per_process_memory_fraction(0.94,device=local)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root/('ATTENTION_BACKEND-rank-'+str(rank)+'.json')).write_text(json.dumps({
        'time':time.time(),'rank':rank,'cudnn_sdp':torch.backends.cuda.cudnn_sdp_enabled(),
        'math_sdp':torch.backends.cuda.math_sdp_enabled(),'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),
        'efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),
        'allocator_fraction':0.94,'torch':torch.__version__},indent=2))
    torch.set_num_threads(2)
    token_root = Path(args.tokens)
    ready = json.loads((token_root / 'READY.json').read_text())
    assert not ready['quarantined'], 'Do not silently train a partial dataset'
    assert ready['format_version'] == 'whole_transactions_v1'
    assert ready['totals']['positive_train_source_trajectories'] > 0
    assert ready['all_windows_native_rerender_equal'] and ready['all_tool_dependencies_closed']
    train = load_dataset('json', data_files=str(token_root / 'positive_train.jsonl'),
                         split='train', cache_dir=str(root/'dataset-cache'))
    validation = load_dataset('json', data_files=str(token_root / 'positive_validation.jsonl'),
                              split='train', cache_dir=str(root/'dataset-cache'))
    validation_full_history=list(validation['full_history'])
    train = train.select_columns(['input_ids', 'labels'])
    validation = validation.select_columns(['input_ids', 'labels'])
    if args.smoke:
        # Exercise the actual longest contexts and largest target spans before
        # full training, rather than accidentally validating only easy rows.
        indices=json.loads((token_root/'smoke-indices.json').read_text())[:world]
        assert len(indices)==world
        train=train.select(indices)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cfg = SFTConfig(
        output_dir=str(root),
        model_init_kwargs={'dtype': 'auto', 'local_files_only': True,
                           'attn_implementation': 'sdpa',
                           'distributed_config': DistributedConfig(fsdp_size=world, ep_size=world)},
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, num_train_epochs=1,
        max_steps=1 if args.smoke else -1,
        learning_rate=1e-4, warmup_steps=0 if args.smoke else 4, lr_scheduler_type='cosine',
        max_grad_norm=1.0, weight_decay=0.0,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False, 'early_stop': False, 'offload': True},
        max_length=ready['max_length'], packing=False,
        dataset_kwargs={'skip_prepare_dataset': True},
        remove_unused_columns=False, dataset_num_proc=4,
        loss_type='chunked_nll', logging_steps=1, report_to='none',
        save_only_model=True, save_strategy='steps', save_steps=5,
        save_total_limit=2, eval_strategy='no',torch_empty_cache_steps=1,
        dataloader_num_workers=0, dataloader_pin_memory=True,
        ddp_timeout=1800, seed=42, data_seed=42,
    )
    trainer = AdapterSyncTrainer(
        model=args.model, args=cfg, train_dataset=train,
        eval_dataset=validation, processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=r'model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'),
        callbacks=[Progress(root, args.deadline)],
    )
    trainable = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert not trainer.is_fsdp_enabled and not trainer.is_deepspeed_enabled, 'Native sharding must stay separate from Accelerate checkpoint plugins'
    assert trainable and all('lora_' in n and 'language_model' in n for n, _ in trainable)
    adapter_config = trainer.model.peft_config['default']
    adapter_config.base_model_name_or_path = 'meta-models/Muse-Spark-Open-816B-A42B'
    adapter_config.revision = '0fb02478b02565b2ce7fc57e6051c91172774ac9'
    if rank == 0:
        print('LORA_PARAMETERS', [(n, tuple(p.shape)) for n, p in trainable[:4]], flush=True)
        (root / 'TRAINING_STARTED.json').write_text(json.dumps({
            'time': time.time(), 'world_size': world, 'ep_size': world,
            'train_windows': len(train), 'validation_windows': len(validation),
            'model': args.model, 'base_frozen': True, 'rank': 16,
            'only_positive_strict_hybrid': True, 'control_excluded': True,
            'optimizer_exact_resume_supported': False,
            'smoke_only': args.smoke}, indent=2))
    if args.adapter_input:
        if not trainer.model.is_gradient_checkpointing:
            trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                'use_reentrant':False,'early_stop':False,'offload':True})
        trainer.model.enable_input_require_grads()
        # Load into the existing modules before their first lazy FSDP forward.
        # This is a fresh process/base load, not a second adapter injection.
        trainer.model.load_adapter(args.adapter_input, adapter_name='default',
                                   is_trainable=True, torch_device='cpu')
        trainer.model.set_adapter('default')
        restored=hashlib.sha256()
        for name,param in sorted(trainable):
            value=param.to_local() if hasattr(param,'to_local') else param
            restored.update(name.encode());restored.update(value.detach().float().cpu().numpy().tobytes())
        proof_path=Path(args.adapter_input).parent/'FINAL.json'
        if not proof_path.exists():
            proof_path=Path(args.adapter_input).parent/'UPDATE_SAVED.json'
        original=json.loads(proof_path.read_text())
        assert restored.hexdigest()==original['adapter_parameters_sha256'], 'Fresh-process adapter hash mismatch'
        if rank==0:
            (root/'ADAPTER_RELOADED.json').write_text(json.dumps({'time':time.time(),'fresh_process':True,
                'parameter_hash_matches':True,'sha256':restored.hexdigest()},indent=2))
        evaluate_saved(trainer, validation, tokenizer, root, world, rank, args.deadline,
                       generation=True, max_batches=2 if args.smoke else None,
                       full_history=validation_full_history)
        if rank == 0:
            (root/'FINAL.json').write_text(json.dumps({'state':'verification_complete','time':time.time(),
                'adapter_input':args.adapter_input,'fresh_process':True,'optimizer_updates':0},indent=2))
        return
    result = trainer.train()
    peak_memory = torch.tensor(torch.cuda.max_memory_allocated(), device=cfg.device)
    torch.distributed.all_reduce(peak_memory, op=torch.distributed.ReduceOp.MAX)
    learned = sum(float((p.to_local() if hasattr(p, 'to_local') else p).detach().float().abs().sum())
                  for n, p in trainable if 'lora_B' in n)
    assert math.isfinite(learned) and learned > 0, 'No nonzero learned LoRA B weights'
    fingerprint = hashlib.sha256()
    for name, param in sorted(trainable):
        local_param = param.to_local() if hasattr(param, 'to_local') else param
        fingerprint.update(name.encode())
        fingerprint.update(local_param.detach().float().cpu().numpy().tobytes())
    fingerprints = [None] * world
    torch.distributed.all_gather_object(fingerprints, fingerprint.hexdigest())
    assert len(set(fingerprints)) == 1, 'Replicated attention adapters diverged across ranks'
    trainer.save_model(str(root / 'adapter-final'))
    if rank == 0:
        (root / 'UPDATE_SAVED.json').write_text(json.dumps({
            'time': time.time(), 'step': trainer.state.global_step, 'metrics': result.metrics,
            'learned_lora_b_l1_local': learned, 'adapter_rank_hashes_identical': True,
            'adapter_parameters_sha256': fingerprints[0],
            'peak_allocated_gpu_gib': peak_memory.item() / 2**30}, indent=2))
    reloaded = False
    reload_loss = None
    if args.smoke:
        torch.distributed.barrier()
        # PEFT's inferred bare 'cuda' is interpreted as device 0 by the
        # safetensors loader. Load on CPU explicitly in this multi-GPU process.
        torch.cuda.empty_cache()
        # Restore the existing adapter, rather than injecting new modules into
        # an FSDP model after its first lazy initialization. Clear learned B
        # first so an accidental no-op load cannot pass the hash comparison.
        with torch.no_grad():
            for name, parameter in trainable:
                if 'lora_B' in name:
                    parameter.zero_()
        trainer.model.load_adapter(str(root / 'adapter-final'), adapter_name='default',
                                   is_trainable=True, torch_device='cpu')
        loaded = [(n, p) for n, p in trainer.model.named_parameters() if '.default.' in n and 'lora_' in n]
        loaded_fingerprint = hashlib.sha256()
        for name, param in sorted(loaded):
            local_param = param.to_local() if hasattr(param, 'to_local') else param
            loaded_fingerprint.update(name.encode())
            loaded_fingerprint.update(local_param.detach().float().cpu().numpy().tobytes())
        assert loaded_fingerprint.hexdigest() == fingerprints[0], 'Saved adapter failed exact reload'
        trainer.model.set_adapter('default')
        trainer.model.train()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        # Verify the actual SFT path, including native target masks and the
        # checkpointed long window, without performing another optimizer step.
        row = train[rank % len(train)]
        probe = {key: torch.tensor(row[key], device=cfg.device).unsqueeze(0)
                 for key in ['input_ids', 'labels']}
        with torch.enable_grad():
            outputs = trainer.model(**probe, use_cache=False)
        assert torch.isfinite(outputs.loss), 'Reloaded adapter produced a non-finite SFT loss'
        reload_loss = outputs.loss.detach().item()
        reloaded = True
        del outputs, probe
        evaluate_saved(trainer, validation, tokenizer, root, world, rank, args.deadline,
                       generation=True, max_batches=2,full_history=validation_full_history)
    if rank == 0:
        completed = trainer.state.global_step >= trainer.state.max_steps
        summary = {'state': 'smoke_complete' if args.smoke else ('complete' if completed else 'lease_bounded_partial'),
                   'time': time.time(), 'step': trainer.state.global_step, 'epoch': trainer.state.epoch,
                   'planned_steps': trainer.state.max_steps, 'metrics': result.metrics,
                   'adapter': str(root / 'adapter-final'), 'learned_lora_b_l1_local': learned,
                   'adapter_rank_hashes_identical': True, 'adapter_parameters_sha256': fingerprints[0],
                   'adapter_reloaded_and_forward_verified': reloaded,
                   'reload_verification_mode': 'SFT forward on real native windows with labels; existing adapter restored after zeroing B',
                   'reload_forward_loss_local': reload_loss,
                   'peak_allocated_gpu_gib': peak_memory.item() / 2**30,
                   'checkpoint_activation_offload': True,
                   'optimizer_exact_resume_supported': False}
        summary['data_version'] = ready['format_version']
        summary['data_totals'] = ready['totals']
        summary['validation_and_generation_gate_passed'] = bool(args.smoke and reloaded and (root/'VALIDATION.json').exists() and (root/'GENERATION.json').exists())
        (root / 'FINAL.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


def evaluate_saved(trainer, dataset, tokenizer, root, world, rank, deadline, generation, max_batches=None, full_history=None):
    """Deterministic loss and synchronized greedy generation on native EP.

    Keep autograd enabled to exercise the supported FSDP/EP forward path;
    no backward or optimizer step runs. Disable dropout explicitly. This is
    a functional greedy probe, not a claim of an inference-mode serving path.
    """
    trainer.model.train()
    dropouts = [m for m in trainer.model.modules() if isinstance(m, torch.nn.Dropout)]
    for module in dropouts:
        module.eval()
    sums = torch.zeros(6, dtype=torch.float64, device=trainer.args.device)
    batches = math.ceil(len(dataset)/world)
    if max_batches is not None:
        batches = min(batches, max_batches)
    for batch in range(batches):
        if time.time() >= deadline:
            raise TimeoutError('Validation deadline reached; no complete validation claim')
        index = batch*world+rank
        row = dataset[index % len(dataset)]
        probe = {k:torch.tensor(row[k],device=trainer.args.device).unsqueeze(0) for k in ['input_ids','labels']}
        count = int((probe['labels'][:,1:]!=-100).sum())
        with torch.enable_grad():
            outputs=trainer.model(**probe,use_cache=False)
        loss=outputs.loss.detach()
        assert torch.isfinite(loss), 'Nonfinite held-out SFT loss'
        if index < len(dataset):
            sums[0]+=loss.double()*count
            sums[1]+=count
            offset=2 if full_history is not None and full_history[index] else 4
            sums[offset]+=loss.double()*count;sums[offset+1]+=count
        del outputs, probe
        torch.cuda.empty_cache()
    torch.distributed.all_reduce(sums)
    if rank==0:
        (root/'VALIDATION.json').write_text(json.dumps({'time':time.time(),
            'token_weighted_loss':(sums[0]/sums[1]).item(),'target_tokens':int(sums[1].item()),
            'windows':min(len(dataset),batches*world),'total_windows':len(dataset),
            'complete':batches*world>=len(dataset),'split_by_task':True,'dropout_disabled':True,
            'full_history_token_loss':(sums[2]/sums[3]).item() if sums[3].item() else None,
            'omitted_history_token_loss':(sums[4]/sums[5]).item() if sums[5].item() else None,
            'autograd_enabled_no_backward':True},indent=2))
    if not generation:
        return
    messages=[{'role':'user','content':'Reply with exactly: OK'}]
    sequence=encode_probe(tokenizer,messages,trainer.args.device)
    # Tiny generation does not need activation checkpoint/offload. Serialize
    # pending FSDP work before switching to the EP process group's collectives.
    # This prevents the observed short-prefill stall at layer 16's count exchange.
    trainer.model.gradient_checkpointing_disable()
    generation_sync_hooks=[]
    for name,module in trainer.model.named_modules():
        if '.language_model.layers.' in name and name.endswith('.mlp.experts'):
            generation_sync_hooks.append(module.register_forward_pre_hook(
                lambda _module,_inputs:torch.cuda.synchronize()))
    assert len(generation_sync_hooks)==62
    generated=[]
    started=time.time()
    for step in range(128):
        if time.time()>=deadline:
            raise TimeoutError('Greedy generation deadline')
        assert sequence.shape[-1]<=512,'Dense math attention is restricted to the short probe'
        from torch.nn.attention import sdpa_kernel,SDPBackend
        def forward_timed_out():
            (root/('GENERATION_HANG-rank-'+str(rank)+'.json')).write_text(json.dumps({
                'time':time.time(),'step':step,'timeout_seconds':180,'rank':rank}))
            os._exit(86)
        watchdog=threading.Timer(180,forward_timed_out);watchdog.daemon=True;watchdog.start()
        try:
            with torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
                outputs=trainer.model(input_ids=sequence,use_cache=False,logits_to_keep=1)
        finally:
            watchdog.cancel()
        logits=outputs.logits[:,-1,:].detach()
        assert torch.isfinite(logits).all(), 'Nonfinite generation logits'
        token=logits.argmax(dim=-1)
        # Same prompt and fixed decoding length on every EP rank. Do not let
        # independent stopping decisions desynchronize expert collectives.
        gathered=[torch.empty_like(token) for _ in range(world)]
        torch.distributed.all_gather(gathered,token)
        assert all(torch.equal(token,t) for t in gathered), 'Replicated generation diverged'
        generated.append(int(token.item()))
        sequence=torch.cat([sequence,token[:,None]],dim=1)
        del outputs, logits
        text=tokenizer.decode(generated,skip_special_tokens=False)
        if token.item()==tokenizer.eos_token_id or ('to=user<|message|>' in text and text.endswith('<|eot|>')):
            break
    text=tokenizer.decode(generated,skip_special_tokens=False)
    assert generated and text.strip()
    user_answers=re.findall(r'to=user<\|message\|>(.*?)(?:<\|eot\|>|<\|eom\|>|$)',text,re.S)
    answer=user_answers[-1].strip() if user_answers else ''
    assert answer, 'No native user-facing final answer from the generation probe'
    # Transport/finite-logit success is distinct from following the instruction.
    if rank==0:
        (root/'GENERATION.json').write_text(json.dumps({'time':time.time(),
            'prompt':messages,'generated_tokens':generated,'text':text,'seconds':time.time()-started,
            'finite_logits':True,'all_ranks_equal':True,'max_new_tokens':128,'final_answer':answer,
            'mode':'greedy_autograd_enabled_no_backward_no_cache',
            'short_probe_attention_backend':'math; limited to at most 512 tokens',
            'gradient_checkpointing_in_probe':False,'serialized_fsdp_to_ep_in_probe':True,
            'instruction_exact_ok':answer=='OK',
            'serving_inference_mode_validated':False},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
