"""Full-checkpoint Megatron Bridge attention-LoRA functional validation.

This is a custom 32-H200 experiment, not the canonical 64-GB200 recipe or an
HF/Bridge convergence comparison. No upstream source files are modified.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
from dataclasses import fields
import time

import torch
import torch.distributed as dist
# Select each rank's device before importing TE/MCore, which may create a CUDA
# context during initialization. CPU environment checks don't set LOCAL_RANK.
if 'LOCAL_RANK' in os.environ:
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from nemo_data import TurnProvider, TurnConfig

logger=logging.getLogger(__name__)


def scalar(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:scalar(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [scalar(v) for v in value]
    return value


def adapter_signature(model):
    params=[]
    for index,chunk in enumerate(model):
        params.extend((f'{index}.{name}',p) for name,p in chunk.named_parameters() if p.requires_grad)
    assert params and all('adapter' in n or 'lora' in n.lower() for n,_ in params),'Base parameters unexpectedly trainable'
    h=hashlib.sha256();count=0
    for name,p in sorted(params):
        assert torch.isfinite(p).all(),'Nonfinite adapter weight'
        h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes());count+=p.numel()
    return {'sha256':h.hexdigest(),'local_parameters':count,'tensors':len(params),'all_finite':True}


class Trace(Callback):
    def __init__(self,root: Path,stage: str):
        self.root,self.stage=root,stage;self.initial=None
        self.training_root=None;self.steps=3

    def on_data_init_start(self,context):
        rank=dist.get_rank();sig=adapter_signature(context.model)
        if self.stage=='reload':
            expected=json.loads((self.training_root/f'adapter-rank-{rank}.json').read_text())
            assert sig['sha256']==expected['sha256'],'Native fresh-process adapter reload mismatch'
            passed=torch.tensor(1,device=torch.cuda.current_device());dist.all_reduce(passed,op=dist.ReduceOp.MIN)
            if rank==0:
                (self.root/'ADAPTER_RELOADED.json').write_text(json.dumps({'time':time.time(),
                    'all_32_rank_hashes_match':bool(passed.item()),'fresh_process':True,
                    'base_initialized_from_original_hf_before_adapter_load':True},indent=2))

    def on_train_start(self,context):
        self.initial=adapter_signature(context.model)

    def on_train_step_end(self,context):
        assert not context.skipped_iter,'Skipped optimizer update'
        losses=scalar(context.loss_dict or {})
        for value in losses.values():
            values=value if isinstance(value,list) else [value]
            assert all(math.isfinite(float(x)) for x in values),'Nonfinite training loss'
        grad=scalar(context.grad_norm)
        if grad is not None:assert math.isfinite(float(grad))
        if dist.get_rank()==0:
            d={'state':'training','time':time.time(),'step':context.state.train_state.step+1,
                'planned_steps':self.steps,'loss':losses,'grad_norm':grad,'skipped_iter':False}
            with (self.root/'metrics.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
            (self.root/'status.json').write_text(json.dumps(d,indent=2))

    def on_train_end(self,context):
        rank=dist.get_rank();sig=adapter_signature(context.model)
        changed=torch.tensor(int(sig['sha256']!=self.initial['sha256']),device=torch.cuda.current_device())
        dist.all_reduce(changed,op=dist.ReduceOp.SUM)
        assert changed.item()>0,'No adapter updates'
        (self.root/f'adapter-rank-{rank}.json').write_text(json.dumps(sig,indent=2))
        dist.barrier()
        if rank==0:
            (self.root/'UPDATES_VERIFIED.json').write_text(json.dumps({'time':time.time(),
                'step':context.state.train_state.step,'planned_steps':self.steps,'changed_ranks':int(changed.item()),
                'all_adapter_weights_finite':True,'only_adapters_trainable':True},indent=2))

    def on_eval_end(self,context):
        if dist.get_rank()==0:
            losses=scalar(context.total_loss_dict)
            assert losses and all(math.isfinite(float(v)) and float(v)>0 for v in losses.values()),'Invalid held-out loss'
            d={'time':time.time(),'loss':losses,
                'evaluation_is_bounded_subset':True,'full_135_window_evaluation':False}
            (self.root/'VALIDATION.json').write_text(json.dumps(d,indent=2))


def configure(model_path: str,root: Path,stage: str,*,data: str,checkpoint: str,steps: int):
    from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
    # Permit only the audited local dataset module in our own native checkpoint
    # configuration, so fresh-process checkpoint restoration can reconstruct it.
    register_allowed_target_prefix('nemo_data.')
    module=importlib.import_module('megatron.bridge.recipes.muse_spark_open.gb200.muse_spark_open')
    # Use the maintained library recipe as the starting point, with a local HF
    # source. This assignment is process-local; upstream files stay unchanged.
    module._MODEL_ID=model_path
    cfg=module.muse_spark_open_peft_64gpu_gb200_bf16_config()
    cfg=TurnConfig(**{f.name:getattr(cfg,f.name) for f in fields(cfg) if f.init})
    assert cfg.model.transformer.num_layers==124
    assert cfg.model.hybrid_layer_pattern.replace('|','')=='*E'*62
    overrides={'tensor_model_parallel_size':4,'pipeline_model_parallel_size':1,
        'expert_model_parallel_size':32,'expert_tensor_parallel_size':1,'context_parallel_size':2,
        'sequence_parallel':True,'cp_comm_type':'a2a','seq_length':16384,
        'moe_token_dispatcher_type':'alltoall','moe_flex_dispatcher_backend':None,
        'moe_permute_fusion_into_hybridep':False,'moe_hybridep_pad_uneven_dispatch_inputs':False,
        'recompute_granularity':'full','recompute_method':'uniform','recompute_num_layers':1}
    for key,value in overrides.items():
        assert hasattr(cfg.model,key),'Unsupported model config field: '+key
        setattr(cfg.model,key,value)
    if hasattr(cfg.model,'moe_router_bias_update_rate'):
        cfg.model.moe_router_bias_update_rate=0.0
    cfg.peft.target_modules=['linear_qkv','linear_proj']
    cfg.peft.dim=16;cfg.peft.alpha=32;cfg.peft.dropout=0.05
    cfg.dataset=TurnProvider(root=data,seq_length=16384)
    cfg.train.num_epochs=None
    cfg.train.train_iters=steps;cfg.train.global_batch_size=32;cfg.train.micro_batch_size=1
    cfg.optimizer.lr=1e-4;cfg.optimizer.min_lr=0.0;cfg.optimizer.weight_decay=0.0
    cfg.scheduler.lr_warmup_iters=0;cfg.scheduler.lr_decay_iters=steps
    cfg.rng.seed=42
    cfg.validation.eval_interval=steps;cfg.validation.eval_iters=1
    cfg.validation.skip_train=stage=='reload'
    cfg.checkpoint.pretrained_checkpoint=model_path
    cfg.checkpoint.hf_source_path=model_path
    cfg.checkpoint.save=checkpoint if stage=='train' else None
    cfg.checkpoint.load=checkpoint if stage=='reload' else None
    cfg.checkpoint.finetune=stage=='train'
    # This gate verifies adapter weights and a held-out forward, not exact Adam
    # or RNG resumption. A fresh verification process performs no updates.
    if stage=='reload':
        cfg.checkpoint.load_optim=False
        cfg.checkpoint.load_rng=False
    cfg.checkpoint.save_interval=steps;cfg.checkpoint.async_save=False
    cfg.checkpoint.also_save_hf_checkpoint=False
    cfg.logger.tensorboard_dir=None;cfg.logger.log_interval=1
    cfg.logger.save_config_filepath=str(root/'resolved.yaml')
    cfg.env_vars={k:v for k,v in cfg.env_vars.items() if k not in
        ['NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN','NVLINK_DOMAIN_SIZE','USE_MNNVL']}
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--stage',choices=['train','reload'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--trained-run',help='Training output containing per-rank parameter signatures')
    p.add_argument('--steps',type=int,default=3)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    cfg=configure(args.model,root,args.stage,data=args.data,checkpoint=args.checkpoint_dir,steps=args.steps)
    trace=Trace(root,args.stage);trace.steps=args.steps
    if args.stage=='reload':
        assert args.trained_run,'--trained-run is required for fresh reload comparison'
        trace.training_root=Path(args.trained_run)
    if int(os.environ.get('RANK','0'))==0:
        (root/'PLAN.json').write_text(json.dumps({'time':time.time(),'stage':args.stage,
            'full_model_62_layers':True,'world_size':32,'tp':4,'ep':32,'cp':2,'etp':1,
            'dispatcher':'alltoall','routing':'natural; no forced balancing; frozen correction bias',
            'full_recompute':True,'sequence_length':16384,'global_batch_size':32,'steps':args.steps,
            'rank':16,'alpha':32,'targets':['linear_qkv','linear_proj'],
            'fused_qkv_adapter_parameterization_differs_from_hf_separate_q_k_v':True,
            'framework_functional_test_not_canonical_gb200_or_convergence_comparison':True},indent=2))
    finetune(cfg,forward_step,callbacks=[trace])
    if int(os.environ.get('RANK','0'))==0:
        assert (root/'VALIDATION.json').exists(),'Held-out validation did not execute'
        if args.stage=='train':
            proof=json.loads((root/'UPDATES_VERIFIED.json').read_text())
            assert proof['step']==args.steps and proof['all_adapter_weights_finite']
        else:
            assert json.loads((root/'ADAPTER_RELOADED.json').read_text())['all_32_rank_hashes_match']
        (root/'FINAL.json').write_text(json.dumps({'state':'functional_train_complete' if args.stage=='train' else 'native_reload_verified',
            'time':time.time(),'optimizer_updates':args.steps if args.stage=='train' else 0,
            'not_a_complete_epoch':True,'hf_export_or_serving_generation_validated':False},indent=2))
        logger.info('NeMo %s completed',args.stage)


if __name__=='__main__':main()
