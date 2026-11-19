"""Megatron dataset provider for audited whole-transaction SFT windows."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from megatron.bridge.data.base import DatasetProvider, DatasetBuildContext
from megatron.bridge.training.config import ConfigContainer


@dataclass(kw_only=True)
class TurnConfig(ConfigContainer):
    """Omit PEFT's explicitly nonpersistent runtime parameter-name set.

    The pinned Container serializer includes init=False runtime fields. Its
    from_yaml path passes YAML sets to OmegaConf before sanitizing them, which
    prevents native checkpoint configuration restoration. The live PEFT set is
    untouched and is rebuilt from the real model by PEFT.set_params_to_save().
    """
    def to_dict(self,**section_overrides):
        data=super().to_dict(**section_overrides)
        if isinstance(data.get('peft'),dict):
            data['peft'].pop('params_to_save',None)
        return data


class TurnDataset(Dataset):
    def __init__(self, root: str, split: str, seq_length: int):
        self.ids=np.load(Path(root)/(split+'-ids.npy'),mmap_mode='r')
        self.labels=np.load(Path(root)/(split+'-labels.npy'),mmap_mode='r')
        self.seq_length=seq_length
        assert self.ids.shape==self.labels.shape and self.ids.shape[1]==seq_length+1

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self,index):
        ids=torch.from_numpy(np.array(self.ids[index,:self.seq_length],dtype=np.int64))
        labels=torch.from_numpy(np.array(self.labels[index,1:self.seq_length+1],dtype=np.int64))
        mask=(labels!=-100).float()
        labels.masked_fill_(labels==-100,0)
        return {'tokens':ids,'labels':labels,'loss_mask':mask,
            'position_ids':torch.arange(self.seq_length,dtype=torch.long)}


@dataclass
class TurnProvider(DatasetProvider):
    root: str='data'
    seq_length: int=16384
    random_seed: int=42
    seed: int=42
    num_workers: int=0
    persistent_workers: bool=False
    dataloader_type: str='single'
    do_train: bool=True
    do_validation: bool=True
    do_test: bool=False
    skip_getting_attention_mask_from_dataset: bool=True

    def build_datasets(self, context: DatasetBuildContext):
        train=TurnDataset(self.root,'positive_train',self.seq_length)
        val=TurnDataset(self.root,'positive_validation',self.seq_length)
        assert len(train)>=context.train_samples and len(val)>=context.valid_samples
        return train,val,None
