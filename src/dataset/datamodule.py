from typing import Optional, Callable
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import DATASET_PATH

from dataset.moving_sprites import MovingSpritesDataset
from dataset.catertex import CaterTexDataset
from torchvision import transforms

def get_custom_dataset(params):
    
    if not params.dataset_path:
        params.dataset_path = DATASET_PATH
    
    if params.custom_root:
        root = params.custom_root
    else:
        root = os.path.join(params.dataset_path, f"{params.dataset}")
        
    if params.dataset in ("moving-sprites2", "moving-sprites2-ood", "moving-sprites2-ood2", "ballet_v2a"):
        dataset_ = MovingSpritesDataset
    elif params.dataset in ('moving-clevr-easy', 'moving-clevrtex-easy', 'moving-clevr-hard', 'moving-clevr-hard-ood'):
        dataset_ = CaterTexDataset
    else:
        raise Exception("Invalid Datset")
    
    print(f"Load Dataset: {root}")
    
    collator=None
    transform=None if not (params.custum_normalize_mean and params.custum_normalize_std) else \
        transforms.Normalize(mean=params.custum_normalize_mean, std=params.custum_normalize_std)
    
    return CustomDataModule(
        dataset=dataset_,
        data_root=root,
        img_size=params.image_size,
        video_len=params.cond_len + params.pred_len,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        num_workers=params.num_workers,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        stochastic_sample=params.stochastic_sample,
        transform=transform,
        collator=collator,
    )
    

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Callable,
        data_root: str,
        img_size: int,
        video_len: int,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        stochastic_sample: bool,
        collator: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.collator = collator

        self.train_dataset = dataset(
            root=data_root,
            split="train",
            img_size=img_size,
            video_len=video_len,
            num_train_images=num_train_images,
            transform=transform,
            stochastic_sample=stochastic_sample,
        )

        self.val_dataset = dataset(
            root=data_root,
            split="test",
            img_size=img_size,
            video_len=video_len,
            num_train_images=num_val_images,
            transform=transform,
            stochastic_sample=stochastic_sample,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.collator,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
