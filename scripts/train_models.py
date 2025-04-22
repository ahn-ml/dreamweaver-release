import torch

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset.datamodule import get_custom_dataset 
from models import TrainEngines, Models
from commons import VideoLogCallback, _create_prefix

from omegaconf import OmegaConf
from configs import CONFIG_PATH

def main(params):
    seed_everything(params.seed, workers=True)
    
    assert params

    if params.is_verbose:
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    # load dataset
    datamodule_to_load = get_custom_dataset(params) 
    # declare model to train
    model = Models[params.model](params)
    
    # training method
    method = TrainEngines[params.model](model=model, datamodule=datamodule_to_load, params=params)
    
    if params.is_logger_enabled:
        prefix_args = {
            "prefix": params.prefix,
            "seed": params.seed,
        }
        logger_name = _create_prefix(prefix_args)
        logger = pl_loggers.WandbLogger(project="vbind", name=logger_name, save_dir=params.wandb_save_dir, config=params.__dict__['_content'])

    callbacks = [LearningRateMonitor("step"), ] if params.is_logger_enabled else []
    if (not params.no_image_logger) and (params.is_logger_enabled): 
        callbacks += [VideoLogCallback()]

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="cuda",
        strategy=params.strategy,
        num_sanity_val_steps=params.num_sanity_val_steps,
        devices=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        check_val_every_n_epoch=params.val_every_n_epoch,
        callbacks=callbacks,
    )

    if params.ckpt_path:
        trainer.fit(method, datamodule_to_load.train_dataloader(), datamodule_to_load.val_dataloader(), ckpt_path=params.ckpt_path)
    else:
        trainer.fit(method, datamodule_to_load.train_dataloader(), datamodule_to_load.val_dataloader())


if __name__ == "__main__":
    conf_base = OmegaConf.load(f"{CONFIG_PATH}/meta_config.yaml")
    conf_cli = OmegaConf.from_cli()
    assert 'model' in conf_cli.keys()
    model_conf = OmegaConf.load(f"{CONFIG_PATH}/{conf_cli.model}_config.yaml")
    confs = [conf_base, model_conf, conf_cli]

    # user specific overrides
    if 'user_config' in conf_cli.keys():
        confs.append(OmegaConf.load(f"{CONFIG_PATH}/user/{conf_cli.user_config}"))

    params = OmegaConf.merge(*confs)
    main(params)