from typing import TypeVar

from datetime import datetime
import wandb

import torch
from pytorch_lightning import Callback

Tensor = TypeVar("torch.tensor")
def _create_prefix(args: dict):
    assert (
        args["prefix"] is not None and args["prefix"] != ""
    ), "Must specify a prefix to use W&B"
    d = datetime.today()
    date_id = f"{d.month}{d.day}{d.hour}{d.minute}{d.second}"
    before = f"{date_id}-{args['seed']}-"

    if args["prefix"] != "debug" and args["prefix"] != "NONE":
        prefix = before + args["prefix"]
        print("Assigning full prefix %s" % prefix)
    else:
        prefix = args["prefix"]

    return prefix



class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log({"images": [wandb.Image(images)]}, commit=False)



class VideoLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()                
                images_map = pl_module.sample_images()
                if type(images_map) == dict:
                    images_map = {k: wandb.Video(((v - v.min()) * 255.0 / (v.max() - v.min())).byte().cpu(),format='mp4') for k, v in images_map.items()}
                else:
                    images_map = {'videos': wandb.Video(((images_map - images_map.min()) * 255.0 / (images_map.max() - images_map.min())).byte().cpu(),format='mp4')}
                trainer.logger.experiment.log(images_map, commit=False)