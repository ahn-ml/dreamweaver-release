import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils
import numpy as np

from commons.utils import cosine_anneal

class DreamweaverTrain(pl.LightningModule):
    def __init__(self, model, datamodule, params):
        super().__init__()
        self.datamodule = datamodule
        self.params = params
        self.validation_step_outputs = []

        # main modules
        self.model = model
        
        # learning plans
        self.warmup_steps_pct = self.params.warmup_steps_pct
        self.decay_steps_pct = self.params.decay_steps_pct
        self.total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        # set manual optimization mode
        self.automatic_optimization = False

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # init optimizers and schedulers
        opts = self.optimizers()
        shs = self.lr_schedulers()

        global_step = self.global_step * self.params.gpus
        tau = cosine_anneal(
            global_step//3,
            self.params.tau_start,
            self.params.tau_final,
            0,
            self.warmup_steps_pct * self.total_steps)
        
        # forward model
        source = batch[:, :self.params.cond_len]
        target = batch[:, -self.params.pred_len:]
        losses, _, _, _, _ = self.model.forward(source, target, tau)
            
        # backpropagate loss
        opts[0].zero_grad(); opts[1].zero_grad(); opts[2].zero_grad()
        self.manual_backward(losses["loss"])
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_gradient_size)
        opts[0].step(); opts[1].step(); opts[2].step()
        shs[0].step(); shs[1].step(); shs[2].step()

        for k, v in losses.items():
            losses[k] = v.item()
        logs = losses
        logs.update({"tau": tau})
        self.log_dict(logs, sync_dist=True)

    def sample_images(self):
        
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[: self.params.n_samples]

        # forward model
        # preprocess for batch
        batch, _, _ = next(iter(dl))            
        source = batch[:, :self.params.cond_len]
        target = batch[:, -self.params.pred_len:]
        source = source[idx].to(self.device)
        target = target[idx].to(self.device)
        batch = source
        
        _, preds, attns, _, _ = self.model.forward(source, target, tau=0.1) # recon through dvae
        recon_tf = self.model.reconstruct_autoregressive(source) # predict target

        video_frames = []
        for t in range(batch.size(1)):
            video_t = batch[:, t, None, :, :, :]
            recon_dvae_t = preds[:, t, None, :, :, :]
            attns_t = attns[:, t, :, :, :, :]
            
            tiles_list = [video_t, recon_dvae_t, attns_t]

            # grid
            tiles = torch.cat(tiles_list, dim=1).flatten(end_dim=1)
            frame = vutils.make_grid(tiles, nrow=(len(tiles_list) - 1 + self.params.num_slots), pad_value=0.8)
            video_frames += [frame]

        video_frames = torch.stack(video_frames, dim=0).unsqueeze(0)
        samples = {"video": video_frames}

        pred_frames = []
        for t in range(self.params.pred_len):
            target_t = target[:, t, None]
            recon_tf_t = recon_tf[:, t, None]
            tiles = torch.cat([target_t, recon_tf_t], dim=1).flatten(end_dim=1)
            frame = vutils.make_grid(tiles, nrow=2, pad_value=0.8)
            pred_frames += [frame]
        pred_frames = torch.stack(pred_frames, dim=0).unsqueeze(0)
        samples["pred"] = pred_frames

        return samples

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # forward model
        batch, _, _ = batch            
        source = batch[:, :self.params.cond_len]
        target = batch[:, -self.params.pred_len:]
        losses, _, _, _, _ = self.model.forward(source, target, tau=0.1)

        for k, v in losses.items():
            losses[k] = v.item()

        logs = losses
        self.validation_step_outputs.append(logs)

    def on_validation_epoch_end(self):
        val_loss = np.array([x["loss"] for x in self.validation_step_outputs]).mean()
        logs = {
            "val_loss": val_loss
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))
        self.validation_step_outputs = []

    def configure_optimizers(self):
        dvae_opt = optim.Adam(
            self.model.dvae.parameters(),
            lr=self.params.lr_dvae,
            weight_decay=self.params.weight_decay,
        )
        enc_opt = optim.Adam(
            self.model.image_encoder.parameters(),
            lr=self.params.lr_enc,
            weight_decay=self.params.weight_decay,
        )
        dec_opt = optim.Adam(
            self.model.image_decoder.parameters(),
            lr=self.params.lr_dec,
            weight_decay=self.params.weight_decay,
        )
        
        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = self.warmup_steps_pct * self.total_steps
            decay_steps = self.decay_steps_pct * self.total_steps

            step = step * self.params.gpus

            if step <= warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        dvae_sch = optim.lr_scheduler.LambdaLR(optimizer=dvae_opt, lr_lambda=lambda x: 1)
        enc_sch = optim.lr_scheduler.LambdaLR(optimizer=enc_opt, lr_lambda=warm_and_decay_lr_scheduler)
        dec_sch = optim.lr_scheduler.LambdaLR(optimizer=dec_opt, lr_lambda=warm_and_decay_lr_scheduler)

        return [dvae_opt, enc_opt, dec_opt], [dvae_sch, enc_sch, dec_sch]
