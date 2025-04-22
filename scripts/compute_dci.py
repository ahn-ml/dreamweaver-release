import torch

from pytorch_lightning import seed_everything

from dataset.datamodule import get_custom_dataset 
from models import Models

from omegaconf import OmegaConf
from configs import CONFIG_PATH
import numpy as np
from tqdm import tqdm
from commons.metrics.dci import gbt, dci
from commons.hungarian import Hungarian
from einops import rearrange
import os


def hungarian_align(true, pred):
    """
        Order for re-ordering the predicted masks
    :param true masks: B, N, 1, H, W
    :param pred masks: B, M, 1, H, W
    :return:
    """

    intersection = true[:, :, None, :, :, :] * pred[:, None, :, :, :, :]  # B, N, M, 1, H, W
    intersection = intersection.flatten(start_dim=3).sum(-1)  # B, N, M

    union = -intersection
    union += true[:, :, None, :, :, :].flatten(start_dim=3).sum(-1)
    union += pred[:, None, :, :, :, :].flatten(start_dim=3).sum(-1)

    iou = intersection / union

    orders = []
    for b in range(iou.shape[0]):
        profit_matrix = iou[b].cpu().numpy()  # N, M
        hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
        hungarian.calculate()
        results = hungarian.get_results()
        order = [j for (i,j) in results]
        orders += [order]

    orders = torch.Tensor(orders).long().to(iou.device)  # B, N

    return orders

def main(params):
    seed_everything(params.seed, workers=True)
    
    assert params

    if params.is_verbose:
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    ###################################################
    # 1. Load Dataset and Pretrained Models
    ###################################################
    # load dataset
    datamodule_to_load = get_custom_dataset(params) 
    # declare model to train
    model = Models[params.model](params)
    
    # load checkpoints to eval
    assert params.ckpt_path
    ckpt = torch.load(params.ckpt_path)
    
    # load model from ckpt
    for k,v in model.state_dict().items():
        v.copy_(ckpt['state_dict']['model.'+k])

    # freeze gradient flows
    for name, param in model.named_parameters():
        param.requires_grad = False
        # print(name, param.requires_grad)
    
    # calculate on cuda
    model.cuda()
    dl = datamodule_to_load.val_dataloader()
    
    
    ##############################################################
    # 2. Collect all slots, masks from dataset
    ##############################################################
    MAX_OBJ=3
    # collecting all slots form val dataset
    slot_collector, mask_preds_collector, factors, masks = [], [], [], []
    for _ in range(1): # sample multiple times to collect closer distribution of all frames
        for batch in tqdm(dl):
            images, label, mask = batch
            source = images[:, :params.cond_len]
            target = images[:, -params.pred_len:]
            _, _, _, slots, mask_preds = model(source.cuda(), target.cuda(), tau=0.1)
            slot_collector.append(slots.detach())
            mask_preds_collector.append(mask_preds.detach())
            factors.append(label)
            masks.append(mask)
            torch.cuda.empty_cache()

    slot_collector = torch.cat(slot_collector)
    mask_preds_collector = torch.cat(mask_preds_collector)
    factors = torch.cat(factors)
    masks = torch.cat(masks)
    
    mask_size = params.mask_size
    B, T, N, _ = slot_collector.size()
    _, _, _, H, W = source.size()

    mask_preds_collector = mask_preds_collector.transpose(-1, -2)\
                            .reshape(B, T, N, 1, mask_size, mask_size)\
                            .repeat_interleave(H // mask_size, dim=-2)\
                            .repeat_interleave(W // mask_size, dim=-1)  # B, T,num_slots, 1, H, W
    
    # use T-th slots to predict target frame
    slot_collector = slot_collector[:,-1] 
    mask_preds_collector = mask_preds_collector[:, -1]
    if 'clevr' in params.dataset:
        factors = rearrange(factors[:, -params.pred_len + params.offset], 'b (f n) -> b n f', n=MAX_OBJ) # use T+1-th (predictive code)
    else:
        factors = factors[:, -params.pred_len + params.offset]
    masks = masks[:, params.cond_len - 1].unsqueeze(2)
    
    # filter by number of objects -> we only choose 3-obj scene to score
    factors_ = []
    fids = []
    for i, f in enumerate(factors):
        if not (-1 in f[-1]):
            factors_ += [f]
            fids.append(i)
    factors_ = torch.cat(factors_, dim=0)
    
    masks_ = masks[np.array(fids)].cpu()
    mask_preds_ = mask_preds_collector[np.array(fids)].cpu()
    order = hungarian_align(masks_, mask_preds_)
    
    slots_ = torch.gather(slot_collector[np.array(fids)].cpu(), 1, order[:, :, None].expand(-1, -1, params.slot_size)).flatten(end_dim=1)  # BN, D
    factors_ = factors_.cpu()
    
    ##############################################################
    # 3. Calculate & Visualize Importance Matrix 
    ##############################################################
    del masks_, mask_preds_, slot_collector, mask_preds_collector
    torch.cuda.empty_cache()
    
    num_points = 2000
    indexs = np.arange(len(slots_))
    np.random.shuffle(indexs)
    
    _, _, acc_dyn = gbt(slots_[indexs[:num_points]].T, factors_[indexs[:num_points]][:, params.dyn_offset:].T)
    importance_matrix, _, informativeness = gbt(slots_[indexs[:num_points]].T, factors_[indexs[:num_points]].T)
    importance_matrix = np.reshape(importance_matrix, (params.num_blocks, params.slot_size // params.num_blocks, factors_.shape[-1]))  # [M, d, G]
    importance_matrix = importance_matrix.sum(axis=1)  # [M, G]
    # visualize heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(14,5))
    ax = sns.heatmap(importance_matrix.T)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(f"importance_matrix_{params.note}.png")
    # save imp_mat
    import pickle
    with open(f"importance_matrix_{params.note}.pkl","wb") as fw:
        pickle.dump(importance_matrix, fw)
    
    disentanglement, completeness = dci(importance_matrix)
    
    os.system(f"echo seed {params.seed}: ACC_DYN = {acc_dyn}> result_{params.note}_acc_dyn.log")
    os.system(f"echo seed {params.seed}: DCI Disentanglement = {disentanglement} \t Completeness = {completeness} \t Informativeness = {informativeness}> result_{params.note}.log")
    print(f'DCI Disentanglement = {disentanglement} \t Completeness = {completeness} \t Informativeness = {informativeness}')

if __name__ == "__main__":
    conf_base = OmegaConf.load(f"{CONFIG_PATH}/meta_config.yaml")
    conf_cli = OmegaConf.from_cli()
    assert 'model' in conf_cli.keys()
    model_conf = OmegaConf.load(f"{CONFIG_PATH}/{conf_cli.model}_config.yaml")
    params = OmegaConf.merge(conf_base, model_conf, conf_cli)

    main(params)