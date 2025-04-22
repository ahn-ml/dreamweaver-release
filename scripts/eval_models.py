import torch

from pytorch_lightning import seed_everything
from torchvision import utils as vutils

from dataset.datamodule import get_custom_dataset 
from models import Models

from omegaconf import OmegaConf
from configs import CONFIG_PATH
from PIL import Image as im 
import numpy as np
from fast_pytorch_kmeans import KMeans
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

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
    # 2. Collect all slots from dataset & learn block labelers
    ##############################################################
    
    # collecting all slots form val dataset
    slot_collector = []
    for _ in range(1): # sample multiple times to collect closer distribution of all frames
        for batch in tqdm(dl):
            batch = batch[0]
            slots = model.encode(batch.cuda())
            slot_collector.append(slots.detach())

    slot_collector = torch.cat(slot_collector)
    B, T, N, D = slot_collector.size()
    slot_collector = slot_collector.view(B*T*N, D)
    
    # learn block labeler
    block_labeler = []
    block_size = int(model.d_slot/model.num_blocks) if type(model).__name__=="PSBBlockAE" else int(model.slot_size/model.num_blocks)
    slots_ = rearrange(slot_collector, 'b (m d) -> b m d', m=model.num_blocks)
    for M in range(model.num_blocks):
        # extract M-th block from slot
        X = slots_[:, M]
        
        labeler = KMeans(n_clusters=params.n_clusters, mode='euclidean', verbose=1)
        labeler.fit(X)
        block_labeler.append(labeler)
    
    
    #####################################################################
    # 3. Pick some samples from dataset and Render frames
    #####################################################################
    
    # sample to infer block labels
    perm = torch.randperm(params.batch_size)
    idx = perm[: params.n_samples]
    
    # preprocess for batch
    batch, _, _ = next(iter(dl))
    source = batch[:, :params.pred_len]
    target = batch[:, -params.pred_len:]
    source = source[idx].cuda()
    target = target[idx].cuda()
    samples = source
    
    B, T, C, H, W = source.size()
    _, preds, attns, all_slots, attns_raw = model.forward(source, target, tau=0.1) # recon through dvae
    recon_tf = model.reconstruct_autoregressive(source) # predict target
    
    # render frames
    video_frames = []
    for t in range(samples.size(1)):
        video_t = samples[:, t, None, :, :, :]
        recon_dvae_t = preds[:, t, None, :, :, :]
        attns_t = attns[:, t, :, :, :, :]
        
        tiles_list = [video_t, recon_dvae_t, attns_t]

        # grid
        tiles = torch.cat(tiles_list, dim=1).flatten(end_dim=1)
        frame = vutils.make_grid(tiles, nrow=(len(tiles_list) - 1 + params.num_slots), pad_value=0.8)
        video_frames += [frame]


    pred_frames = []
    for t in range(params.pred_len):
        target_t = target[:, t, None]
        recon_tf_t = recon_tf[:, t, None]
        tiles = torch.cat([target_t, recon_tf_t], dim=1).flatten(end_dim=1)
        frame = vutils.make_grid(tiles, nrow=2, pad_value=0.8)
        pred_frames += [frame]
    
    def normalize_img(frame):
        _frame = frame.cpu().permute(1,2,0).numpy()
        _min,_max = np.amin(_frame), np.amax(_frame)
        _frame = np.uint8((_frame - _min) * 255.0 / (_max - _min))
        data = im.fromarray(_frame)
        return data 
    
    # save rendered frames
    datas, pred_datas = [],[]
    for i in range(len(video_frames)):
        data = normalize_img(video_frames[i])
        pred_data = normalize_img(pred_frames[i])
        datas.append(data)
        pred_datas.append(pred_data)
    datas[0].save('frames.gif', save_all=True, append_images=datas[1:], duration=500, loop=0)
    pred_datas[0].save('pred_frames.gif', save_all=True, append_images=pred_datas[1:], duration=500, loop=0)
    
    # compute pred_attns
    # pred_attns = attns_raw\
    #     .transpose(-1, -2)\
    #     .reshape(B, T, N, 1, 32, 32)\
    #     .repeat_interleave(H // 32, dim=-2)\
    #     .repeat_interleave(W // 32, dim=-1)  # B, T,num_slots, 1, H, W
    # pred_attns = recon_tf.unsqueeze(2) * pred_attns[:, -1:] + (1. - pred_attns[:, -1:])  # B, T, num_slots, C, H, W
    
    #####################################################################
    # 4. Categorize samples by Block Clustering
    #####################################################################
    block_collector = {M:{i: [] for i in range(params.n_clusters)} for M in range(model.num_blocks)}
    # infer sample's blocks
    for i, sample in enumerate(all_slots):
        # print(f"<sample {i}>")
        for t, slots in enumerate(sample):
            if (t < len(sample)-1): continue
            for N, slot in enumerate(sample[t]):
                all_labels = []
                blockslot = rearrange(slot, '(m d) -> m d', m=model.num_blocks)
                for M in range(model.num_blocks):
                    label = block_labeler[M].predict(blockslot[M].unsqueeze(0))
                    block_collector[M][int(label[0].cpu())].append((i, t, N))
                    all_labels.append(label.cpu())
                # print(f"({t}, {N}): {torch.cat(all_labels)}")
    
    # render categorized blocks
    MAX_SAMPLES = 16
    for M in range(model.num_blocks):
        all_block_attns = []
        for K in range(params.n_clusters): # number of clusters
            block_attns, preds_attns = [], []
            for block_coord in block_collector[M][K]:
                i, t, N = block_coord
                block_attns.append(attns[i, :, N])
                # preds_attns.append(pred_attns[i, 0, N])
            
            if len(block_attns) == 0:
                continue
            
            block_attns = torch.stack(block_attns)
            block_attns = block_attns.transpose(0, 1)
            block_attns = block_attns[:, :MAX_SAMPLES]
            block_attns = F.pad(block_attns, (0, 0, 0, 0, 0, 0, 0, max(MAX_SAMPLES - block_attns.shape[-4], 0)))
            all_block_attns.append(block_attns)
            
            # frames = [vutils.make_grid(block_attns, nrow=6, pad_value=tiles.max())]
            # frames += [vutils.make_grid(torch.stack(preds_attns), nrow=6, pad_value=tiles.max())]

        all_block_attns = torch.cat(all_block_attns, dim=1)
        datas = []
        for a in all_block_attns:
            _frame = vutils.make_grid(a, nrow=MAX_SAMPLES, pad_value=tiles.max())
            data = normalize_img(_frame)
            datas.append(data)
        datas[0].save(f'block_{M}.gif', save_all=True, append_images=datas[1:], duration=500, loop=0) 
    #####################################################################
    # 5. Autoregressive Imagination
    #####################################################################
    frames = torch.cat((samples, recon_tf), dim=1)
    IM_HORIZON = 2
    for i in range(IM_HORIZON): # imagination horizons
        pred_frame = model.reconstruct_autoregressive(frames[:, -params.cond_len:])
        frames = torch.cat((frames, pred_frame), dim=1)
    
    def draw_green_boundary(x):
        _min, _max = x.min(), x.max()
        x[:, :, :, :4], x[:, :, 1, :4] = _min, _max
        x[:, :, :, -4:], x[:, :, 1, -4:] = _min, _max
        x[:, :, :, :, :4], x[:, :, 1, :, :4] = _min, _max
        x[:, :, :, :, -4:], x[:, :, 1, :, -4:] = _min, _max
        
        return x

    frames[:, params.cond_len:] = draw_green_boundary(frames[:, params.cond_len:])
    frame = vutils.make_grid(frames.flatten(end_dim=1), nrow=frames.size(1), pad_value=frames.max())
    
    data = normalize_img(frame)
    data.save(f'imagination.png') 

    #####################################################################
    # 6. Compositional Imagination
    #####################################################################
    # pick a video clip
    idx = 48
    idx2 = 31#44
    
    # visualize x_i
    frame = vutils.make_grid(frames[idx], nrow=frames.size(1), pad_value=frames.max())
    data = normalize_img(frame)
    data.save(f"org_imagination.png")
    frame = vutils.make_grid(frames[idx2], nrow=frames.size(1), pad_value=frames.max())
    data = normalize_img(frame)
    data.save(f"org_imagination2.png")
    
    datas = []
    for fp in frames[idx][params.cond_len:]:
        datas += [normalize_img(fp)]
    datas[0].save('frames_oim.gif', save_all=True, append_images=datas[1:], duration=500, loop=0)
    
    datas = []
    for fp in frames[idx2][params.cond_len:]:
        datas += [normalize_img(fp)]
    datas[0].save('frames_oim2.gif', save_all=True, append_images=datas[1:], duration=500, loop=0)
    
    # visualize attn_map
    frame = vutils.make_grid(attns_t[idx], nrow=params.num_slots, pad_value=attns_t.max())
    data = normalize_img(frame)
    data.save(f"org_attnmap.png")
    
    # extract block-slot
    bs = all_slots[idx, -1]
    bs = rearrange(bs, 'n (m d) -> n m d', m=model.num_blocks)
    
    # extract block-slot
    bs2 = all_slots[idx2, -1]
    bs2 = rearrange(bs2, 'n (m d) -> n m d', m=model.num_blocks)
    
    # swap block-slots
    tmp = bs.detach().clone()
    tmp2 = bs2.detach().clone()
    # v2------------------------------------------------------------
    # moving sprites (idx: 7)
    # 1. color-swap
    # bs[0] = torch.stack((tmp[2, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    # bs[2] = torch.stack((tmp[0, 0], tmp[2, 1], tmp[2, 2], tmp[2, 3], tmp[2, 4], tmp[2, 5], tmp[2, 6], tmp[2, 7]))
        
    # # 2. shape-swap
    # bs[1] = torch.stack((tmp[1, 0], tmp[2, 1], tmp[1, 2], tmp[1, 3], tmp[1, 4], tmp[1, 5], tmp[1, 6], tmp[1, 7]))
    # bs[2] = torch.stack((tmp[2, 0], tmp[0, 1], tmp[2, 2], tmp[2, 3], tmp[2, 4], tmp[2, 5], tmp[2, 6], tmp[2, 7]))
    
    # # 3. position-swap
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[1, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    # bs[1] = torch.stack((tmp[1, 0], tmp[1, 1], tmp[1, 2], tmp[1, 3], tmp[0, 4], tmp[1, 5], tmp[1, 6], tmp[1, 7]))
    
    # # 4. change direction(1)
    # bs[1] = torch.stack((tmp[1, 0], tmp[1, 1], tmp[1, 2], tmp[0, 3], tmp[1, 4], tmp[1, 5], tmp[1, 6], tmp[1, 7]))
    # # 4. change direction(2)
    # bs[2] = torch.stack((tmp[2, 0], tmp[2, 1], tmp[2, 2], tmp[0, 3], tmp[2, 4], tmp[2, 5], tmp[2, 6], tmp[2, 7]))
    # # 4. change direction(3)
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp[1, 5], tmp[0, 6], tmp[0, 7]))
    # # 5. change speed(1)
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp[0, 5], tmp[2, 6], tmp[0, 7]))
    
    # moving-clevr-easy (idx: 19)
    # 1. color-swap
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[4, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[3, 7]))
    # bs[4] = torch.stack((tmp[4, 0], tmp[4, 1], tmp[4, 2], tmp[3, 3], tmp[4, 4], tmp[4, 5], tmp[4, 6], tmp[4, 7]))
    
    # 2. direction change (1)
    # bs[4] = torch.stack((tmp[4, 0], tmp[2, 1], tmp[4, 2], tmp[4, 3], tmp[4, 4], tmp[4, 5], tmp[4, 6], tmp[4, 7]))
    
    # 3. direction change (2)    
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[4, 6], tmp[3, 7]))
    
    # moving-clevrtex-easy (idx: 19)
    # 1. background change
    # bs2 = all_slots[idx+1, -1]
    # bs2 = rearrange(bs2, 'n (m d) -> n m d', m=model.num_blocks)
    
    # bs[0] = torch.stack((bs2[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    # bs[1] = torch.stack((bs2[1, 0], bs2[1, 1], bs2[1, 2], bs2[1, 3], bs2[1, 4], bs2[1, 5], bs2[1, 6], bs2[1, 7]))
    # bs[2] = torch.stack((bs2[2, 0], bs2[2, 1], bs2[2, 2], bs2[2, 3], bs2[2, 4], bs2[2, 5], bs2[2, 6], bs2[2, 7]))
    # bs[3] = torch.stack((bs2[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[3, 7]))
    # bs[4] = torch.stack((bs2[4, 0], tmp[4, 1], tmp[4, 2], tmp[4, 3], tmp[4, 4], tmp[4, 5], tmp[4, 6], tmp[4, 7]))
    
    # # 2. material swap
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp[3, 5], tmp[3, 6], tmp[0, 7]))
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[0, 5], tmp[0, 6], tmp[3, 7]))
    
    # # 3. shape swap
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[4, 7]))
    # bs[4] = torch.stack((tmp[4, 0], tmp[4, 1], tmp[4, 2], tmp[4, 3], tmp[4, 4], tmp[4, 5], tmp[4, 6], tmp[3, 7]))
    # # 4. position swap
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[4, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    # bs[4] = torch.stack((tmp[4, 0], tmp[4, 1], tmp[4, 2], tmp[4, 3], tmp[0, 4], tmp[4, 5], tmp[4, 6], tmp[4, 7]))
    
    # moving-clevr-hard (idx: 5)
    # change dynamics (1)
    # bs[2] = torch.stack((tmp[3, 0], tmp[2, 1], tmp[2, 2], tmp[2, 3], tmp[2, 4], tmp[2, 5], tmp[2, 6], tmp[2, 7]))
    # bs[3] = torch.stack((tmp[2, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[3, 7]))
    # change dynamics (2)
    # bs[2] = torch.stack((tmp[2, 0], tmp[2, 1], tmp[2, 2], tmp[2, 3], tmp[2, 4], tmp[2, 5], tmp[3, 6], tmp[2, 7]))
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[3, 7]))
    # change dynamics (3)
    # bs[2] = torch.stack((tmp[2, 0], tmp[2, 1], tmp[2, 2], tmp[2, 3], tmp[2, 4], tmp[2, 5], tmp[2, 6], tmp[2, 7]))
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[2, 6], tmp[3, 7]))
    # shape change
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[3, 7]))
    
    # v5.2------------------------------------------------------------
    # moving sprites (idx: 7)
    # 1. shape-swap
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[1, 2], tmp[0, 3], tmp[0, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    # bs[1] = torch.stack((tmp[1, 0], tmp[1, 1], tmp[0, 2], tmp[1, 3], tmp[1, 4], tmp[1, 5], tmp[1, 6], tmp[1, 7]))
        
    # 2. color-swap
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp[2, 5], tmp[0, 6], tmp[0, 7]))
    # bs[2] = torch.stack((tmp[2, 0], tmp[2, 1], tmp[2, 2], tmp[2, 3], tmp[2, 4], tmp[0, 5], tmp[2, 6], tmp[2, 7]))
    
    # # 3. change direction(1)
    # bs[1] = torch.stack((tmp[1, 0], tmp[1, 1], tmp[1, 2], tmp[2, 3], tmp[2, 4], tmp[1, 5], tmp[1, 6], tmp[1, 7]))
    
    # # 4. pos-swap
    # bs[0] = torch.stack((tmp[0, 0], tmp[0, 1], tmp[0, 2], tmp[2, 3], tmp[0, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    # bs[2] = torch.stack((tmp[2, 0], tmp[2, 1], tmp[2, 2], tmp[0, 3], tmp[2, 4], tmp[2, 5], tmp[2, 6], tmp[2, 7]))
    
    # ballet (idx: 10)
    # 1. pattern change
    # bs[1] = torch.stack((tmp[3, 0], tmp[1, 1], tmp[1, 2], tmp[1, 3], tmp[1, 4], tmp[1, 5], tmp[1, 6], tmp[1, 7]))
    # bs[3] = torch.stack((tmp[3, 0], tmp[3, 1], tmp[3, 2], tmp[3, 3], tmp[3, 4], tmp[3, 5], tmp[3, 6], tmp[3, 7]))
    # bs[4] = torch.stack((tmp[4, 0], tmp[4, 1], tmp[4, 2], tmp[4, 3], tmp[4, 4], tmp[4, 5], tmp[4, 6], tmp[4, 7]))
    
    # ood cim sprites-ood2 --------------------------------------------------------
    # idx 4,8
    # unseen comb --> yellow, spoke_6, down : left -> down (dynamic ood)
    # unseen comb --> blue, circle, left : spoke_5 -> circle (shape ood)
    # unseen comb --> cyan, squre, left : brown -> cyan (color ood)
    
    # color ood
    # bs[1] = torch.stack((tmp[1, 0], tmp[1, 1], tmp[1, 2], tmp[1, 3], tmp[1, 4], tmp[4, 5], tmp[1, 6], tmp[1, 7]))
    
    # dynamic ood
    # bs[0] = torch.stack((tmp2[0, 0], tmp[0, 1], tmp2[0, 2], tmp[0, 3], tmp[0, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    
    # shape ood 1
    # bs[4] = torch.stack((tmp[4, 0], tmp2[0, 1], tmp[4, 2], tmp2[0, 3], tmp[4, 4], tmp[4, 5], tmp[4, 6], tmp[4, 7]))
    
    # shape ood 2
    # bs[0] = torch.stack((tmp[0, 0], tmp2[0, 1], tmp[0, 2], tmp2[0, 3], tmp[0, 4], tmp[0, 5], tmp[0, 6], tmp[0, 7]))
    
    # ood cim clevr-hard-ood --------------------------------------------------------
    
    # unseen comb --> blue, cube, up
    # unseen comb --> green, sphere, left
    # unseen comb --> brown, cylinder, forward 
    
    # idx 48, 44
    # ood: green, sphere, left
    # bs[0] = torch.stack((tmp2[0, 0], tmp[0, 1], tmp[0, 2], tmp[0, 3], tmp[0, 4], tmp2[0, 5], tmp[0, 6], tmp[0, 7]))
    
    # idx 48, 31
    # ood: brown, cylinder, backward
    # bs2[3] = torch.stack((tmp2[3, 0], tmp2[3, 1], tmp2[3, 2], tmp2[3, 3], tmp[4, 4], tmp2[3, 5], tmp2[3, 6], tmp2[3, 7]))
    
    # idx 48, 31
    # ood: gray, sphere, up
    # bs[1] = torch.stack((tmp2[1, 0], tmp[1, 1], tmp[1, 2], tmp[1, 3], tmp[1, 4], tmp2[1, 5], tmp[1, 6], tmp[1, 7]))
    
    bs = rearrange(bs, 'n m d -> n (m d)')
    bs_pred = model.reconstruct_autoregressive_from_slot(bs)
    
    # visualize x'
    frames_prime = torch.cat((frames[idx, :params.cond_len], bs_pred)).unsqueeze(0)
    for i in range(IM_HORIZON): # imagination horizons
        pred_frame = model.reconstruct_autoregressive(frames_prime[:, -params.cond_len:])
        frames_prime = torch.cat((frames_prime, pred_frame), dim=1)
    frames_prime[:, params.cond_len:] = draw_green_boundary(frames_prime[:, params.cond_len:])
    frame = vutils.make_grid(frames_prime.flatten(end_dim=1), nrow=frames_prime.size(1), pad_value=frames.max())
    data = normalize_img(frame)
    data.save(f"compositional_imagination.png")
    
    datas = []
    for fp in frames_prime.flatten(end_dim=1)[params.cond_len:]:
        datas += [normalize_img(fp)]
    datas[0].save('frames_cim.gif', save_all=True, append_images=datas[1:], duration=500, loop=0)
    
    


if __name__ == "__main__":
    conf_base = OmegaConf.load(f"{CONFIG_PATH}/meta_config.yaml")
    conf_cli = OmegaConf.from_cli()
    assert 'model' in conf_cli.keys()
    model_conf = OmegaConf.load(f"{CONFIG_PATH}/{conf_cli.model}_config.yaml")
    params = OmegaConf.merge(conf_base, model_conf, conf_cli)

    main(params)