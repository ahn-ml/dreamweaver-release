import torch
from torch import nn
import torch.nn.functional as F

from commons.utils import linear, BlockLinear, Conv2dBlock, conv2d, CartesianPositionalEmbedding, \
    OneHotDictionary, LearnedPositionalEmbedding1D, gumbel_softmax
from models.dreamweaver.transformer import TransformerEncoder, TransformerDecoder
from commons.modules.dvae import dVAE
from models.dreamweaver.binder import Binder


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        if not args.use_dino:
            self.cnn = nn.Sequential(
                Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
            ) if not args.use_deeper_cnn else nn.Sequential(
                Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 2, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
            )
            if args.use_deeper_cnn:
                out_dim = args.image_size // 4
            else:
                out_dim = args.image_size if args.image_size == 64 else args.image_size // 2
                
            self.pos = CartesianPositionalEmbedding(args.d_model, out_dim)

            self.layer_norm = nn.LayerNorm(args.d_model)

            self.mlp = nn.Sequential(
                linear(args.d_model, args.d_model, weight_init='kaiming'),
                nn.ReLU(),
                linear(args.d_model, args.d_model))
        else:
            self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            self.proj = linear(self.dino.embed_dim, args.d_model, weight_init='kaiming')

        self.binder = Binder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.num_prototypes, args.num_blocks, 
            args.use_bi_attn, args.just_use_mlp,
            args.skip_prototype_memory, args.prototype_memory_on_last_slot,
            args.num_predictor_layers, args.num_predictor_heads, args.predictor_dropout)


class ImageDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_proj = BlockLinear(args.slot_size, args.d_model * args.num_blocks, args.num_blocks)

        self.block_pos = nn.Parameter(torch.zeros(1, 1, args.d_model * args.num_blocks), requires_grad=True)
        self.block_pos_proj = nn.Sequential(
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks),
            nn.ReLU(),
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks)
        )

        self.block_coupler = TransformerEncoder(num_blocks=1, d_model=args.d_model, num_heads=4)

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)
            
        if args.use_sln:
            self.cond_w = nn.Parameter(torch.Tensor(args.pred_len, 1, args.d_model))
            nn.init.xavier_uniform_(self.cond_w)
        else:
            self.W_ = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.decoder_pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_layers, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout, use_sln=True if args.use_sln else False)

        self.head = linear(args.d_model, args.vocab_size, bias=False)

class DreamweaverModel(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.num_prototypes = args.num_prototypes
        self.image_channels = args.img_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.num_blocks = args.num_blocks
        self.use_sln = args.use_sln
        self.cond_len = args.cond_len
        self.pred_len = args.pred_len
        self.use_dino = args.use_dino
        self.use_block_coupling = args.use_block_coupling

        # dvae
        self.dvae = dVAE(args.vocab_size, args.img_channels)

        # encoder networks
        self.image_encoder = ImageEncoder(args)

        # decoder networks
        self.image_decoder = ImageDecoder(args)

    def forward(self, srcv, dstv, tau):
        B, T, C, H, W = srcv.size()
        video = torch.cat([srcv, dstv], dim=1)
        Tv = video.size(1)
        srcv_flat = srcv.flatten(end_dim=1)                               # B * T, C, H, W
        video_flat = video.flatten(end_dim=1)                             # B * (Tv), C, H, W

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)       # B * (Tv), vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, False, dim=1)                  # B * (Tv), vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()         # B * (Tv), vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                         # B * (Tv), H_enc * W_enc, vocab_size
        z_emb = self.image_decoder.dict(z_hard)                                                     # B * (Tv), H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.dvae.decoder(z_soft).reshape(B, Tv, C, H, W)               # B, Tv, C, H, W
        dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * (Tv))                      # 1
        
        # image feature extraction (via cnn or dino)
        if not self.use_dino:
            emb = self.image_encoder.cnn(srcv_flat)  # B*T, cnn_hidden_size, H, W
            emb = self.image_encoder.pos(emb)  # B*T, cnn_hidden_size, H, W
            H_enc, W_enc = emb.shape[-2:]

            emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B*T, H * W, cnn_hidden_size
            emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B*T, H * W, d_model
        else:
            emb_set = self.image_encoder.dino.get_intermediate_layers(srcv_flat)[0] # BT, H * W + 1 , dino.emb_dim
            emb_set = emb_set[:,1:] # remove CLS token /  BT, H * W, dino.emb_dim
            H_enc = W_enc = int(emb_set.shape[-2]**(0.5))
            
            emb_set = self.image_encoder.proj(emb_set) # BT, H * W, d_model
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)  # B, T, H * W, d_model

        # block-slot binder
        slots, attns = self.image_encoder.binder(emb_set)  # slots: B, T, num_slots, slot_size
                                                              # attns: B, T, num_inputs, num_slots
        slots_out = slots.detach().clone()
        attn_raw = attns.detach().clone()
        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)  # B, T,num_slots, 1, H, W
        attns = srcv.unsqueeze(2) * attns + (1. - attns)  # B, T, num_slots, C, H, W

        slots = self.image_decoder.slot_proj(slots)  # B, T, num_slots, num_blocks * d_model
        if self.use_block_coupling:
            # block coupling
            slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, T, num_slots, num_blocks * d_model
            slots = slots.reshape(B*T, self.num_slots, self.num_blocks, -1)  # BT, num_slots, num_blocks, d_model
            slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # BT * num_slots, num_blocks, d_model
            slots = slots.reshape(B, T, self.num_slots * self.num_blocks, -1)  # B, T, num_slots * num_blocks, d_model

        # decode                    
        if not self.use_sln: # random jumpy prediction with step indicator
            z_emb = z_emb.view(B, Tv, -1, self.d_model)[:, -self.pred_len:]          # B, Tp, H_enc * W_enc, d_model
            # stochastic frame choice for jumpy prediction
            rnd_ind = torch.randint(high=self.pred_len, size=(B, ))           
            z_emb = torch.stack([z[rnd_ind[i]] for i, z in enumerate(z_emb)]) # random prediction targets
            
            # mode 5.3: non-linearly dependent indicator (indicator = (f^k)*BOS, f=mlp)
            bosmap = [self.image_decoder.bos]
            for _ in range(self.pred_len-1):
                bosmap += [self.image_decoder.W_(bosmap[-1])]
            bosmap = torch.stack(bosmap,dim=1)
            z_emb = torch.cat([bosmap[:, rnd_ind].flatten(end_dim=1), z_emb], dim=1)                   # B, 1 + H_enc * W_enc, d_model
                
            dec_in = self.image_decoder.decoder_pos(z_emb)                                                       # B, 1 + H_enc * W_enc, d_model
            z_hard = z_hard.view(B, Tv, -1, self.vocab_size)[:, -self.pred_len:]
            target = torch.stack([z[rnd_ind[i]] for i, z in enumerate(z_hard)])
            pred = self.image_decoder.tf(dec_in[:, :-1], slots[:, -1])                                  # B, H_enc * W_enc, d_model
            pred = self.image_decoder.head(pred)                                                                  # BT, H_enc * W_enc, vocab_size
            cross_entropy = -(target * torch.log_softmax(pred, dim=-1)).sum() / (B)                        # 1
            
        else: # random jumpy using self-modularization norm
            z_emb = z_emb.view(B, Tv, -1, self.d_model)[:, -self.pred_len:]          # B, Tp, H_enc * W_enc, d_model
            # stochastic frame choice for jumpy prediction
            rnd_ind = torch.randint(high=self.pred_len, size=(B, ))           
            z_emb = torch.stack([z[rnd_ind[i]] for i, z in enumerate(z_emb)]) # random prediction targets
            # add bos token
            z_emb = torch.cat([self.image_decoder.bos.expand(B, -1, -1), z_emb], dim=1)             # B, 1 + H_enc * W_enc, d_model
            
            # preprocess for decoder
            dec_in = self.image_decoder.decoder_pos(z_emb)[:, :-1]                                                   # B, 1 + H_enc * W_enc, d_model
            z_hard = z_hard.view(B, Tv, -1, self.vocab_size)[:, -self.pred_len:]
            target = torch.stack([z[rnd_ind[i]] for i, z in enumerate(z_hard)])
            
            # compute decoder
            w = self.image_decoder.cond_w[rnd_ind].expand(-1, dec_in.size(1), -1)
            pred = self.image_decoder.tf(dec_in, slots[:, -1], w)                                  # B, H_enc * W_enc, d_model
            pred = self.image_decoder.head(pred)                                                                  # BT, H_enc * W_enc, vocab_size
            cross_entropy = -(target * torch.log_softmax(pred, dim=-1)).sum() / (B)                        # 1
            
        losses = {'loss': cross_entropy+dvae_mse, 'dvae_mse': dvae_mse, 'cross_entropy': cross_entropy}
        
        return losses, dvae_recon, attns, slots_out, attn_raw

    def encode(self, video):
        """
        image: B, C, H, W
        """
        B, T, C, H, W = video.size()
        video_flat = video.flatten(end_dim=1)

        # image feature extraction (via cnn or dino)
        if not self.use_dino:
            emb = self.image_encoder.cnn(video_flat)  # B*T, cnn_hidden_size, H, W
            emb = self.image_encoder.pos(emb)  # B*T, cnn_hidden_size, H, W
            H_enc, W_enc = emb.shape[-2:]

            emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B*T, H * W, cnn_hidden_size
            emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B*T, H * W, d_model
        else:
            emb_set = self.image_encoder.dino.get_intermediate_layers(video_flat)[0] # BT, H * W + 1 , dino.emb_dim
            emb_set = emb_set[:,1:] # remove CLS token /  BT, H * W, dino.emb_dim
            H_enc = W_enc = int(emb_set.shape[-2]**(0.5))
            
            emb_set = self.image_encoder.proj(emb_set) # BT, H * W, d_model
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)  # B, T, H * W, d_model

        # block-slot binder
        slots, _ = self.image_encoder.binder(emb_set)  # slots: B, T, num_slots, slot_size

        return slots

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, T, N, D = slots.size()
        slots_flat = slots.flatten(end_dim=1)
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # BT, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # BT, num_slots, num_blocks * d_model
        slots = slots.reshape(B*T, self.num_slots, self.num_blocks, -1)  # BT, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # BT * num_slots, num_blocks, d_model
        slots = slots.reshape(B, T, self.num_slots * self.num_blocks, -1)  # B, T, num_slots * num_blocks, d_model

        # generate image tokens auto-regressively
        if not self.use_sln:
            z_gen = []
            for t in range(self.pred_len):
                bosmap = [self.image_decoder.bos]
                for _ in range(self.pred_len-1):
                    bosmap += [self.image_decoder.W_(bosmap[-1])]
                bosmap = torch.stack(bosmap,dim=1)
                input = bosmap[:, t].expand(B, 1, -1)
                z_gen_ = slots.new_zeros(0)
                for _ in range(gen_len):
                    decoder_output = self.image_decoder.tf(
                        self.image_decoder.decoder_pos(input),
                        slots[:, -1]
                    )
                    pred = self.image_decoder.head(decoder_output)[:, -1:]
                    z_next = F.one_hot(pred.argmax(dim=-1), self.vocab_size)
                    z_gen_ = torch.cat((z_gen_, z_next), dim=1)
                    input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)
                z_gen += [z_gen_]
            z_gen = torch.stack(z_gen,dim=1).flatten(end_dim=1)
            z_gen = z_gen.transpose(-1, -2).float().reshape(B*self.pred_len, -1, H_enc, W_enc)
        else:
            z_gen = []
            for t in range(self.pred_len):
                z_gen_ = slots.new_zeros(0)
                input = self.image_decoder.bos.expand(B, 1, -1)
                w = self.image_decoder.cond_w[t]
                for _ in range(gen_len):
                    decoder_output = self.image_decoder.tf(
                        self.image_decoder.decoder_pos(input),
                        slots[:, -1], w
                    )
                    pred = self.image_decoder.head(decoder_output)[:, -1:]
                    z_next = F.one_hot(pred.argmax(dim=-1), self.vocab_size)
                    z_gen_ = torch.cat((z_gen_, z_next), dim=1)
                    input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)
                z_gen += [z_gen_]
            z_gen = torch.stack(z_gen,dim=1).flatten(end_dim=1)
            z_gen = z_gen.transpose(-1, -2).float().reshape(B*self.pred_len, -1, H_enc, W_enc)
            
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer

    def reconstruct_autoregressive(self, video):
        """
        video: batch_size x timestep x image_channels x H x W
        """
        B, T, C, H, W = video.size()
        slots = self.encode(video)
        recon_transformer = self.decode(slots)
        recon_transformer = recon_transformer.reshape(B, self.pred_len, C, H, W)

        return recon_transformer

    def reconstruct_autoregressive_from_slot(self, slots):
        """
        slots: batch_size x timestep x image_channels x H x W
        """
        N, D = slots.size()
        recon_transformer = self.decode(slots.view(1, 1, N, D))

        return recon_transformer
