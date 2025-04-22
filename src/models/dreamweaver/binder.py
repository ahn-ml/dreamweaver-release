from torch import nn
import torch
from models.dreamweaver.transformer import TransformerEncoder
from commons.utils import linear, BlockLayerNorm, BlockAttention, BlockGRU, BlockLinear
import torch.nn.functional as F

class BlockPrototypeMemory(nn.Module):
    def __init__(self, num_prototypes, num_blocks, d_model):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_block = self.d_model // self.num_blocks

        # block prototype memory
        self.mem_params = nn.Parameter(torch.zeros(1, num_prototypes, num_blocks, self.d_block), requires_grad=True)
        nn.init.trunc_normal_(self.mem_params)
        self.mem_proj = nn.Sequential(
            linear(self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, self.d_block)
        )

        # norms
        self.norm_mem = BlockLayerNorm(d_model, num_blocks)
        self.norm_query = BlockLayerNorm(d_model, num_blocks)

        # block attention
        self.attn = BlockAttention(d_model, num_blocks)

    def forward(self, queries):
        '''
        queries: B, N, d_model
        return: B, N, d_model
        '''

        B, N, _ = queries.shape

        # get memories
        mem = self.mem_proj(self.mem_params)  # 1, num_prototypes, num_blocks, d_block
        mem = mem.reshape(1, self.num_prototypes, -1)  # 1, num_prototypes, d_model

        # norms
        mem = self.norm_mem(mem)  # 1, num_prototypes, d_model
        queries = self.norm_query(queries)  # B, N, d_model

        # broadcast
        mem = mem.expand(B, -1, -1)  # B, num_prototypes, d_model

        # read
        return self.attn(queries, mem, mem)  # B, N, d_model


class Binder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, num_prototypes, num_blocks, 
                 use_bi_attn=False, just_use_mlp=False,
                 skip_prototype_memory=False,
                 prototype_memory_on_last_slot=False,
                 num_predictor_layers=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_blocks = num_blocks
        self.epsilon = epsilon
        self.use_bi_attn = use_bi_attn
        self.just_use_mlp = just_use_mlp
        self.skip_prototype_memory = skip_prototype_memory
        self.prototype_memory_on_last_slot = prototype_memory_on_last_slot

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = BlockLayerNorm(slot_size, num_blocks)

        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = BlockGRU(slot_size, slot_size, num_blocks)
        
        self.mlp = nn.Sequential(
            BlockLinear(slot_size, mlp_hidden_size, num_blocks),
            nn.ReLU(),
            BlockLinear(mlp_hidden_size, slot_size, num_blocks))
        
        if (not self.skip_prototype_memory) or self.prototype_memory_on_last_slot:
            self.prototype_memory = BlockPrototypeMemory(num_prototypes, num_blocks, slot_size)

        self.predictor = TransformerEncoder(num_predictor_layers, slot_size // (num_blocks if use_bi_attn else 1), 
                                            num_predictor_heads, dropout)


    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.                
                slots = self.gru(
                    updates.view(-1, self.slot_size),
                    slots_prev.view(-1, self.slot_size)
                )
                slots = slots.view(-1, self.num_slots, self.slot_size)

                # use MLP only when more than one iterations
                if self.just_use_mlp or (i < self.num_iterations - 1):
                    slots = slots + self.mlp(self.norm_mlp(slots))
                if (not self.skip_prototype_memory) or (self.prototype_memory_on_last_slot and (t==T-1)):
                    slots = self.prototype_memory(slots)
            corrector_slots = slots


            # (block-wise independent) predictor 
            if self.use_bi_attn:
                slots = slots.view(B, self.num_slots*self.num_blocks, -1)
                slots = self.predictor(slots) 
                slots = slots.view(B, self.num_slots, -1)
            else:
                slots = self.predictor(slots)

            # collect
            attns_collect += [attn_vis]
            slots_collect += [corrector_slots]
            
        attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size
        
        return slots_collect, attns_collect
