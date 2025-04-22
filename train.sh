CUDA_VISIBLE_DEVICES=4 python scripts/train_models.py \
prefix=dw-moving-sprites2-cond2-pred2-ns5-sd768-nb8-pm64-dl8-bi-iter3-lr1/2/2e-4-clip0.1-400k model=dw mode=predict dataset=moving-sprites2 \
image_size=64 num_decoder_layers=8 num_slots=5 num_blocks=8 slot_size=768 batch_size=24 val_batch_size=24 num_prototypes=64 cond_len=2 pred_len=2 max_epochs=90 \
warmup_steps_pct=0.15 decay_steps_pct=1.25 use_bi_attn=true just_use_mlp=true use_sln=false lr_enc=1e-4 lr_dvae=2e-4 lr_dec=2e-4 max_gradient_size=0.1 gpus=1 seed=2024 \
is_logger_enabled=false