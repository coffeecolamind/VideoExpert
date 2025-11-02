#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

current_time=$(date "+%m-%d_%H-%M")
output_prefix="Output_Pretrain_VideoExpert."
MASTER_PORT=29570

hostname

# For Hybrid datasets pretraining with Stage 2.
deepspeed --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed VideoExpert_Scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path Training_Data/Pretrain \
    --feat_folder Datasets/PreTrain_Data \
    --pretrain_mm_mlp_adapter checkpoints/CLSPatch_checkpoints/CLSPatch_stage1/mm_projector.bin \
    --output_dir ./Results/Pretain \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to none \
    --mm_projector_type linear \
    --tasks "pretrain||videoqa" \
    --task_sample_rate 5 1 \
    --pretrain_data "Stage2||Stage3" \
    --pretrain_sample_rate 4 1 \
    --loc_interaction_type "simp_add" \
    --patch_filter_type cls_patch \
    --clip_path checkpoints/CLIP/ViT-L-14.pt
    --samples_per_epoch 50000  >> "${output_prefix}${current_time}.txt" 2>&1