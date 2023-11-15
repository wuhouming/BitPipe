#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/workspace/ckpts
VOCAB_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/vocab.json
MERGE_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/merges.txt
DATA_PATH=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/train_text_document

WANDB_PROJECT=moe
WANDB_EXP_NAME=moe_1p3b_16e_slimpj_test
WANDB_SAVE_DIR=/workspace/wandb

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 1 \
    --seq-length 64 \
    --max-position-embeddings 64 \
    --micro-batch-size 8 \
    --global-batch-size 1024 \
    --lr 0.00015 \
    --train-iters 500 \
    --lr-decay-iters 500 \
    --lr-decay-style cosine \
    --min-lr 0.001 \
    --constant-lr 0.05 \
    --constant-fraction 0.4 \
    --inv-sqrt-cooldown-fraction 0.3 \
    --inv-sqrt-scale 30.0 \
    --num-cycles 1 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --num-experts 16 \
    --expert-model-parallel-size 8 \
    --recompute-granularity selective \
    --use-flash-attn
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --num-workers 0 \
    --distributed-timeout-minutes 120
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5 \
    --eval-interval 5 \
    --eval-iters 10 \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR
"

torchrun $DISTRIBUTED_ARGS /workspace/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
