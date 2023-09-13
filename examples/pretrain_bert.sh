#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CHECKPOINT_PATH=<Specify path>
# VOCAB_FILE=<Specify path to file>/bert-vocab.txt
# DATA_PATH=<Specify path and file prefix>_text_sentence

CHECKPOINT_PATH=/workspace/checkpoints/bert-large-uncased/
VOCAB_FILE=vocab/bert-large-uncased-vocab.txt
DATA_PATH=output_prefix/my-bert-uncased_text_sentence

BERT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.0001 \
    --train-iters 2000000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

WANDB="
    --wandb-project Megatron-LM \
    --wandb-name bert \
    --wandb-save_code True \
    --wandb-tags baseline \
    --wandb-model bert \
    --wandb-optimizer adam \
    --wandb-optimizer-version original \
    --wandb-id bert-large-uncased1 \
"

torchrun pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $WANDB \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
