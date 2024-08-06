This repository is a fork of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). The original README can be found [here](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#readme).

# BitPipe-Bidirectional Interleaved Pipeline Parallelism

BitPipe is a bidirectional interleaved pipeline parallelism for accelerating large models training. Specifically, a hybrid scheme of fusing interleaved pipelines with bidirectional pipelines is proposed to reduce the computational time of each single micro-batch and multiply the number of simultaneous execution devices. A V-shaped schedule with eager
gradient synchronization is introduced to reduce and overlap the communication between devices. 

# Usage
Quick settings to enable BitPipe:
<pre>
    --enable-bitpipe-schedule 
</pre>

## BERT Pretraining
<pre>
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=ibp
  
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=1234
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/enwiki/bert_case_check
VOCAB_FILE=/data/enwiki/bert-large-cased-vocab.txt
DATA_PATH=/data/enwiki/my-bert_text_sentence

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --pipeline-model-parallel-size 8 \
    --enable-bitpipe-schedule \
    --num-layers 64 \
    --hidden-size 2560 \
    --num-attention-heads 64 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
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
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
</pre>
## GPT Pretraining

<pre>
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=ibp

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=1234
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/gpt2-openwebtext-data/gpt2_test
VOCAB_FILE=/data/gpt2-openwebtext-data/gpt2-vocab.json
MERGE_FILE=/data/gpt2-openwebtext-data/gpt2-merges.txt
DATA_PATH=/data/gpt2-openwebtext-data/my-gpt2_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --pipeline-model-parallel-size 8 \
    --enable-bitpipe-schedule \
    --num-layers 96 \
    --hidden-size 3072 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
</pre>

-->


