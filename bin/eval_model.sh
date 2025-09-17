#!/bin/bash

BEST_MODEL_PATH=${1:-"checkpoints/master_student_model_iwslt14/checkpoint_best.pt"}
LOG_FILE=${2:-"master_student_model_iwslt14.txt"}
DATA_SET_DIR=${3:-"data-bin/iwslt14.tokenized.de-en"}

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_SET_DIR \
    --path $BEST_MODEL_PATH \
    --batch-size 128 --beam 5 --remove-bpe 2>&1 | tee $LOG_FILE