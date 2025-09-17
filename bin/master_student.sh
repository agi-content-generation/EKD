#!/bin/bash

SAVE_DIR=${1:-"checkpoints/master_student_model_iwslt14"}
LOG_FILE=${2:-"junior_senior_student_model_iwslt14.txt"}
DATA_SET_DIR=${3:-"data-bin/iwslt14.tokenized.de-en"}
JUNIOR_STUDENT_MODEL_PATH=${4:-"checkpoints/junior_student_model_iwslt14/checkpoint_best.pt"}
TEACHER_MODEL_PATH=${5:-"checkpoints/senior_teacher_model_iwslt14/checkpoint_best.pt"}
MODEL=${6:-"transformer_iwslt_de_en"}
MAX_EPOCH=${7:-100}

CUDA_VISIBLE_DEVICES=2 fairseq-train \
    $DATA_SET_DIR \
    --arch $MODEL --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --use-junior-student-model \
    --junior-student-model-path $JUNIOR_STUDENT_MODEL_PATH \
    --learn-from-teacher-model \
    --teacher-model-path $TEACHER_MODEL_PATH \
    --distil-strategy distil_all \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch $MAX_EPOCH \
    --batch-size 1024 \
    --save-dir $SAVE_DIR 2>&1 | tee $LOG_FILE