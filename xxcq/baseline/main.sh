#!/bin/bash

DATA_DIR='./datasets/CAIL司法文本信息抽取'
MODEL_TYPE='bert'
MODEL_NAME_OR_PATH='bert-base-chinese'
#MODEL_NAME_OR_PATH='output/best_checkpoint'

OUTPUT_DIR='./output'
LABEL='./datasets/CAIL司法文本信息抽取/labels.txt'

CUDA_VISIBLE_DEVICES='0' /home/user/miniconda/bin/python run_softmax_ner.py \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--labels $LABEL \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--adv_training fgm \
--num_train_epochs 1 \
--max_seq_length 512 \
--logging_steps 0.9 \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--learning_rate 5e-5 \
--bert_lr 5e-5 \
--classifier_lr  5e-5 \
--overwrite_cache \
--overwrite_output_dir \
