#!/bin/bash
##########################################################################
# File Name: train.sh
# Author: kxz
# mail: jackie_kxz@outlook.com
# Created Time: Monday, September 26, 2022 PM02:43:39 HKT
#########################################################################
CODE_DIR=`dirname $0`/../src
DATA_DIR=`dirname $0`/../data
CKPT_DIR=`dirname $0`/../ckpts
export PYTHONPATH=$CODE_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # specify the GPU you want to use

python ${CODE_DIR}/guacamol_exps/goal_directed/goal_directed_grad.py \
  --ckpt ${CKPT_DIR}/zinc250k/zinc_guaca_goal/epoch5.ckpt \
  --pred_ckpt ${CKPT_DIR}/zinc250k/zinc_guaca_goal/predictor_epoch9.ckpt \
  --gpu 0 \
  --output_dir results