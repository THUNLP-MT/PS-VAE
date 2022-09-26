#!/bin/bash
##########################################################################
# File Name: scripts/qed_opt.sh
# Author: kxz
# mail: jackie_kxz@outlook.com
# Created Time: Monday, September 26, 2022 PM02:54:07 HKT
#########################################################################
CODE_DIR=`dirname $0`/../src
DATA_DIR=`dirname $0`/../data
CKPT_DIR=`dirname $0`/../ckpts
export PYTHONPATH=$CODE_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # specify the GPU you want to use

python ${CODE_DIR}/generate.py \
  --ckpt ${CKPT_DIR}/zinc250k/constraint_prop_opt/epoch5.ckpt \
  --props logp \
  --n_samples 800 \
  --output_path cons_results \
  --lr 0.1 \
  --max_iter 80 \
  --patience 3 \
  --target 2 \
  --constraint_optim \
  --zinc800_logp ${DATA_DIR}/zinc250k/jtvae_zinc800_logp.smi \
  --cpus 8 \
  --gpus 0