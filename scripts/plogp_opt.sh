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


python ${CODE_DIR}/generate.py --eval \
  --ckpt ${CKPT_DIR}/zinc250k/prop_opt/epoch5.ckpt \
  --props logp \
  --n_samples 10000 \
  --output_path qed.smi \
  --lr 0.1 \
  --max_iter 100 \
  --patience 3 \
  --target 2 \
  --cpus 8