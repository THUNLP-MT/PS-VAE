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

python ${CODE_DIR}/train.py \
	--train_set ${DATA_DIR}/zinc250k/train.txt \
	--valid_set ${DATA_DIR}/zinc250k/valid.txt \
	--test_set ${DATA_DIR}/zinc250k/test.txt \
	--vocab ${CKPT_DIR}/zinc250k/prop_opt/zinc_bpe_300.txt \
	--batch_size 32 \
	--shuffle \
	--alpha 0.1 \
	--beta 0 \
	--max_beta 0.01 \
	--step_beta 0.002 \
	--kl_anneal_iter 1000 \
	--kl_warmup 0 \
	--lr 1e-3 \
	--save_dir ${CKPT_DIR} \
	--grad_clip 10.0 \
	--epochs 6 \
	--gpus 0 \
	--props qed logp \
	--latent_dim 56 \
	--node_hidden_dim 300 \
	--graph_embedding_dim 400 \
	--patience 3