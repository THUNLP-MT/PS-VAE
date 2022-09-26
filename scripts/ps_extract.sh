#!/bin/bash
##########################################################################
# File Name: ps_extract.sh
# Author: kxz
# mail: jackie_kxz@outlook.com
# Created Time: Monday, September 26, 2022 PM02:29:25 HKT
#########################################################################
CODE_DIR=`dirname $0`/../src
DATA_DIR=`dirname $0`/../data
export PYTHONPATH=$CODE_DIR:$PYTHONPATH

python ${CODE_DIR}/data/mol_bpe.py \
    --data ${DATA_DIR}/zinc250k/train.txt \
    --output ${DATA_DIR}/zinc_bpe_300.txt \
    --vocab_size 300