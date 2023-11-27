#!/bin/bash

EXPR_DIR='./rslt/test/'

python ./src/train.py  \
       --model "cmnist" \
       --experiment "${EXPR_DIR}/cmnist" \
       --obj "cross" \
       --batch-size 128 \
       --epochs 3 \
       --seed $SEED_INDEX
