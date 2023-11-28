#!/bin/bash

EXPR_DIR='./rslt/test/'

python main.py  \
       --model "cmnist" \
       --experiment "${EXPR_DIR}/cmnist" \
       --obj "cross" \
       --batch-size 128 \
       --epochs 3 \
       --seed $SEED_INDEX
