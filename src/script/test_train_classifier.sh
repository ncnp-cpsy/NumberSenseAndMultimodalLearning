#!/bin/bash

EXPR_DIR='./rslt/test'

python main.py  \
       --model "Classifier_CMNIST" \
       --experiment "${EXPR_DIR}/cmnist" \
       --obj "cross" \
       --batch-size 128 \
       --epochs 30 \
       --seed 4 \
