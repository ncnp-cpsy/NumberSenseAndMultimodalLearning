#!/bin/bash

EXPR_DIR='test'

python main.py  \
       --experiment "${EXPR_DIR}/VAE_CMNIST" \
       --run-type "train" \
       --model "VAE_CMNIST" \
       --obj "elbo" \
       --K 30 \
       --batch-size 128 \
       --epochs 10 \
       --learn-prior \
       --seed 4

python main.py  \
       --experiment "${EXPR_DIR}/VAE_OSCN" \
       --run-type "train" \
       --model "VAE_OSCN" \
       --obj "elbo" \
       --K 30 \
       --batch-size 128 \
       --epochs 10 \
       --learn-prior \
       --seed 4
