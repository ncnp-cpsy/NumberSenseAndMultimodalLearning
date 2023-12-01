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

python main.py  \
       --experiment "${EXPR_DIR}/OSCN" \
       --run-type "classify" \
       --model "Classifier_OSCN"

python main.py  \
       --experiment "${EXPR_DIR}/CMNIST" \
       --run-type "classify" \
       --model "Classifier_CMNIST"

python main.py \
       --run-type 'analyse' \
       --model 'VAE_OSCN' \
       --pre-trained './rslt/test/VAE_OSCN/2023-12-01T14:52:04.918500cieaxeki' \
       --target-modality 0 \
       --target-property 0 \
       --output-dir 'rslt/test/VAE_OSCN/analyse/' \
       > "./VAE_OSCN_0_0.txt"

python main.py \
       --run-type 'analyse' \
       --model 'VAE_CMNIST' \
       --pre-trained './rslt/test/VAE_CMNIST/2023-12-01T14:50:24.203159x4mnj8f3' \
       --target-modality 0 \
       --target-property 0 \
       --output-dir 'rslt/test/VAE_OSCN/analyse/' \
       > "./VAE_CMNIST_0_0.txt"
