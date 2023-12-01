#!/bin/bash

EXPR_DIR='test'

python main.py  \
       --experiment "${EXPR_DIR}/Classifier_OSCN" \
       --run-type "classify" \
       --model "Classifier_OSCN"

python main.py  \
       --experiment "${EXPR_DIR}/Classifier_CMNIST" \
       --run-type "classify" \
       --model "Classifier_CMNIST"
