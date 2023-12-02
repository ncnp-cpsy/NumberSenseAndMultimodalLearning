#!/bin/bash

DIR_CMNIST="2023-12-02T18:47:17.809494hfrjalg0"
DIR_OSCN="2023-12-02T18:56:27.657830yyo49vww"

python main.py \
       --run-type "analyse" \
       --model "VAE_CMNIST" \
       --pre-trained "./rslt/test/VAE_CMNIST/${DIR_CMNIST}" \
       --output-dir "./rslt/test/VAE_CMNIST/${DIR_CMNIST}/analyse"

python main.py \
       --run-type "analyse" \
       --model "VAE_OSCN" \
       --pre-trained "./rslt/test/VAE_OSCN/${DIR_OSCN}" \
       --output-dir "./rslt/test/VAE_OSCN/${DIR_OSCN}/analyse"
