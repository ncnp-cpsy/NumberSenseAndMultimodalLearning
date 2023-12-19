#!/bin/bash

DIR_CMNIST="2023-12-03T22:56:17.623511znaqe_3f"
DIR_OSCN="2023-12-03T22:48:54.511450fzcz4d3o"

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
