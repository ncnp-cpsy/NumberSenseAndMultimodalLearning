#!/bin/bash

python main.py \
       --run-type 'analyse' \
       --model 'VAE_OSCN' \
       --pre-trained './rslt/test/VAE_OSCN/2023-12-01T14:52:04.918500cieaxeki' \
       --output-dir './rslt/test/VAE_OSCN/analyse/'
python main.py \
       --run-type 'analyse' \
       --model 'VAE_CMNIST' \
       --pre-trained './rslt/test/VAE_CMNIST/2023-12-01T14:50:24.203159x4mnj8f3' \
       --output-dir './rslt/test/VAE_CMNIST/analyse/'
