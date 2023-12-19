#!/bin/bash

OUTPUT_DIR="final_latent_images_test_231024_2/"
OUTPUT_DIR_QUAL="${OUTPUT_DIR}qualitative/"
OUTPUT_DIR_QUAN="${OUTPUT_DIR}quantitative/"
mkdir -pv $OUTPUT_DIR_QUAL $OUTPUT_DIR_QUAN

# CMNIST-OSCN
python visualize_latent.py \
       --model 'cmnist_oscn' \
       --pre-trained '../experiments/cmnist-oscn/final' \
       --target-modality 0 \
       --target-property 0 \
       --output-dir $OUTPUT_DIR_QUAL \
       > "${OUTPUT_DIR_QUAN}cmnist_oscn_0_0.txt"
python visualize_latent.py \
       --model 'cmnist_oscn' \
       --pre-trained '../experiments/cmnist-oscn/final' \
       --target-modality 0 \
       --target-property 1 \
       --output-dir $OUTPUT_DIR_QUAL \
       > "${OUTPUT_DIR_QUAN}cmnist_oscn_0_1.txt"
python visualize_latent.py \
       --model 'cmnist_oscn' \
       --pre-trained '../experiments/cmnist-oscn/final' \
       --target-modality 0 \
       --target-property 2 \
       --output-dir $OUTPUT_DIR_QUAL \
       > "${OUTPUT_DIR_QUAN}cmnist_oscn_0_2.txt"
python visualize_latent.py \
       --model 'cmnist_oscn' \
       --pre-trained '../experiments/cmnist-oscn/final' \
       --target-modality 1 \
       --target-property 0 \
       --output-dir $OUTPUT_DIR_QUAL \
       > "${OUTPUT_DIR_QUAN}cmnist_oscn_1_0.txt"

python visualize_latent.py --model 'cmnist_oscn' --pre-trained '../experiments/cmnist-oscn/final' --target-modality 1 --target-property 1 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_1_1.txt"
python visualize_latent.py --model 'cmnist_oscn' --pre-trained '../experiments/cmnist-oscn/final' --target-modality 1 --target-property 2 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_1_2.txt"

# CMNIST
python visualize_latent.py --model 'cmnist' --pre-trained '../experiments/cmnist/final' --target-modality 0 --target-property 0 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_0_0.txt"
python visualize_latent.py --model 'cmnist' --pre-trained '../experiments/cmnist/final' --target-modality 0 --target-property 1 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_0_1.txt"

# OSCN
python visualize_latent.py --model 'oscn' --pre-trained '../experiments/oscn/final' --target-modality 0 --target-property 0 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}oscn_0_0.txt"
python visualize_latent.py --model 'oscn' --pre-trained '../experiments/oscn/final' --target-modality 0 --target-property 1 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}oscn_0_1.txt"
python visualize_latent.py --model 'oscn' --pre-trained '../experiments/oscn/final' --target-modality 0 --target-property 2 --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}oscn_0_2.txt"
