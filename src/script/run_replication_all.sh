#!/bin/bash -l

# TODO: use commandline argumants?

shopt -s expand_aliases
alias my-python='/home/taka/.pyenv/shims/python'

SUFFIX="231029_1"

EXPR_DIR="../experiments/loop-${SUFFIX}"
EXPR_DIR_CMNIST_OSCN="${EXPR_DIR}/cmnist-oscn"
EXPR_DIR_CMNIST="${EXPR_DIR}/cmnist"
EXPR_DIR_OSCN="${EXPR_DIR}/oscn"

OUTPUT_DIR="./analysis/loop-${SUFFIX}"

echo "Experiments:" $EXPR_DIR
echo "Analysis:" $OUTPUT_DIR
mkdir -pv $EXPR_DIR_CMNIST_OSCN $EXPR_DIR_CMNIST $EXPR_DIR_OSCN

# TRAINING
for SEED_INDEX_RAW in {0..9}; do
        SEED_INDEX=`expr $SEED_INDEX_RAW \* 1000`
        echo $SEED_INDEX

        # CMNIST
        my-python main.py  \
               --model "cmnist" \
               --experiment "${EXPR_DIR}/cmnist" \
               --obj "dreg" \
               --K 30 \
               --batch-size 128 \
               --epochs 30 \
               --learn-prior \
               --seed $SEED_INDEX

        # OSCN
        my-python main.py  \
               --model "oscn" \
               --experiment "${EXPR_DIR}/oscn" \
               --obj "elbo" \
               --K 30 \
               --batch-size 128 \
               --epochs 30 \
               --learn-prior \
               --seed $SEED_INDEX

        # CMNIST-OSCN
        my-python main.py \
               --model "cmnist_oscn" \
               --experiment "${EXPR_DIR}/cmnist-oscn" \
               --obj "dreg" \
               --K 30 \
               --batch-size 128 \
               --epochs 30 \
               --learn-prior \
               --seed $SEED_INDEX
done

# Analysis of CMNIST
dirs=`find "${EXPR_DIR}/cmnist/" -maxdepth 1 -mindepth 1 -type d | sed 's!^.*/!!'`
echo "Target run_id:" $dirs
for dir in $dirs; do
    EXPR_DIR_TMP="${EXPR_DIR}/cmnist/${dir}"
    OUTPUT_DIR_QUAL="${OUTPUT_DIR}/cmnist/${dir}/qualitative/"
    OUTPUT_DIR_QUAN="${OUTPUT_DIR}/cmnist/${dir}/quantitative/"
    echo "Experiment" $EXPR_DIR_TMP
    echo "Output:" $OUTPUT_DIR_QUAL
    mkdir -pv "${OUTPUT_DIR_QUAL}" "${OUTPUT_DIR_QUAN}"

    my-python visualize_latent.py \
           --model "cmnist" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 0 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_0_0.txt"
    my-python visualize_latent.py \
           --model "cmnist" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 1 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_0_1.txt"
done

# Analysis of OSCN
dirs=`find "${EXPR_DIR}/oscn/" -maxdepth 1 -mindepth 1 -type d | sed 's!^.*/!!'`
echo "Target run_id:" $dirs
for dir in $dirs; do
    EXPR_DIR_TMP="${EXPR_DIR}/oscn/${dir}"
    OUTPUT_DIR_QUAL="${OUTPUT_DIR}/oscn/${dir}/qualitative/"
    OUTPUT_DIR_QUAN="${OUTPUT_DIR}/oscn/${dir}/quantitative/"
    echo "Experiment" $EXPR_DIR_TMP
    echo "Output:" $OUTPUT_DIR_QUAL
    mkdir -pv "${OUTPUT_DIR_QUAL}" "${OUTPUT_DIR_QUAN}"

    my-python visualize_latent.py \
           --model "oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 0 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}oscn_0_0.txt"
    my-python visualize_latent.py \
           --model "oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 1 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}oscn_0_1.txt"
    my-python visualize_latent.py \
           --model "oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 2 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}oscn_0_2.txt"
done

# Analysis of CMNIST-OSCN
dirs=`find "${EXPR_DIR}/cmnist-oscn/" -maxdepth 1 -mindepth 1 -type d | sed 's!^.*/!!'`
echo "Target run_id:" $dirs
for dir in $dirs; do
    EXPR_DIR_TMP="${EXPR_DIR}/cmnist-oscn/${dir}"
    OUTPUT_DIR_QUAL="${OUTPUT_DIR}/cmnist-oscn/${dir}/qualitative/"
    OUTPUT_DIR_QUAN="${OUTPUT_DIR}/cmnist-oscn/${dir}/quantitative/"
    echo "Experiment" $EXPR_DIR_TMP
    echo "Output folder:" $OUTPUT_DIR_QUAL $OUTPUT_DIR_QUAN
    mkdir -pv "${OUTPUT_DIR_QUAL}" "${OUTPUT_DIR_QUAN}"

    # ANALYSIS
    my-python visualize_latent.py \
           --model "cmnist_oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 0 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_0_0.txt"
    my-python visualize_latent.py \
           --model "cmnist_oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 1 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_0_1.txt"
    my-python visualize_latent.py \
           --model "cmnist_oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 0 \
           --target-property 2 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_0_2.txt"
    my-python visualize_latent.py \
           --model "cmnist_oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 1 \
           --target-property 0 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_1_0.txt"
    my-python visualize_latent.py \
           --model "cmnist_oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 1 \
           --target-property 1 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_1_1.txt"
    my-python visualize_latent.py \
           --model "cmnist_oscn" \
           --pre-trained "${EXPR_DIR_TMP}" \
           --target-modality 1 \
           --target-property 2 \
           --output-dir $OUTPUT_DIR_QUAL > "${OUTPUT_DIR_QUAN}cmnist_oscn_1_2.txt"
done
