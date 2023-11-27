#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15G:ngpus=1:host=cpsy-t5820
#PBS -N MMVAE
#PBS -j oe

echo "working directory: " $PBS_O_WORKDIR
echo "omp thread num: " $OMP_NUM_THREADS
echo "ncpus: " $NCPUS
echo "cuda visible devices: " $CUDA_VISIBLE_DEVICES

date +"%Y/%m/%d %p %I:%M:%S"
start_time=`date +%s`
cd $PBS_O_WORKDIR

./run_replication_all.sh

date +"%Y/%m/%d %p %I:%M:%S"
end_time=`date +%s`
run_time=$((end_time - start_time))

echo "start time: " $start_time
echo "end time: " $end_time
echo "run time: " $run_time
