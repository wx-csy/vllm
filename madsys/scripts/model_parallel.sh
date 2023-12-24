#!/bin/bash

# This script is used to evaluate the model parallelism

model_path='./parts/model_config/llama-13b/'
output_path='./parts/workspace/data/vllm-13b.csv'
echo 'model_path,pp,tp,bsz,seq_len,test_iter,elapsed,throughput,latency' >> $output_path
tp=1
bsz_list=(256 192 128 96 64 48 32 24 16 12 8 6 4 3 2 1)
export CUDA_VISIBLE_DEVICES='4,5,6,7'


for bsz in ${bsz_list[@]}
do
    echo "running bsz: $bsz"
    python3 -m madsys.ubench --model_path $model_path -bsz $bsz -csv --out_file $output_path -tp $tp
done
