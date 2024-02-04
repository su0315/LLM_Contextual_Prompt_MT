#!/bin/sh

export PYTHONPATH=.:$PYTHONPATH
device=5
criterias="lexical_cohesion formality" 
context_sizes="1-2 2-3 1-4 1-5" 
tgt_langs="ko" 
data_dir=/mnt/data-poseidon/sumire/thesis

export CUDA_VISIBLE_DEVICES=$device 
echo "CHECK CUDA DEVICE"
echo $CUDA_VISIBLE_DEVICES

for tgt_lang in $tgt_langs; do
    for context_size in $context_sizes; do
        for criteria in $criterias; do
            output_dir=$data_dir/$context_size/en-${tgt_lang}/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-${tgt_lang}-${context_size}
            echo $output_dir
            echo $criterias
            python ./evaluation/mt_quality/eval_bleu_comet.py \
                --output_dir $output_dir \
                --criteria $criteria
        done
    done
done

