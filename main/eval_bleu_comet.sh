#!/bin/sh

export PYTHONPATH=.:$PYTHONPATH

device=6
criterias="lexical_cohesion formality pronouns verb_form" 
context_sizes="2-1 3-1 4-1 5-1"
tgt_langs="ja"
data_dir=/mnt/data-poseidon/sumire/thesis

export CUDA_VISIBLE_DEVICES=$device 
for tgt_lang in $tgt_langs; do
    for context_size in $context_sizes; do
        for criteria in $criterias; do
            output_dir=$data_dir/$context_size/en-${tgt_lang}/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-${tgt_lang}-${context_size}
            echo $output_dir
            echo $criteria
            python main/eval_bleu_comet.py \
                --output_dir $output_dir \
                --criteria $criteria
        done
    done
done