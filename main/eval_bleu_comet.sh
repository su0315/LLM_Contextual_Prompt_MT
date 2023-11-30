#!/bin/sh

export PYTHONPATH=.:$PYTHONPATH

device=6
criterias="None" #"lexical_cohesion formality pronouns verb_form" 
context_sizes="1-6 1-7 1-8"
tgt_langs="zh ja de fr"
data_dir=/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/test
#/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/test/en-ko/tgt-Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ko-1-4
export CUDA_VISIBLE_DEVICES=$device 
for tgt_lang in $tgt_langs; do
    for context_size in $context_sizes; do
        #for criteria in $criterias; do
        output_dir=$data_dir/en-${tgt_lang}/tgt-Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-${tgt_lang}-${context_size}
        echo $output_dir
        echo $criterias
        python main/eval_bleu_comet.py \
            --output_dir $output_dir \
            --criteria $criterias
        #done
    done
done
