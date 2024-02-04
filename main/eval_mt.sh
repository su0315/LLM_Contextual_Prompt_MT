#! /bin/bash
export PYTHONPATH=.:$PYTHONPATH:/home/sumire/text-generation-inference/clients/python
src_side="src"
tgt_side="tgt"

#### Hyperparameter ###
tgt_langs="zh" # later do Japanese from context size 2 or 3
context_sizes="8"
context_side="src"

for tgt_lang in $tgt_langs; do
    for context_size in $context_sizes; do
        #for summarized_size in $summarized_sizes; do
        if [ "$context_side" = "$src_side" ]; then 
            echo 
            cfg_file="./config/zs-p1-Llama-2-70b-instruct-v2/$tgt_lang/$((context_size+1))-1.yaml"
        else
            cfg_file="./config/zs-p1-Llama-2-70b-instruct-v2/$tgt_lang/1-$((context_size+1)).yaml"
        fi
        echo $cfg_file
        python ./evaluation/mt_quality/eval_mt.py \
            --cfg "$cfg_file"
        
    done
done
