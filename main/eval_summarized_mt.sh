
#! /bin/sh
export PYTHONPATH=.:$PYTHONPATH:/home/sumire/text-generation-inference/clients/python

#! /bin/bash
src_side="src"
tgt_side="tgt"


#### Hyperparameter ###
tgt_langs="de" # later do ja ko
context_sizes="8"
summarized_sizes="4"
context_side="src"


for tgt_lang in $tgt_langs; do
    for context_size in $context_sizes; do
        for summarized_size in $summarized_sizes; do
            if [ "$context_side" = "$src_side" ]; then 
                echo "aaaaaaaaaaaaaa"
                cfg_file="/home/sumire/thesis/LLM_Contextual_Prompt_MT/main/config/zs-p1-Llama-2-70b-instruct-v2/summarized_context/ted/$tgt_lang/$((context_size+1))-1to$((summarized_size+1))-1.yaml"
            else
                cfg_file="/home/sumire/thesis/LLM_Contextual_Prompt_MT/main/config/zs-p1-Llama-2-70b-instruct-v2/summarized_context/ted/$tgt_lang/1-$((context_size+1))to1-$((summarized_size+1)).yaml"
            fi
            echo $cfg_file
            python main/eval_mt.py \
                --cfg "$cfg_file"
        done
    done
done
