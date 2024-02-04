#! /bin/bash
export PYTHONPATH=.:/mnt/data-poseidon/sumire/repos/transformersum/src:$PYTHONPATH
src_side="src"
tgt_side="tgt"

#### Hyperparameter 
tgt_langs="de zh ar ja fr ko" 
context_sizes="8"
summarized_sizes="1 2 3 4"
context_side="src"

for tgt_lang in $tgt_langs; do
    for context_size in $context_sizes; do
        for summarized_size in $summarized_sizes; do
            if [ "$context_side" = "$src_side" ]; then 
                cfg_file="/home/sumire/thesis/LLM_Contextual_Prompt_MT/main/config/summarization/distilroberta/ted/$tgt_lang/$((context_size+1))-1to$((summarized_size+1))-1.yaml"
            else
                cfg_file="/home/sumire/thesis/LLM_Contextual_Prompt_MT/main/config/summarization/distilroberta/ted/$tgt_lang/1-$((context_size+1))to1-$((summarized_size+1)).yaml"
            fi  
            echo $cfg_file
            python main/eval_summ.py \
                --cfg "$cfg_file"
        done
    done
done
