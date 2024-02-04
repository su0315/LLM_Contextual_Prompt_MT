#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH
src_side="src"
tgt_side="tgt"

data_dir=/mnt/data-poseidon/sumire/thesis/running
base_data_dir=/mnt/data-poseidon/sumire/thesis
model_type=Llama-2-70b-instruct-v2-sum-distilroberta-ted
tgt_langs="ar"
context_sizes="8"
summarized_sizes="1"
context_side="src"
echo "print"

for lang in $tgt_langs; do
    echo "$lang"
    in_path_base=/mnt/data-poseidon/sumire/thesis/1-1/en-${lang}/cxmi-Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-${lang}-1-1
    for context_size in $context_sizes; do
        for summarized_size in $summarized_sizes; do
            if [ "$context_side" = "$src_side" ]; then
                context="$((context_size+1))-1to$((summarized_size+1))-1"
                echo "aaaaaaaa"
            else
                context="$tgt_lang/1-$((context_size+1))to1-$((summarized_size+1))"
            fi
            in_path_context=${data_dir}/cont_summ_ted/eval_mt/en-${lang}/${model_type}-${lang}-${context}
            echo "$in_path_context"
            python /home/sumire/thesis/LLM_Contextual_Prompt_MT/main/cxmi.py \
                --in_path_context ${in_path_context} \
                --in_path_base ${in_path_base} \
                --out_path ${in_path_context}/cxmi.txt 
            
        done
    done
done
