#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH

data_dir=/mnt/data-poseidon/sumire/thesis/running
model_type=Llama-2-70b-instruct-v2-usas-zs-p1-nsplit
lang=ja
context="10-1" 
in_path_context=${data_dir}/ted/eval_mt/test/en-${lang}/cxmi-${model_type}-${lang}-${context}
python /home/sumire/thesis/LLM_Contextual_Prompt_MT/main/cxmi.py \
    --in_path_context ${in_path_context} \
    --in_path_base ${data_dir}/ted/eval_mt/test/en-${lang}/cxmi-${model_type}-${lang}-1-1 \
    --out_path ${in_path_context}/cxmi.txt 