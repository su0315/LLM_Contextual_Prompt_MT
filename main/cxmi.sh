#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH

data_dir=/mnt/data-poseidon/sumire/thesis/running
base_data_dir=/mnt/data-poseidon/sumire/thesis
model_type=Llama-2-70b-instruct-v2-usas-zs-p1-nsplit
lang=zh
context="1-8" 
in_path_context=${data_dir}/ted/eval_mt/test/en-${lang}/tgt-${model_type}-${lang}-${context}
in_path_base=${base_data_dir}/1-1/en-${lang}/cxmi-${model_type}-${lang}-1-1 # test
python /home/sumire/thesis/LLM_Contextual_Prompt_MT/main/cxmi.py \
    --in_path_context ${in_path_context} \
    --in_path_base ${in_path_base} \
    --out_path ${in_path_context}/cxmi.txt 

