#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH

data_dir=/mnt/data-poseidon/sumire/thesis/running
base_data_dir=/mnt/data-poseidon/sumire/thesis
model_type=Llama-2-70b-instruct-v2-usas-zs-p1-nsplit
langs="ar fr zh ja ko fr de"
context="9-1" 

for lang in $langs; do
    in_path_context=${data_dir}/ted/eval_mt/test/en-${lang}/${model_type}-${lang}-${context}
    #in_path_context=${base_data_dir}/$context/en-${lang}/cxmi-${model_type}-${lang}-${context}
    in_path_base=${base_data_dir}/1-1/en-${lang}/cxmi-${model_type}-${lang}-1-1 # test
    python ./evaluation/cxmi/cxmi.py \
        --in_path_context ${in_path_context} \
        --in_path_base ${in_path_base} \
        --out_path ${in_path_context}/cxmi.txt
done 

