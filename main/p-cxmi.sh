#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH

model_type=Llama-2-70b-instruct-v2-usas-zs-p1-nsplit
langs="ko"
contexts="4-1"

for lang in $langs; do
    for context in $contexts; do
        data_dir=/mnt/data-poseidon/sumire/thesis/$context
        in_path_context=${data_dir}/en-${lang}/cxmi-${model_type}-${lang}-${context} # test
        echo $in_path_context
        python ./evaluation/p_cxmi/p-cxmi.py \
            --in_path_context ${in_path_context} \
            --in_path_base /mnt/data-poseidon/sumire/thesis/1-1/en-${lang}/cxmi-${model_type}-${lang}-1-1 \
            --out_path ${in_path_context} \
            --muda_tag_path /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/${lang}/ted_en${lang}.tags

    done
done