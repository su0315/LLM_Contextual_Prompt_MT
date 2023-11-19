#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH

#data_dir=/mnt/data-poseidon/sumire/thesis/running
model_type=Llama-2-70b-instruct-v2-usas-zs-p1-nsplit
langs="ko ja de fr ar zh"
contexts="1-2 1-3 1-4 1-5"

for lang in $langs; do
    for context in $contexts; do
        data_dir=/mnt/data-poseidon/sumire/thesis/$context
        in_path_context=${data_dir}/en-${lang}/${model_type}-${lang}-${context}
        echo $in_path_context
        python /home/sumire/thesis/LLM_Contextual_Prompt_MT/main/p-cxmi.py \
            --in_path_context ${in_path_context} \
            --in_path_base /mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/test/en-${lang}/cxmi-${model_type}-${lang}-1-1 \
            --out_path ${in_path_context}/p-cxmi.txt \
            --muda_tag_path /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/${lang}/ted_en${lang}.tags

    done
done