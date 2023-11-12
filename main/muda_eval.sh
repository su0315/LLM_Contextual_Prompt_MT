#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH:/mnt/data-poseidon/sumire/repos/MuDA/
lang=zh
echo "$lang"
python /mnt/data-poseidon/sumire/repos/MuDA/muda/main.py \
    --src /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/$lang/ted_test_src.en \
    --tgt /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/$lang/ted_test_tgt.$lang \
    --docids /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/$lang/ted_test.docids \
    --hyps /mnt/data-poseidon/sumire/thesis/1-1/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-1-1/translations.txt /mnt/data-poseidon/sumire/thesis/2-1/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-2-1/translations.txt /mnt/data-poseidon/sumire/thesis/3-1/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-3-1/translations.txt /mnt/data-poseidon/sumire/thesis/4-1/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-4-1/translations.txt /mnt/data-poseidon/sumire/thesis/5-1/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-5-1/translations.txt \
    --tgt-lang "$lang"