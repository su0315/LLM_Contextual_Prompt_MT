#! /bin/sh

export PYTHONPATH=.:$PYTHONPATH:/mnt/data-poseidon/sumire/repos/MuDA/
muda_lang=ar
lang=ar
echo "$lang"
python /mnt/data-poseidon/sumire/repos/MuDA/muda/main.py \
    --src /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/$lang/ted_test_src.en \
    --tgt /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/$lang/ted_test_tgt.$lang \
    --docids /home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/$lang/ted_test.docids \
    --hyps /mnt/data-poseidon/sumire/thesis/1-1/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-1-1/translations.txt /mnt/data-poseidon/sumire/thesis/1-2/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-1-2/translations.txt /mnt/data-poseidon/sumire/thesis/1-3/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-1-3/translations.txt /mnt/data-poseidon/sumire/thesis/1-4/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-1-4/translations.txt /mnt/data-poseidon/sumire/thesis/1-5/en-$lang/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-$lang-1-5/translations.txt \
    --tgt-lang "$muda_lang"