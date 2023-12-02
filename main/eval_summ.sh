
#! /bin/sh
export PYTHONPATH=.:/mnt/data-poseidon/sumire/repos/transformersum/src:$PYTHONPATH

python main/eval_summ.py \
    --cfg /home/sumire/thesis/LLM_Contextual_Prompt_MT/main/config/summarization/distilroberta/ted/5-1.yaml