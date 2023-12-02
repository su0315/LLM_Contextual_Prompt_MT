
#! /bin/sh
export PYTHONPATH=.:$PYTHONPATH:/mnt/data-poseidon/sumire/miniconda3:home/sumire/text-generation-inference
python main/eval_mt.py \
    --cfg /home/sumire/thesis/LLM_Contextual_Prompt_MT/main/config/zs-p1-Llama-2-70b-instruct-v2/summarized_context/ted/5-1to3-1.yaml