from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, XGLMTokenizerFast
from datasets import load_dataset, concatenate_datasets, load_from_disk
import evaluate
import numpy as np
import torch
from torch import tensor 
import os
from transformers import DataCollatorForLanguageModeling
from functools import partial
import json
from main.preprocess import preprocess_function, generate_prompt
from main.metrics import compute_metrics



training_args = Seq2SeqTrainingArguments(
    output_dir="./results/xglm",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_eval_batch_size=4, 
    do_predict=True,
    do_eval = True,
    predict_with_generate=True,
    include_inputs_for_metrics=True,
    warmup_steps = 500,
    #metric_for_best_model = "comet"
    #load_best_model_at_end = True,
    greater_is_better = True,
    #save_strategy = "steps",
    eval_delay = 0.0,
    eval_accumulation_steps = 20,
    label_names = ["labels"]
    
)
