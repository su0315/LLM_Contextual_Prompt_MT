from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM
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
    eval_accumulation_steps = 20
    
)

def main():
    # Parameters
    tgt_lang = "ja"
    data_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/"
    output_dir = "./results/playingaround"
    #model_checkpoint = "facebook/xglm-1.7B"
    #model_checkpoint = "facebook/xglm-2.9B"
    #model_checkpoint = "facebook/xglm-4.5B"
    model_checkpoint = "facebook/xglm-7.5B"
    #model_checkpoint = "bigscience/mt0-small"
    k = 3
    prompt_talk_id =  int(1548)
    
    tokenizer = XGLMTokenizer.from_pretrained(model_checkpoint,padding_side = 'left', max_new_tokens=250, max_length=250, truncation=True, padding='max_length')
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,padding_side = 'left', truncation=True, padding='max_length', max_new_tokens=250)
    
    model =  XGLMForCausalLM.from_pretrained(model_checkpoint)
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    max_length = 128

    data_files = { "test": f"{data_path}ted_en-{tgt_lang}"}
    dataset = load_dataset("json", data_files=data_files)
    #print ("Primt dataset"dataset["test"][1]["doc"]["en"])
    prompt = generate_prompt(dataset, tgt_lang, k, prompt_talk_id)
    """
    tokenized_datasets = dataset.map(
        partial(preprocess_function, tgt_lang, prompt, prompt_talk_id, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names,
        )
    tokenized_datasets.save_to_disk(f'./tokenized/ted_en-{tgt_lang}')
    """
    tokenized_datasets = load_from_disk(f'./tokenized/ted_en-{tgt_lang}')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        #train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=partial(compute_metrics, dataset, output_dir, tgt_lang, tokenizer),
        data_collator=data_collator,
        )
    model.eval()
    trainer.evaluate()
    

if __name__ == "__main__":
    main()

