from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, XGLMTokenizerFast, XGLMConfig#, LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets, load_from_disk
import evaluate
import numpy as np
import torch
from torch import tensor 
import os
from transformers import DataCollatorForLanguageModeling
from functools import partial
import json
from main.preprocess import preprocess_function, generate_prompt, preprocess_function_bsd, generate_prompt_bsd, generate_context
from main.metrics import compute_metrics
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from tqdm import tqdm

from transformers import  LlamaTokenizer, LlamaForCausalLM
        
tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
tokenizer.add_special_tokens({"pad_token":"<pad>"})
model = LlamaForCausalLM.from_pretrained(model_checkpoint, use_auth_token=True)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

print (tokenizer("\n"))