from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, XGLMTokenizerFast, XGLMConfig, LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets, load_from_disk
import evaluate
import numpy as np
import torch
from torch import tensor 
import os
from transformers import DataCollatorForLanguageModeling
from functools import partial
import json
from main.preprocess import preprocess_function, generate_few_shots, preprocess_function_bsd, generate_prompt_bsd
from main.metrics import compute_metrics
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
import argparse                         
from tqdm import tqdm
from eval_only import read_config, select_instances, initialize_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

def read_output_dirs():
    parser = ArgumentParser(description='Read output directory of the model we evaluate.')
    parser.add_argument('--output_dir_with_c', type=str, help='The directory path with context')
    parser.add_argument('--output_dir_without_c', type=str, help='The directory path without context')
    parser.add_argument('--metrics',  type=str, help = "Comma-separated list of strings", default= "sacrebleu,comet", required=False)
    args = parser.parse_args()
    output_dir_with_c = args.output_dir_with_c
    output_dir_without_c = args.output_dir_without_c
    metrics = args.metrics.split(",")
    print ("metrics", metrics)
    return output_dir_with_c, output_dir_without_c, metrics

def compare_score(
    preds_with_c,
    preds_without_c, 
    metrics,
    api,
    model_checkpoint, 
    tokenizer,
    batch_size, 
    max_new_tokens, 
    max_length, 
    device, 
    tgt_lang,
    sources,
    labels, 
    output_dir_with_c,
    output_dir_without_c,
    prompt_type
    ):

    # Define mBART language code 
    lang_to_code = {"ja": "ja_XX", "ar":"ar_AR", "de":"de_DE", "fr":"fr_XX","ko":"ko_KR", "zh": "zh_CN"}
    #bleu_binaries = []
    #comet_binaries = []
    #bleu_score_diffs = []
    #comet_score_diffs = []

    if api: 
        for pred_with_c, pred_without_c, label, source in zip(preds_with_c[:2], preds_without_c[:2], labels[:2], sources[:2]):

            # Evaluate
            eval_pred_with_c = ([pred_with_c], [label], [source])
            eval_pred_without_c = ([pred_without_c], [label], [source])
            result_with_c, decoded_pred_with_c, decoded_label, decoded_input_id = compute_metrics(metrics, api, model_checkpoint, output_dir_with_c, tgt_lang, tokenizer, eval_pred_with_c, prompt_type)
            result_without_c, decoded_pred_without_c, decoded_label, decoded_input_id = compute_metrics(metrics, api, model_checkpoint, output_dir_without_c, tgt_lang, tokenizer, eval_pred_without_c, prompt_type)
            
            for metric in metrics:
                score_with_c = result_with_c[metric]
                score_without_c = result_without_c[metric]
                
                score_diff = np.round(score_with_c - score_without_c, 4)
                if score_diff > 0:
                    binary = 1 
                else:
                    binary = 0
                """
                comet_score_diff = np.round(comet_score_with_c - comet_score_without_c, 4)
                #comet_score_diffs.append(comet_score_diff)
                if comet_score_diff > 0:
                    comet_binary = 1
                else:
                    comet_binary = 0
                """
            
                with open(output_dir_with_c+f'/{metric}_binary.txt','a', encoding='utf8') as wf:
                    #for i in bleu_score_diffs:
                    wf.write(str(binary)+ "\n")
                """
                with open(output_dir_with_c+f'/{metric}_binary.txt','a', encoding='utf8') as wf:
                    #for i in comet_score_diffs:
                    wf.write(str(comet_binary)+ "\n")
                """

                with open(output_dir_with_c+f'/{metric}_score_diff.txt','a', encoding='utf8') as wf:
                    #for i in bleu_score_diffs:
                    wf.write(str(score_diff)+ "\n")
                """
                with open(output_dir_with_c+'/comet_score_diff.txt','a', encoding='utf8') as wf:
                    #for i in comet_score_diffs:
                    wf.write(str(comet_score_diff)+ "\n")
                """

def main():
    output_dir_with_c, output_dir_without_c, metrics = read_output_dirs()
    
    tgt_lang, data_path, src_context_size_with_c, api, model_checkpoint, batch_size, k, prompt_type, max_new_tokens, max_length, cfg_name_with_c = read_config(output_dir_with_c)
    tgt_lang, data_path, src_context_size_without_c, api, model_checkpoint, batch_size, k, prompt_type, max_new_tokens, max_length, cfg_name_with_c = read_config(output_dir_without_c)
    print ("src_context_size_with_c:", src_context_size_with_c, "src_context_size_with_c:", src_context_size_without_c)
    
    # Initialize Model
    tokenizer = initialize_tokenizer(model_checkpoint, api)
    criteria = None
    preds_with_c, sources, labels = select_instances(criteria, output_dir_with_c)
    preds_without_c, sources, labels = select_instances(criteria, output_dir_without_c)

    # Generate and Evaluate
    score_differences = compare_score(
        preds_with_c,
        preds_without_c,
        metrics,
        api,
        model_checkpoint,
        tokenizer, 
        batch_size, 
        max_new_tokens, 
        max_length, 
        device, 
        tgt_lang,
        sources,
        labels, 
        output_dir_with_c,
        output_dir_without_c,
        prompt_type
        )
    print ("Evaluation Successful")

if __name__ == "__main__":
    main()
