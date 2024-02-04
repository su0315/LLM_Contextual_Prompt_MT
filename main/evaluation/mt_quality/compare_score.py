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
from preprocess import preprocess_function, generate_few_shots, preprocess_function_bsd, generate_prompt_bsd
from metrics import compute_metrics
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

def compare_and_and_store_score(score_with_c, score_without_c, metric, output_dir_with_c):
    score_diff = np.round(score_with_c - score_without_c, 4)
    if score_diff > 0:
        binary = 1 
    else:
        binary = 0

    with open(output_dir_with_c+f'/{metric}_binary.txt','a', encoding='utf8') as wf:
        #for i in bleu_score_diffs:
        wf.write(str(binary)+ "\n")

    with open(output_dir_with_c+f'/{metric}_score_diff.txt','a', encoding='utf8') as wf:
        #for i in bleu_score_diffs:
        wf.write(str(score_diff)+ "\n")

def compare_score(
    preds_with_c,
    preds_without_c, 
    metrics,
    with_c_metrics,
    without_c_metrics,
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

    if api: 
        
        # Read all exisiting data
        all_read_scores = {}
        
        for with_c_or_without_c, output_dir, metrics_for_each_model in zip(["with_c", "without_c"],[output_dir_with_c, output_dir_without_c], [with_c_metrics, without_c_metrics]):
            all_read_scores[with_c_or_without_c] = {}
            
            for metric in metrics:
                if metric not in metrics_for_each_model:
                    scores = []
                    with open(output_dir+f'/{metric}_each_score.txt','r', encoding='utf8') as rf:
                        for line in rf:
                            score = float(line.strip())
                            scores.append(score)
                    all_read_scores[with_c_or_without_c][metric] = scores

        ### Calculating both models for selected metric for each
        if preds_without_c != None and preds_with_c != None:
            print ("calculating both c and without c ")

            for sent_id, pred_with_c, pred_without_c, label, source in zip(range(len(labels)), preds_with_c, preds_without_c, labels, sources):

                # Evaluate
                eval_pred_with_c = ([pred_with_c], [label], [source])
                result_with_c, decoded_pred_with_c, decoded_label, decoded_input_id = compute_metrics(with_c_metrics, api, model_checkpoint, output_dir_with_c, tgt_lang, tokenizer, eval_pred_with_c, prompt_type)
                eval_pred_without_c = ([pred_without_c], [label], [source])
                result_without_c, decoded_pred_without_c, decoded_label, decoded_input_id = compute_metrics(without_c_metrics, api, model_checkpoint, output_dir_without_c, tgt_lang, tokenizer, eval_pred_without_c, prompt_type)     
                
                for metric in metrics:
                    if metric in with_c_metrics:
                        score_with_c = result_with_c[metric]
                        with open(output_dir_with_c+f'/{metric}_each_score.txt','a', encoding='utf8') as wf:
                            wf.write(str(score_with_c)+ "\n") 
                          
                    else:
                        score_with_c = all_read_scores["with_c"][metric][sent_id]

                    if metric in without_c_metrics: # without_c hasnt got the metric before. so store and write it
                        score_without_c = result_without_c[metric] 
                        
                        with open(output_dir_without_c+f'/{metric}_each_score.txt','a', encoding='utf8') as wf:
                            wf.write(str(score_without_c)+ "\n")   

                    else: # without_c has metric already, so read it and store 
                        score_without_c = all_read_scores["without_c"][metric][sent_id]
                    
                    compare_and_and_store_score(score_with_c, score_without_c, metric, output_dir_with_c)
        
        
        ### Caluculating only with_C model
        elif preds_without_c == None and preds_with_c != None:
            print ("calculating only with_C")
            
            # Testing the length of read data 
            for metric in metrics:
                without_c_len = len(all_read_scores["without_c"][metric])
                with_c_len = len(preds_with_c)
                
                if with_c_len == without_c_len:
                    print ("length match")

                else:
                    raise ValueError(f"The length of C model ({metric}) and without C model({metric}) is different")
            
            for sent_id, pred_with_c, label, source in zip(range(len(labels)), preds_with_c, labels, sources):

                # Evaluate
                eval_pred_with_c = ([pred_with_c], [label], [source])
                result_with_c, decoded_pred_with_c, decoded_label, decoded_input_id = compute_metrics(with_c_metrics, api, model_checkpoint, output_dir_with_c, tgt_lang, tokenizer, eval_pred_with_c, prompt_type)        
                
                for metric in metrics:
                    if metric in with_c_metrics:
                        score_with_c = result_with_c[metric]
                        with open(output_dir_with_c+f'/{metric}_each_score.txt','a', encoding='utf8') as wf:
                            wf.write(str(score_with_c)+ "\n") 
                    else:
                        score_with_c = all_read_scores["with_c"][metric][sent_id]
        
                    score_without_c = all_read_scores["without_c"][metric][sent_id]
                    
                    print ("score_without_c", score_without_c)
                    print ("score with c", score_with_c)
                    compare_and_and_store_score(score_with_c, score_without_c, metric, output_dir_with_c)
        

        ### Caluculating only Without C
        elif preds_without_c != None and preds_with_c == None:
            print ("calculating only withoutC")

            # Testing the length of read data 
            for metric in metrics:
                with_c_len = len(all_read_scores["with_c"][metric])
                without_c_len = len(preds_without_c)
                
                if with_c_len == without_c_len:
                    print ("length match")

                else:
                    raise ValueError(f"The length of C model ({metric}) and without C model({metric}) is different")
            
            for sent_id, pred_without_c, label, source in zip(range(len(labels)), preds_without_c, labels, sources):

                # Evaluate
                eval_pred_without_c = ([pred_without_c], [label], [source])
                result_without_c, decoded_pred_without_c, decoded_label, decoded_input_id = compute_metrics(without_c_metrics, api, model_checkpoint, output_dir_without_c, tgt_lang, tokenizer, eval_pred_without_c, prompt_type)     
                
                for metric in metrics:

                    if metric in without_c_metrics:
                        score_without_c = result_without_c[metric]

                        with open(output_dir_without_c+f'/{metric}_each_score.txt','a', encoding='utf8') as wf:
                            wf.write(str(score_without_c)+ "\n") 
                    
                    else:
                        score_without_c = all_read_scores["without_c"][metric][sent_id]

                    score_with_c = all_read_scores["with_c"][metric][sent_id] 
                    compare_and_and_store_score(score_with_c, score_without_c, metric, output_dir_with_c)
        

        ### Not calculating without_c metrics at all, just read them
        elif preds_without_c == None and preds_with_c == None: 
            print ("no calculations at all")

            for metric in metrics:
                with_c_len = len(all_read_scores["without_c"][metric])
                without_c_len = len(all_read_scores["with_c"][metric])
                
                if with_c_len == without_c_len:
                    
                    for sent_id in range(len(all_read_scores["without_c"][metric])):
                        score_without_c = all_read_scores["without_c"][metric][sent_id]
                        score_with_c = all_read_scores["with_c"][metric][sent_id] 
                        compare_and_and_store_score(score_with_c, score_without_c, metric, output_dir_with_c)

                else:
                    raise ValueError(f"The length of C model ({metric}) and without C model({metric}) is different")
def main():
    output_dir_with_c, output_dir_without_c, metrics = read_output_dirs()
    
    tgt_lang, data_path, src_context_size_with_c, api, model_checkpoint, batch_size, k, prompt_type, max_new_tokens, max_length, cfg_name_with_c = read_config(output_dir_with_c)
    tgt_lang, data_path, src_context_size_without_c, api, model_checkpoint, batch_size, k, prompt_type, max_new_tokens, max_length, cfg_name_with_c = read_config(output_dir_without_c)
    print ("src_context_size_with_c:", src_context_size_with_c, "src_context_size_with_c:", src_context_size_without_c)
    
    # Initialize Model
    tokenizer = initialize_tokenizer(model_checkpoint, api)
    criteria = None
    preds_with_c, sources, labels = select_instances(criteria, output_dir_with_c)
    
    # Check if we need to get prediction and evaluate without_c in the metrics
    without_c_metrics = []
    with_c_metrics = []

    for m in metrics:
        each_score_file_name = f'{m}_each_score.txt' 
        each_score_path_without_c = os.path.join(output_dir_without_c, each_score_file_name)   
        each_score_path_with_c = os.path.join(output_dir_with_c, each_score_file_name)  

        #print (each_score_path_without_c)
        if not os.path.exists(each_score_path_without_c):
            without_c_metrics.append(m)
            print (m, "needs to be calculated for withoutc")
        else:
            print (m, "already there for without c")
        
        if not os.path.exists(each_score_path_with_c):
            print (m, "needs to be calculated for withc")
            with_c_metrics.append(m)
        else:
            print (m, "already there for with c")
        
    if len(without_c_metrics) > 0:
        preds_without_c, sources, labels = select_instances(criteria, output_dir_without_c)
    else:
        print ("no for without_c")
        preds_without_c = None
    
    if len(with_c_metrics) > 0:
        preds_with_c, sources, labels = select_instances(criteria, output_dir_with_c)
    else:
        print ("no for with_c")
        preds_with_c = None

    # Generate and Evaluate
    compare_score(
        preds_with_c,
        preds_without_c,
        metrics,
        with_c_metrics,
        without_c_metrics,
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
