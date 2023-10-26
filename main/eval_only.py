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
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

def read_output_dir():
    parser = ArgumentParser(description='Read output directory of the model we evaluate.')
    parser.add_argument('--output_dir', type=str, help='The directory path')
    args = parser.parse_args()
    output_dir = args.output_dir
    return output_dir

def read_config(output_dir):
    # Initialize an empty dictionary to store the configuration
    config = {}
    
    try:
        with open(f"{output_dir}/config", 'r') as file:
            for line in file:

                if ":" in line and "prompt+source" not in line:
                    # Split each line at the colon (":") to separate the key and value
                    print (line)
                    key, value = line.strip().split(":")
                    # Store the key-value pair in the dictionary, trimming any extra whitespace
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"File '{output_dir}' not found.")

    # Now you have the configuration as a dictionary
    # You can access individual elements like this: ################# Correct CO
    tgt_lang = config.get('tgt_lang', None)
    data_path = config.get('data_path', None)
    src_context_size = int(config.get('src_context_size', None))
    api = config.get('api', None)
    model_checkpoint = config.get('model_checkpoint', None)
    batch_size = int(config.get('batch_size', None))
    max_new_tokens = int(config.get('max_new_tokens', None))
    prompt_type = int(config.get('prompt_type', None))
    max_length = int(config.get('max_length', None))
    cfg_name = config.get('cfg_name', None)
    k = int(config.get('k', None))

    return tgt_lang, data_path, src_context_size, api, model_checkpoint, batch_size, k, prompt_type, max_new_tokens, max_length, cfg_name


def initialize_tokenizer(model_checkpoint, api):

    if "llama" in model_checkpoint:
        tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
   
    if api:
        tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)

    elif "xglm" in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
        #configuration = XGLMConfig()
        #model = XGLMForCausalLM(configuration).from_pretrained(model_checkpoint) 

    elif "mbart" in model_checkpoint:
        from transformers import MBartConfig, MBart50Tokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
        #configuration = MBartConfig()
        tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
        tokenizer.src_lang="en_XX"
        #model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

    return tokenizer 

def select_instances(criteria, output_dir):
    
    try:
        with open(f'{output_dir}/without_postprocess.txt' , 'r') as file:
            lines = file.readlines()

        # Initialize an empty list to store the extracted chunks
        selected_preds = []

        # Initialize an empty string to accumulate lines before "#########" lines
        current_chunk = ""
        
        current_id = 0
        ids = []

        for line in lines:
            line = line.strip()
            
            if line == "##########":
                current_id += 1
                # Check if the current chunk is not empty and append it to the list
                if current_chunk != "" :
                    #if criteria in current_chunk:
                    selected_preds.append(current_chunk)
                    ids.append(current_id)

                # Reset the current chunk
                current_chunk = ""
            else:
                # Accumulate lines before "#########" lines in the current_chunk
                current_chunk += line 

        # Append the last chunk if it exists
        if current_chunk != "":
            selected_preds.append(current_chunk)

    except FileNotFoundError:
        print(f"File {output_dir}/without_postprocess.txt not found.")
        selected_preds = []

    # Select source/reference with the selected ids

    try:
        with open(f'{output_dir}/source.txt' , 'r') as file:
            sources = file.readlines()
            selected_srcs = [sources[i-1] for i in ids]

    except FileNotFoundError:
        print(f"File {output_dir}/sources.txt not found.")
        selected_srcs = None

    try:
        with open(f'{output_dir}/references.txt' , 'r') as file:
            references = file.readlines()
            selected_refs = [references[i-1] for i in ids]

    except FileNotFoundError:
        print(f"File {output_dir}/references.txt not found.")
        selected_refs = None

    
    try:
        if len(selected_preds) != len(selected_refs):
            print ("length are same")
    except NotImplementedError:
        print ("The length are not matching")
    

    return selected_preds, selected_srcs, selected_refs

def evaluate_instances(
    api,
    model_checkpoint, 
    tokenizer, 
    batch_size,  
    max_new_tokens, 
    max_length, 
    device, 
    tgt_lang,
    preds,
    sources,
    labels, 
    output_dir,
    prompt_type
    ):

    results = []
    num_batches = 0

    # Define mBART language code 
    lang_to_code = {"ja": "ja_XX", "ar":"ar_AR", "de":"de_DE", "fr":"fr_XX","ko":"ko_KR", "zh": "zh_CN"}

    if api: 

        # Evaluate
        eval_preds = (np.asarray(preds), np.asarray(labels), np.asarray(sources))
        result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_preds, prompt_type)


        with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
            
            bleu_score = result["bleu"]
            comet_score = result["comet"]
            gen_len_score = result["gen_len"]

            for metric, score in zip(["bleu", "comet", "gen_len"], [bleu_score, comet_score, gen_len_score]):
                wf.write(f"{metric}: {score}\n") 

        with open(output_dir+'/translations.txt','w', encoding='utf8') as wf:
            for pred in decoded_preds:
                wf.write(pred.strip()+'\n')

        """
        with open(output_dir+'/src_with_b.txt','w', encoding='utf8') as wf:
            for src in decoded_input_ids:
                wf.write(src.strip()+'\n')

        with open(output_dir+'/ref_with_b.txt','w', encoding='utf8') as wf:
            for ref in decoded_labels:
                wf.write(f"{ref}\n")
        """

def main():
    output_dir = read_output_dir()
    tgt_lang, data_path, src_context_size, api, model_checkpoint, batch_size, k, prompt_type, max_new_tokens, max_length, cfg_name = read_config(output_dir)
    
    # Initialize Model
    tokenizer = initialize_tokenizer(model_checkpoint, api)
    criteria = None
    preds, sources, labels = select_instances(criteria, output_dir)

    # Generate and Evaluate
    evaluate_instances(
        api,
        model_checkpoint, 
        tokenizer, 
        batch_size, 
        max_new_tokens, 
        max_length, 
        device, 
        tgt_lang,
        preds,
        sources,
        labels, 
        output_dir,
        prompt_type
        )
    print ("Evaluation Successful")

if __name__ == "__main__":
    main()
