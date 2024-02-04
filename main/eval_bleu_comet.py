from transformers.models.llama.tokenization_llama import LlamaTokenizer # Adapted for new version transformers 
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM,  XGLMTokenizerFast, XGLMConfig
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
    parser.add_argument('--criteria', type=str, help='The criteria you select the sentences to be evaluated')
    args = parser.parse_args()
    output_dir = args.output_dir
    criteria = args.criteria
    return output_dir, criteria

def read_config(output_dir):
    # Initialize an empty dictionary to store the configuration
    config = {}
    try:
        with open(f"{output_dir}/config", 'r') as file:
            for line in file:
                print (line)
                if ":" in line and "prompt+source" not in line:
                    # Split each line at the colon (":") to separate the key and value
                    print (line)
                    key, value = line.strip().split(":")
                    # Store the key-value pair in the dictionary, trimming any extra whitespace
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"File '{output_dir}' not found.")

    tgt_lang = config.get('tgt_lang', None)
    data_path = config.get('data_path', None)
    api = config.get('api', None)
    model_checkpoint = config.get('model_checkpoint', None)
    prompt_type = int(config.get('prompt_type', 1))
    metrics = [ "sacrebleu", "comet"]#, "sacrebleu"
    cfg_name = config.get('cfg_name', None)

    return tgt_lang, data_path, api, model_checkpoint, prompt_type, metrics, cfg_name


def initialize_tokenizer(model_checkpoint, api):
    print (model_checkpoint)
    print (type(LlamaTokenizer))
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

def select_muda_tags(criteria, output_dir, tgt_lang, preds, sources, labels):
    muda_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted"
    muda_tagged_path = f"{muda_path}/{tgt_lang}/ted_en{tgt_lang}.tags"
    selected_preds = []
    ids = [] # sent ids 

    try:
        with open(muda_tagged_path, 'r') as file:
            tagged_data = json.load(file)

    except FileNotFoundError:
        print(f"File {muda_tagged_path} not found.")
        tagged_data = None

    sent_id = 0
    n_tagged_sents = 0
    for doc in tagged_data:
        for sent in doc:
            sent_id += 1
            sent_checked = False
            if sent_checked == False:
                for token in sent:
                    if token["tags"]:
                        tagged_token = token["token"]
                        tags = token["tags"]
                        for tag in tags:
                            if tag == criteria:
                                if sent_id not in ids:
                                    ids.append(sent_id)
                                    n_tagged_sents += 1
                                    sent_checked = True
            
                            
    print ("n_sents", sent_id)
    print ("n_tagged_sents", n_tagged_sents)                 
    print ("ids", ids)
    selected_preds= [preds[i-1] for i in ids]
    selected_refs = [labels[i-1] for i in ids]
    selected_srcs = [sources[i-1] for i in ids]

    try:
        if len(selected_preds) != len(selected_refs):
            print ("length are same")
    except NotImplementedError:
        print ("The length are not matching")
    
    return selected_preds, selected_srcs, selected_refs

            
def select_all_instances(criteria, output_dir):
    
    try:
        with open(f'{output_dir}/without_postprocess.txt' , 'r') as file:
            lines = file.readlines()

        # Initialize an empty list to store the extracted chunks
        selected_preds = []

        # Initialize an empty string to accumulate lines before "#########" lines
        current_chunk = ""
        
        #current_id = 0
        #ids = []

        for line in lines:
            line = line.strip()
            
            if line == "##########":
                #current_id += 1
                # Check if the current chunk is not empty and append it to the list
                if current_chunk != "" :
                    #if criteria in current_chunk:
                    selected_preds.append(current_chunk)
                    #ids.append(current_id)

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
            #selected_srcs = [sources[i-1] for i in ids]
            selected_srcs = sources

    except FileNotFoundError:
        print(f"File {output_dir}/sources.txt not found.")
        selected_srcs = None

    try:
        with open(f'{output_dir}/references.txt' , 'r') as file:
            references = file.readlines()
            #selected_refs = [references[i-1] for i in ids]
            selected_refs = references

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
    device, 
    tgt_lang,
    preds,
    sources,
    labels, 
    output_dir,
    prompt_type,
    metrics,
    criteria
    ):

    results = []
    num_batches = 0

    # Define mBART language code 
    lang_to_code = {"ja": "ja_XX", "ar":"ar_AR", "de":"de_DE", "fr":"fr_XX","ko":"ko_KR", "zh": "zh_CN"}

    if api: 
        # Evaluate
        eval_preds = (np.asarray(preds), np.asarray(labels), np.asarray(sources))
        result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(metrics, api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_preds, prompt_type)
        print (decoded_preds, decoded_labels)
        if criteria == "None":
            score_file_name = output_dir+'/test_score.txt'
            translation_file_name = output_dir+'/translations.txt'
        else:
            categorized_dir = output_dir+f"/categorized/{criteria}"
            if not os.path.exists(categorized_dir):
                os.makedirs(categorized_dir, exist_ok=True)
                
            score_file_name = f"{categorized_dir}/{criteria}_test_score.txt"
            translation_file_name = f"{categorized_dir}/{criteria}_translations.txt"

            # Write source and reference of the tagged sents
            with open(f"{categorized_dir}/{criteria}_src.txt",'w', encoding='utf8') as wf:
                for src in decoded_input_ids:
                    wf.write(src+'\n')
            with open(f"{categorized_dir}/{criteria}_ref.txt",'w', encoding='utf8') as wf:
                for label in decoded_labels:
                    for str_label in label:
                        wf.write(str_label.strip()+'\n')

        # Write the averaged score
        with open(score_file_name,'a', encoding='utf8') as wf:
            for metric in [  "sacrebleu", "comet"]:  #"sacrebleu",
                wf.write(f"{metric}: {result[metric]}\n") 

        # Write Translation Output after postprocess
        with open(translation_file_name,'w', encoding='utf8') as wf:
            for pred in decoded_preds:
                wf.write(pred+'\n')

def main():
    output_dir, criteria = read_output_dir()
    criteria_list = ["muda", "pronouns", "lexical_cohesion", "formality", "verb_form"]

    if criteria == "None" or criteria in criteria_list:
        tgt_lang, data_path, api, model_checkpoint, prompt_type, metrics, cfg_name = read_config(output_dir)
        
        # Initialize Model
        tokenizer = initialize_tokenizer(model_checkpoint, api)
        preds, sources, labels = select_all_instances(criteria, output_dir)

        if criteria in criteria_list:
            print (criteria)
            preds, sources, labels = select_muda_tags(criteria, output_dir, tgt_lang, preds, sources, labels)

        # Generate and Evaluate
        evaluate_instances(
            api,
            model_checkpoint, 
            tokenizer, 
            device, 
            tgt_lang,
            preds,
            sources,
            labels, 
            output_dir,
            prompt_type,
            metrics,
            criteria
            )
        print ("Evaluation Successful")
    
    else:
        print (f"No {criteria} found in this data.")


if __name__ == "__main__":
    main()
