from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, XGLMTokenizerFast, LlamaTokenizer, LlamaModel
from datasets import load_dataset, concatenate_datasets, load_from_disk
import evaluate
import numpy as np
import torch
from torch import tensor 
import os
from transformers import DataCollatorForLanguageModeling
from functools import partial
import json
from main.preprocess import preprocess_function, generate_prompt, preprocess_function_bsd, generate_prompt_bsd
from main.metrics import compute_metrics
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"

def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for evaluating models.")
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--generic.tgt_lang", required=True, help="target language")
    #parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    #parser.add_argument("--cfg", action=ActionConfigFile)
    #parser.add_argument("--generic.src_lang", required=True)
    parser.add_argument("--generic.data_path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    #parser.add_argument("--generic.src_context",type=int, default="src", help="the number of the target context sentence for each input")
    #parser.add_argument("--generic.dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    #parser.add_argument("--generic.speaker", type=bool, default=False)
    #parser.add_argument("--generic.random_context", type=bool, default=False)
    #parser.add_argument("--generic.tag", type=bool, default=False)
    #parser.add_argument("--generic.output_dir", required=True, metavar="DIRECTORY", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.batch_size", type=int, default=0, help="the batch size of evaluation")
    parser.add_argument("--generic.model_checkpoint", required=True, metavar="FILE", help="model_checkpoint")
    parser.add_argument("--generic.k", type=int, default=0, help="the number of few shot")
    parser.add_argument("--generic.prompt_talk_id", type=int, default=0, help="the talk id to be used for few shot learning")
    parser.add_argument("--generic.max_new_tokens", type=int, default=0, help="max_new_tokens")
    parser.add_argument("--generic.max_length", type=int, default=0, help="max_length for input and labels")
    parser.add_argument("--generic.cfg_name", required=True, metavar="FILE", help="config file name")


    
    return parser

def main():
    parser = read_arguments()
    cfg = parser.parse_args()
    print(cfg)

    tgt_lang = cfg.generic.tgt_lang
    data_path = cfg.generic.data_path
    model_checkpoint = cfg.generic.model_checkpoint
    batch_size = cfg.generic.batch_size
    k = cfg.generic.k
    prompt_talk_id =  cfg.generic.prompt_talk_id
    max_new_tokens = cfg.generic.max_new_tokens
    max_length = cfg.generic.max_length
    cfg_name = cfg.generic.cfg_name

    if "llama" in model_checkpoint:
        tokenizer = LlamaTokenizerTokenizer.from_pretrained(model_checkpoint)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
        model = LlamaForCausalLM.from_pretrained(model_checkpoint)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
        model = XGLMForCausalLM.from_pretrained(model_checkpoint)
    
    
    if "iwslt_hf" in data_path:
        data_files = { "test": f"{data_path}ted_en-{tgt_lang}"}
        dataset = load_dataset("json", data_files=data_files)
    
        prompt = generate_prompt(dataset["test"], tgt_lang, k, prompt_talk_id)
        inputs = preprocess_function(tgt_lang, prompt, prompt_talk_id, max_length, tokenizer, dataset["test"]).input_ids
        labels = preprocess_function(tgt_lang, prompt, prompt_talk_id, max_length, tokenizer, dataset["test"]).labels
        output_dir = f"./results/{model_checkpoint}/ted/en-{tgt_lang}/{cfg_name}/"

    elif "BSD-master" in data_path:
        data_files = { "test":"/home/sumire/discourse_context_mt/data/BSD-master/test.json"}
        dataset = load_dataset("json", data_files=data_files)
        prompt = generate_prompt_bsd(dataset["test"], tgt_lang, k)
        inputs = preprocess_function_bsd(tgt_lang, prompt, max_length, tokenizer, dataset["test"]).input_ids
        labels = preprocess_function_bsd(tgt_lang, prompt, max_length, tokenizer, dataset["test"]).labels
        output_dir = f"./results/{model_checkpoint}/BSD/en-{tgt_lang}/{cfg_name}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Predict all in once
    #outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95) # 
    #predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print ("predictions", predictions)

    outputs_list = []
    decoded_preds_list = []
    decoded_labels_list = []
    decoded_input_ids_list = []
    results = []
    bleu_sum = 0
    comet_sum = 0
    gen_len_sum = 0

    model.eval()
    with torch.no_grad():
        num_batches = 0
        for batch in range(0, len(inputs), batch_size):
            num_batches += 1
            print ("batch", batch, "to", batch+batch_size)
            input = inputs[batch:batch+batch_size, :]
            print ("INPUT", tokenizer.batch_decode(input, skip_special_tokens=True))
            label = labels[batch:batch+batch_size, :]
            output = model.generate(input, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95) # if max_length only doesn't work, need to put max_new_tokens for XGLM model
            print ("generate is done")
            outputs_list.append(output)
            eval_preds = (output, label, input)
            result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(dataset, output_dir, tgt_lang, tokenizer, eval_preds)
            decoded_preds_list.append(decoded_preds)
            decoded_labels_list.append(decoded_labels)
            decoded_input_ids_list.append(decoded_input_ids)
            bleu_sum += result["bleu"]
            comet_sum += result["comet"]
            gen_len_sum += result["gen_len"]
            break

    # Store prediction inference
    with open(output_dir+'/translations.txt','w', encoding='utf8') as wf:
        for decoded_preds, outputs in zip(decoded_preds_list, outputs_list):
            for translation, ids in zip(decoded_preds, outputs):
                wf.write(translation.strip()+'\n')

    with open(output_dir+'/references.txt','w', encoding='utf8') as wf:
        for decoded_labels in (decoded_labels_list):
            for i in decoded_labels:
                for reference in i:
                    wf.write(reference.strip()+'\n')

    with open(output_dir+'/source.txt','w', encoding='utf8') as wf:
        for decoded_input_ids in (decoded_input_ids_list):
            for i in decoded_input_ids:
                for source in i:
                    wf.write(source.strip()+'\n')

    # Store the score
    with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
        bleu = bleu_sum / num_batches
        comet = comet_sum / num_batches
        gen_len = gen_len_sum/ num_batches

        wf.write(f"bleu: {bleu}\n") #ensure_ascii=False
        wf.write(f"comet: {comet}\n") 
        wf.write(f"gen_len: {gen_len}\n") 

    
    with open(output_dir+'/config','w', encoding='utf8') as wf:
        for i in [ tgt_lang, data_path, model_checkpoint, batch_size, k,prompt_talk_id, max_new_tokens, max_length, cfg_name]:
            wf.write(f"{i}\n")
    
    
if __name__ == "__main__":
    main()

