

from extractive import ExtractiveSummarizer
import torch
import os 
from transformers import AutoConfig, AutoModel, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk
from preprocess_summ import preprocess_function_contrapro, preprocess_function_iwslt
from metrics_summ import compute_metrics

import numpy as np
import torch
import os
import json
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from tqdm import tqdm

def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for evaluating models.")
    parser.add_argument("--cfg", action=ActionConfigFile)
    #parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--generic.tgt_lang", required=True, type=str)
    parser.add_argument("--generic.data_path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.src_context", default="src", help="the number of the source context sentence for each input")
    parser.add_argument("--generic.tgt_context", default="src", help="the number of the target context sentence for each input")
    #parser.add_argument("--generic.dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    #parser.add_argument("--generic.speaker", type=bool, default=False)
    #parser.add_argument("--generic.random_context", type=bool, default=False)
    #parser.add_argument("--generic.tag", type=bool, default=False)
    #parser.add_argument("--generic.output_dir", required=True, metavar="DIRECTORY", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.batch_size", type=int, default=0, help="the batch size of evaluation")
    parser.add_argument("--generic.model_checkpoint", required=True, metavar="FILE", help="model_checkpoint")
    #parser.add_argument("--generic.k", type=int, default=0, help="the number of few shot")
    #parser.add_argument("--generic.cfg_name", required=True, metavar="FILE", help="config file name")
    parser.add_argument("--generic.api", type=bool, default=False, metavar="FILE", help="Whether using text generation api or not")
    parser.add_argument("--generic.device",type=int, default=6,  help="The GPU device id")
    parser.add_argument("--generic.num_summary_sentences",  type=int, default=1, help="The num_summary_sentences")
    parser.add_argument("--generic.prompt_type",  type=int, default=0, help="The type of prompt, if none 0")
    return parser


def initialize_model(model_checkpoint, api, device):
    """
    if api is True:
        from text_generation import Client

        TGI_CENTRAL_ADDRESS="localhost:8765"
        models = Client.list_from_central(central_url=f"http://{TGI_CENTRAL_ADDRESS}")
        
        
        if "Llama-2-70b-instruct-v2" in model_checkpoint:
            model_name = None
            for i in range(len(models)):
                if models[i]["name"] == "upstage/Llama-2-70b-instruct-v2":
                    model_name, model_addr = models[i]["name"], models[i]["address"]
                    print (model_name, model_addr)
                    model = Client("http://" + model_addr)
                    model.timeout = 500 # Increasing timeout in seconds, Client class: self.timeout = 10 in default             
                    tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)

                
            if model_name is None:
                raise Exception('model upstage/Llama-2-70b-instruct-v2 is not available.')
    """
    if "transformersum" in model_checkpoint:
     
        model = ExtractiveSummarizer.load_from_checkpoint(model_checkpoint).to(device)
        #print (model.hparams)
    return model

def read_data(
    data_path,   
    api,
    model_checkpoint, 
    src_context_size,
    tgt_context_size,
    cfg_name,
    tgt_lang,
    prompt_type,
    num_summary_sentences
    ):

    if "ContraPro" in data_path:
        inputs, labels, sources = preprocess_function_contrapro(data_path, model_checkpoint, src_context_size, api)
        output_dir = f"/mnt/data-poseidon/sumire/thesis/running/summarization/contrapro/{cfg_name}/"
        labels = np.asarray(labels)
        sources = np.asarray(sources)
        
    if "iwslt_hf" in data_path:
        data_files = { "train": f"{data_path}train_ted_en-{tgt_lang}",  "val": f"{data_path}val_ted_en-{tgt_lang}",  "test": f"{data_path}test_ted_en-{tgt_lang}"}
        dataset = load_dataset("json", data_files=data_files)
        #sources = np.asarray([sent for doc in dataset["test"]["doc"] for sent in doc["en"]])
        #labels = np.asarray([sent for doc in dataset["test"]["doc"] for sent in doc[tgt_lang]])
        sources=None
        labels=None
        if src_context_size > 0:
            output_dir = f"/mnt/data-poseidon/sumire/thesis/running/summarization/ted/en-{tgt_lang}/{src_context_size+1}-1to{num_summary_sentences+1}-1/{cfg_name}/"
        if tgt_context_size > 0:
            output_dir = f"/mnt/data-poseidon/sumire/thesis/running/summarization/ted/en-{tgt_lang}/1-{tgt_context_size+1}to1-{num_summary_sentences+1}/{cfg_name}/"
        
        if api is True:
            inputs = preprocess_function_iwslt(src_context_size, tgt_context_size, tgt_lang, api, model_checkpoint, prompt_type, dataset["test"])
        
        else:
            inputs = preprocess_function_iwslt(src_context_size, tgt_context_size, tgt_lang, api, model_checkpoint, prompt_type, dataset["test"]).input_ids
    
    else:
        print ("preprocess not defined")

    return sources, inputs, labels, output_dir

def evaluate_summarization(
    data_path,
    api,
    model_checkpoint, 
    model, 
    batch_size,  
    device, 
    sources,
    inputs, 
    labels, 
    num_summary_sentences,
    output_dir,
    ):

    
    
    if "ContraPro" in data_path:
        all_preds = []
        all_labels = []
        all_srcs = []

        # Generate
        
        for inp, label, src in zip(inputs, labels, sources): 
            pred = model.predict(inp, raw_scores=False, num_summary_sentences=num_summary_sentences)
            
            all_preds.append(pred)
            all_labels.append(label)
            all_srcs.append(src)
            
            if "llama" in model_checkpoint or "Llama" in model_checkpoint:
                with open(output_dir+'/without_postprocess.txt','a', encoding='utf8') as wf:
                    wf.write(pred.strip()+'\n##########\n') # if with batch maybe need to adapt

                with open(output_dir+'/prompt+source.txt','a', encoding='utf8') as wf:
                    wf.write(inp.strip()+'\n')


            with open(output_dir+'/references.txt','a', encoding='utf8') as wf:
                wf.write(label.strip()+'\n')

            with open(output_dir+'/source.txt','a', encoding='utf8') as wf:
                wf.write(src.strip()+'\n')

        
        # Evaluate    
        eval_preds = (np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_srcs))
        result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(api, model_checkpoint, output_dir, eval_preds)

        
        with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
            rouge_score = result

            for metric, score in zip(["rouge"], [rouge_score]):
                wf.write(f"{metric}: {score}\n") 

        with open(output_dir+'/summarized_contexts.txt','a', encoding='utf8') as wf:
            for pred in decoded_preds:
                wf.write(pred.strip()+'\n')
                
    else:
        all_preds = []
        all_contexts = []

        # Generate
        
        for inp in inputs: 
            if inp != "":
                pred = model.predict(inp, raw_scores=False, num_summary_sentences=num_summary_sentences)
                all_contexts.append(inp)
            else:
                pred = " "
                all_contexts.append(" ")
            all_preds.append(pred)
            
            
            if "llama" in model_checkpoint or "Llama" in model_checkpoint:
                with open(output_dir+'/without_postprocess.txt','a', encoding='utf8') as wf:
                    wf.write(pred.strip()+'\n##########\n') # if with batch maybe need to adapt
        

                with open(output_dir+'/prompt+source.txt','a', encoding='utf8') as wf:
                    wf.write(inp.strip()+'\n')

        with open(output_dir+'/source.txt','a', encoding='utf8') as wf:
            for cont in all_contexts :
                wf.write(cont.strip()+'\n')
                
        decoded_preds = np.asarray(all_preds)
        with open(output_dir+'/summarized_contexts.txt','a', encoding='utf8') as wf:
            for pred in decoded_preds:
                wf.write(pred.strip()+'\n')
        
    

def main():
    parser = read_arguments()
    cfg = parser.parse_args()

    data_path = cfg.generic.data_path
    src_context_size = cfg.generic.src_context
    tgt_context_size = cfg.generic.tgt_context
    model_checkpoint = cfg.generic.model_checkpoint
    batch_size = cfg.generic.batch_size
    #cfg_name = cfg.generic.cfg_name
    api = cfg.generic.api
    num_summary_sentences = cfg.generic.num_summary_sentences
    prompt_type = cfg.generic.prompt_type
    device = torch.device(f'cuda:{cfg.generic.device}' if torch.cuda.is_available() else 'cpu')
    tgt_lang=cfg.generic.tgt_lang
    
    if "distilroberta" in model_checkpoint:
        summarized_contexts = "distilroberta"
    if "iwslt" in data_path:
        data_name = "ted"
    if src_context_size > 0:
        context_size = f"{src_context_size+1}-1to{num_summary_sentences+1}-1"
    elif tgt_context_size > 0:
        context_size = f"1-{tgt_context_size+1}to1-{num_summary_sentences+1}"
    cfg_name = f"transforemersum-{summarized_contexts}-{data_name}-{tgt_lang}-{context_size}"
    print (cfg_name)
  
    # Initialize Model
    
    model = initialize_model(model_checkpoint, api, device)
    # Load Dataset
    sources, inputs, labels, output_dir  = read_data(
        data_path, 
        api,
        model_checkpoint, 
        src_context_size,
        tgt_context_size,
        cfg_name,
        tgt_lang,
        prompt_type,
        num_summary_sentences
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store Hyperparameter in text file
    with open(output_dir+'/config','w', encoding='utf8') as wf:
        for i in [
            f"data_path: {data_path}", 
            f"src_context_size: {src_context_size}",  
            f"tgt_context_size: {tgt_context_size}",
            f"tgt_lang: {tgt_lang}",
            f"api: {api}",
            f"model_checkpoint: {model_checkpoint}", 
            f"batch_size: {batch_size}", 
            f"cfg_name: {cfg_name}",
            f"prompt_type: {prompt_type}",
            f"device: {device}"
            ]:
            wf.write(f"{i}\n")

    # Generate and Evaluate
    evaluate_summarization(
        data_path,
        api,
        model_checkpoint, 
        model, 
        batch_size, 
        device, 
        sources,
        inputs, 
        labels, 
        num_summary_sentences,
        output_dir,
        )

    
    print ("Evaluation Successful")

if __name__ == "__main__":
    main()



"""
#model_checkpoint
#input_sentences


model_checkpoint = "/mnt/data-poseidon/sumire/repos/transformersum/ckpt/epoch=3.ckpt"
model = ExtractiveSummarizer.load_from_checkpoint(model_checkpoint).to(device)


# Load the model onto the specified GPU (CUDA:4)
# device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
#model = model.to(device)

# Make sure the input data is also on the same device
text_to_summarize = "And keep it clean!Mr. Chandler!My friends, you are all familiar with Beaugard.This is perhaps his masterpiece."
input_sentences = ["And keep it clean!", "Mr. Chandler!", "My friends, you are all familiar with Beaugard.", "This is perhaps his masterpiece."]
#input_sentences = [sent.to(devices) for sent in input_sentences]

summary = model.predict_sentences(input_sentences, raw_scores=False, num_summary_sentences=1)

print(model.device)  # Should now correctly reflect "cuda:4"
print(summary)


# How to pass model_name_or_path to ExtractiveSummarizer    
from argparse import ArgumentParser

# Assuming you have a parent parser
parent_parser = ArgumentParser(add_help=False)

# Instantiate your ExtractiveSummarizer and call the add_model_specific_args method
model = ExtractiveSummarizer(hparams={})
# model = ExtractiveSummarizer(hparams.model_name_or_path="allenai/longformer-large-4096") # "allenai/longformer-base-4096"
parser = model.add_model_specific_args(parent_parser)

# Parse the arguments
args = parser.parse_args()

# Now you can access the arguments like this
model_name_or_path = args.model_name_or_path

"""
