

from extractive import ExtractiveSummarizer
import torch
import os 
from transformers import AutoConfig, AutoModel, AutoTokenizer
from preprocess_summ import preprocess_function_contrapro
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
    #parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    #parser.add_argument("--cfg", action=ActionConfigFile)
    #parser.add_argument("--generic.src_lang", required=True)
    parser.add_argument("--generic.data_path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.src_context", default="src", help="the number of the target context sentence for each input")
    #parser.add_argument("--generic.dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    #parser.add_argument("--generic.speaker", type=bool, default=False)
    #parser.add_argument("--generic.random_context", type=bool, default=False)
    #parser.add_argument("--generic.tag", type=bool, default=False)
    #parser.add_argument("--generic.output_dir", required=True, metavar="DIRECTORY", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.batch_size", type=int, default=0, help="the batch size of evaluation")
    parser.add_argument("--generic.model_checkpoint", required=True, metavar="FILE", help="model_checkpoint")
    #parser.add_argument("--generic.k", type=int, default=0, help="the number of few shot")
    parser.add_argument("--generic.cfg_name", required=True, metavar="FILE", help="config file name")
    parser.add_argument("--generic.api", type=bool, default=False, metavar="FILE", help="Whether using text generation api or not")
    parser.add_argument("--generic.device",type=int, default=6,  help="The GPU device id")
    
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
        #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)   # ,  truncation=True, return_tensors="pt") # padding_side = 'left',
        
        model = ExtractiveSummarizer.load_from_checkpoint(model_checkpoint).to(device)
        print (model.hparams)
    return model

def read_data(
    data_path,   
    api,
    model_checkpoint, 
    src_context_size,
    cfg_name
    ):

    if "ContraPro" in data_path:
        inputs, labels, sources = preprocess_function_contrapro(data_path, model_checkpoint, src_context_size, api)
        output_dir = f"./results/summarization/contrapro/{cfg_name}/"
        labels = np.asarray(labels)
        sources = np.asarray(sources)

    else:
        print ("preprocess not defined")

    return sources, inputs, labels, output_dir

def evaluate_summarization(
    api,
    model_checkpoint, 
    model, 
    batch_size,  
    device, 
    sources,
    inputs, 
    labels, 
    output_dir,

    ):

    all_preds = []
    all_labels = []
    all_srcs = []
    print ("Hi")

    # Generate
    
    for inp, label, src in zip(inputs, labels, sources): 
        pred = model.predict(inp, raw_scores=False, num_summary_sentences=1)
        
        all_preds.append(pred)
        all_labels.append(label)
        all_srcs.append(src)
        
        if "llama" in model_checkpoint or "Llama" in model_checkpoint:
            with open(output_dir+'/without_postprocess.txt','a', encoding='utf8') as wf:
                wf.write(pred.strip()+'\n##########\n') # if with batch maybe need to adapt
                print ("Hi3")

            with open(output_dir+'/prompt+source.txt','a', encoding='utf8') as wf:
                wf.write(inp.strip()+'\n')


        with open(output_dir+'/references.txt','a', encoding='utf8') as wf:
            wf.write(label.strip()+'\n')

        with open(output_dir+'/source.txt','a', encoding='utf8') as wf:
            wf.write(src.strip()+'\n')

    
    # Evaluate    
    eval_preds = (np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_srcs))
    result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(api, model_checkpoint, output_dir, eval_preds)
    print (decoded_preds)
    
    with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
        rouge_score = result

        for metric, score in zip(["rouge"], [rouge_score]):
            wf.write(f"{metric}: {score}\n") 

    with open(output_dir+'/summarized_contexs.txt','a', encoding='utf8') as wf:
        for pred in decoded_preds:
            wf.write(pred.strip()+'\n')
    

def main():
    parser = read_arguments()
    cfg = parser.parse_args()

    data_path = cfg.generic.data_path
    src_context_size = cfg.generic.src_context
    model_checkpoint = cfg.generic.model_checkpoint
    batch_size = cfg.generic.batch_size
    cfg_name = cfg.generic.cfg_name
    api = cfg.generic.api
    device = torch.device(f'cuda:{cfg.generic.device}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model
    model = initialize_model(model_checkpoint, api, device)
    # Load Dataset
    sources, inputs, labels, output_dir  = read_data(
        data_path, 
        api,
        model_checkpoint, 
        src_context_size,
        cfg_name
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store Hyperparameter in text file
    with open(output_dir+'/config','w', encoding='utf8') as wf:
        for i in [
            f"data_path: {data_path}", 
            f"src_context_size: {src_context_size}",  
            f"api: {api}",
            f"model_checkpoint: {model_checkpoint}", 
            f"batch_size: {batch_size}", 
            f"cfg_name: {cfg_name}",
            f"device: {device}"
            ]:
            wf.write(f"{i}\n")

    # Generate and Evaluate
    evaluate_summarization(
        api,
        model_checkpoint, 
        model, 
        batch_size, 
        device, 
        sources,
        inputs, 
        labels, 
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


"""
