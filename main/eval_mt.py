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
from main.preprocess import preprocess_function, preprocess_function_contrapro, generate_few_shots, preprocess_function_bsd, generate_prompt_bsd
from main.metrics import compute_metrics
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for evaluating models.")
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--generic.summarized_contexts", type=str, required=False, default = None,  help="boolean")
    parser.add_argument("--generic.tgt_lang", required=True, help="target language")
    #parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    #parser.add_argument("--cfg", action=ActionConfigFile)
    #parser.add_argument("--generic.src_lang", required=True)
    parser.add_argument("--generic.data_path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.do_train", type=bool, required=False, default =False,  help="if you do evaluate train data")
    parser.add_argument("--generic.src_context", default="src", help="the number of the target context sentence for each input")
    #parser.add_argument("--generic.dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    #parser.add_argument("--generic.speaker", type=bool, default=False)
    #parser.add_argument("--generic.random_context", type=bool, default=False)
    #parser.add_argument("--generic.tag", type=bool, default=False)
    #parser.add_argument("--generic.output_dir", required=True, metavar="DIRECTORY", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.batch_size", type=int, default=0, help="the batch size of evaluation")
    parser.add_argument("--generic.model_checkpoint", required=True, metavar="FILE", help="model_checkpoint")
    parser.add_argument("--generic.k", type=int, default=0, help="the number of few shot")
    parser.add_argument("--generic.prompt_type", type=int, default=1, help="the type of the prompt")
    parser.add_argument("--generic.max_new_tokens", type=int, default=0, help="max_new_tokens")
    parser.add_argument("--generic.max_length", type=int, default=0, help="max_length for input and labels")
    parser.add_argument("--generic.cfg_name", required=True, metavar="FILE", help="config file name")
    parser.add_argument("--generic.api", type=bool, default=False, metavar="FILE", help="Whether using text generation api or not")
    parser.add_argument('--generic.metrics',  type=str, help = "Comma-separated list of strings", default= "sacrebleu,comet", required=False)
    return parser


def initialize_model(model_checkpoint, api):
    if "llama" in model_checkpoint:
        
        tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        #tokenizer.add_special_tokens({"sep_token":"</s>"}) # To replace "/n[]"
        model = LlamaForCausalLM.from_pretrained(model_checkpoint, use_auth_token=True)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    
    elif api is True:
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

                    


    elif "xglm" in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
        configuration = XGLMConfig()
        model = XGLMForCausalLM(configuration).from_pretrained(model_checkpoint) 

    elif "mbart" in model_checkpoint:
        print ("mbart model")
        from transformers import MBartConfig, MBart50Tokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
        #configuration = MBartConfig()
        tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
        tokenizer.src_lang="en_XX"
        model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

    return model, tokenizer 

def read_data(
    data_path,  
    tgt_lang, 
    api,
    model_checkpoint, 
    src_context_size,
    k, 
    prompt_type, 
    max_length, 
    tokenizer,
    summarized_contexts,
    do_train,
    cfg_name
    ):

    if "iwslt_hf" in data_path:
        data_files = { "train": f"{data_path}train_ted_en-{tgt_lang}",  "val": f"{data_path}val_ted_en-{tgt_lang}",  "test": f"{data_path}test_ted_en-{tgt_lang}"}
        dataset = load_dataset("json", data_files=data_files)
        if do_train:
            train_few_shots = generate_few_shots(dataset["train"], src_context_size, tgt_lang, model_checkpoint, k, prompt_type)
            train_sources = np.asarray([sent for doc in dataset["train"]["doc"] for sent in doc["en"]])
            train_labels = np.asarray([sent for doc in dataset["train"]["doc"] for sent in doc[tgt_lang]])
            #val_few_shots = generate_few_shots(dataset["val"], src_context_size, tgt_lang, model_checkpoint, k, prompt_type)
            val_sources = np.asarray([sent for doc in dataset["val"]["doc"] for sent in doc["en"]])
            val_labels = np.asarray([sent for doc in dataset["val"]["doc"] for sent in doc[tgt_lang]])
            
            few_shots = train_few_shots
            sources = np.append(train_sources, val_sources)
            labels = np.append(train_labels, val_labels)
            output_dir = f"/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-{tgt_lang}/{cfg_name}/"
        else:
            few_shots = generate_few_shots(dataset["test"], src_context_size, tgt_lang, model_checkpoint, k, prompt_type)
            sources = np.asarray([sent for doc in dataset["test"]["doc"] for sent in doc["en"]])
            labels = np.asarray([sent for doc in dataset["test"]["doc"] for sent in doc[tgt_lang]])
            output_dir = f"/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/test/en-{tgt_lang}/{cfg_name}/"
        
        print ("The num of sent in train set before preprocess", len([sent for doc in dataset["train"]["doc"] for sent in doc["en"]]))
        print ("The num of sent in test set before preprocess", len([sent for doc in dataset["test"]["doc"] for sent in doc["en"]]))

        if api is True:
            if do_train:
                train_inputs = preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, train_few_shots, prompt_type, max_length, tokenizer, dataset["train"])
                val_inputs = preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, few_shots, prompt_type, max_length, tokenizer, dataset["val"])
                inputs = train_inputs + val_inputs
            else:
                inputs = preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, few_shots, prompt_type, max_length, tokenizer, dataset["test"])
        
        else:
            if do_train:
                train_inputs = preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, train_few_shots, prompt_type, max_length, tokenizer, dataset["train"]).input_ids
                val_inputs = preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, val_few_shots, prompt_type, max_length, tokenizer, dataset["val"]).input_ids
                inputs = train_inputs + val_inputs
            else:
                inputs = preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, few_shots, prompt_type, max_length, tokenizer, dataset["test"]).input_ids
        
    elif "BSD-master" in data_path:
        data_files = {"train":data_path+"train.json","test":data_path+"test.json"}
        dataset = load_dataset("json", data_files=data_files)
        few_shots = generate_prompt_bsd(dataset["train"], src_context_size, tgt_lang, model_checkpoint, k, prompt_type)
        sources = np.asarray([sent['en_sentence'] for doc in dataset["test"]["conversation"] for sent in doc])
        if api is True:
            inputs = preprocess_function_bsd(src_context_size, tgt_lang, api, model_checkpoint, few_shots, prompt_type, max_length, tokenizer, dataset["test"])
        else:
            inputs = preprocess_function_bsd(src_context_size, tgt_lang, api, model_checkpoint, few_shots, prompt_type, max_length, tokenizer, dataset["test"]).input_ids
        #labels = preprocess_function_bsd(tgt_lang, prompt, max_length, tokenizer, dataset["test"]).labels
        labels = np.asarray([sent['ja_sentence'] for doc in dataset["test"]["conversation"] for sent in doc])
        output_dir = f"/mnt/data-poseidon/sumire/thesis/running/BSD/en-{tgt_lang}/{cfg_name}/"

    if "ContraPro" in data_path:
        if api is True:
            inputs, labels, sources, few_shots = preprocess_function_contrapro(data_path, tgt_lang, src_context_size, prompt_type, api, max_length, summarized_contexts)
            output_dir = f"/mnt/data-poseidon/sumire/thesis/running/contrapro/en-{tgt_lang}/{cfg_name}/"
            print (output_dir)
            labels = np.asarray(labels)
            sources = np.asarray(sources)

        else:
            print ("preprocess not defined")

    return few_shots, sources, inputs, labels, output_dir

def evaluate_mt(
    api,
    model_checkpoint, 
    model, 
    tokenizer, 
    batch_size,  
    max_new_tokens, 
    max_length, 
    device, 
    tgt_lang,
    sources,
    inputs, 
    labels, 
    output_dir,
    prompt_type, 
    do_train,
    metrics
    ):

    # Define mBART language code 
    lang_to_code = {"ja": "ja_XX", "ar":"ar_AR", "de":"de_DE", "fr":"fr_XX","ko":"ko_KR", "zh": "zh_CN"}

    if api is True: # No batch, directly input string to the model
        if do_train: # Predict and eval for each instance
            for inp, label, src in zip(inputs, labels, sources): 
                pred = model.generate(inp, max_new_tokens=max_new_tokens).generated_text
                with open(output_dir+'/without_postprocess.txt','a', encoding='utf8') as wf:
                    wf.write(pred.strip()+'\n##########\n') # if with batch maybe need to adapt
                    print ("Hi3")

                with open(output_dir+'/references.txt','a', encoding='utf8') as wf:
                    wf.write(label.strip()+'\n')

                with open(output_dir+'/source.txt','a', encoding='utf8') as wf:
                    wf.write(src.strip()+'\n')

                with open(output_dir+'/prompt+source.txt','a', encoding='utf8') as wf:
                    wf.write(inp.strip()+'\n')

                # Eval
                eval_pred = ([pred], [label], [src])
                result, decoded_pred, decoded_label, decoded_input_id = compute_metrics(metrics, api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_pred, prompt_type)
                print (decoded_pred)

                with open(output_dir+'/translations.txt','a', encoding='utf8') as wf:
                    for i in decoded_pred:
                        wf.write(i.strip()+'\n')

                for metric in metrics:
                    with open(output_dir+f'/{metric}_each_score.txt','a', encoding='utf8') as wf:
                        wf.write(f"{result[metric]}\n") 

            # Open each_sent_score file and calculate the average
            all_scores = {}
            for metric in metrics:
                print (metric)
                sum_scores = 0
                
                with open(output_dir+f'/{metric}_each_score.txt') as rf:
                    for each_score in rf:
                        sum_scores += float(each_score.strip())
                        score = sum_scores / len(sources)
                        all_scores[metric]= score
            print (all_scores)
            
            # Write the averaged score
            with open(output_dir+'/test_score.txt','a', encoding='utf8') as wf:
                for metric in metrics:  
                    wf.write(f"{metric}: {all_scores[metric]}\n") 


        else: # Only Test, soPredict everything first and eval in the end once
            all_preds = []
            all_labels = []
            all_srcs = []
            print ("Hi")

            # Generate
            for inp, label, src in zip(inputs, labels, sources): 
                print ("Hi2")
                
                print ("INP", inp)
                print ("LABEL", label)
                pred = model.generate(inp, max_new_tokens=max_new_tokens).generated_text
                
                all_preds.append(pred)
                all_labels.append(label)
                all_srcs.append(src)
            

                with open(output_dir+'/without_postprocess.txt','a', encoding='utf8') as wf:
                    wf.write(pred.strip()+'\n##########\n') # if with batch maybe need to adapt
                    print ("Hi3")

                with open(output_dir+'/references.txt','a', encoding='utf8') as wf:
                    wf.write(label.strip()+'\n')

                with open(output_dir+'/source.txt','a', encoding='utf8') as wf:
                    wf.write(src.strip()+'\n')

                with open(output_dir+'/prompt+source.txt','a', encoding='utf8') as wf:
                    wf.write(inp.strip()+'\n')
            
            print ("Hi4")


            # Evaluate
            eval_preds = (np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_srcs))
            result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(metrics, api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_preds, prompt_type)
            print (decoded_preds)

            with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
                
                bleu_score = result["bleu"]
                comet_score = result["comet"]
                gen_len_score = result["gen_len"]

                for metric, score in zip(["bleu", "comet", "gen_len"], [bleu_score, comet_score, gen_len_score]):
                    wf.write(f"{metric}: {score}\n") 

            with open(output_dir+'/translations.txt','a', encoding='utf8') as wf:
                for pred in decoded_preds:
                    wf.write(pred.strip()+'\n')
            

    else: 
        model.to(device)    
        model.eval()
        num_batches = 0    
        for batch in tqdm(range(0, len(inputs), batch_size), total = len(inputs)/batch_size, desc="Completed Batches"):
            num_batches += 1
            print ("batch", batch, "to", batch+batch_size)
            batch_ip = inputs[batch:batch+batch_size,:].to(device)
            print ("INPUT", tokenizer.batch_decode(batch_ip, skip_special_tokens=True))
            batch_source = sources[batch:batch+batch_size]
            batch_label = labels[batch:batch+batch_size]
            if "mbart" in model_checkpoint:
                batch_output = model.generate(
                    batch_ip, forced_bos_token_id=tokenizer.lang_code_to_id[lang_to_code[tgt_lang]], max_new_tokens=max_new_tokens, do_sample=False
                    ) # if max_length only doesn't work, need to put max_new_tokens for XGLM model
            else:
                batch_output = model.generate(
                    batch_ip, max_new_tokens=max_new_tokens, do_sample=False
                    ) # if max_length only doesn't work, need to put max_new_tokens for XGLM model
                batch_output = batch_output[:, max_length:]
            print ("generate is done")
            
            # Evaluate
            eval_preds = (batch_output.cpu(), batch_label, batch_source)# To convert to numpy in evaluate function
            result, decoded_preds, decoded_labels, decoded_input_ids = compute_metrics(metrics, api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_preds, prompt_type)
        
            # Write results to text file
            with open(output_dir+'/translations.txt','a', encoding='utf8') as wf:
                for decoded_pred in decoded_preds:
                    wf.write(decoded_pred.strip()+'\n')

            with open(output_dir+'/references.txt','a', encoding='utf8') as wf:
                for decoded_label in (decoded_labels):
                    for item in decoded_label:
                        wf.write(item.strip()+'\n')

            with open(output_dir+'/source.txt','a', encoding='utf8') as wf:
                for decoded_input_id in (decoded_input_ids):
                    wf.write(decoded_input_id.strip()+'\n')

            with open(output_dir+'/prompt+source.txt','a', encoding='utf8') as wf:
                for decoded_prompt in (tokenizer.batch_decode(batch_ip)):
                    wf.write(decoded_prompt.strip()+'\n')
        
            bleu_sum += result["bleu"]
            comet_sum += result["comet"]
            gen_len_sum += result["gen_len"]

            # Store the score
            with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
                bleu = bleu_sum / num_batches
                comet = comet_sum / num_batches
                gen_len = gen_len_sum / num_batches

                wf.write(f"bleu: {bleu}\n") #ensure_ascii=False
                wf.write(f"comet: {comet}\n") 
                wf.write(f"gen_len: {gen_len}\n") 

def main():
    parser = read_arguments()
    cfg = parser.parse_args()
    summarized_contexts = cfg.generic.summarized_contexts
    tgt_lang = cfg.generic.tgt_lang
    data_path = cfg.generic.data_path
    src_context_size = cfg.generic.src_context
    model_checkpoint = cfg.generic.model_checkpoint
    batch_size = cfg.generic.batch_size
    k = cfg.generic.k
    prompt_type =  cfg.generic.prompt_type
    max_new_tokens = cfg.generic.max_new_tokens
    max_length = cfg.generic.max_length
    cfg_name = cfg.generic.cfg_name
    api = cfg.generic.api
    do_train = cfg.generic.do_train
    metrics = cfg.generic.metrics.split(",")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Initialize Model
    model, tokenizer = initialize_model(model_checkpoint, api)

    # Load Dataset
    few_shots, sources, inputs, labels, output_dir  = read_data(
        data_path, 
        tgt_lang, 
        api,
        model_checkpoint, 
        src_context_size,
        k, 
        prompt_type, 
        max_length, 
        tokenizer,
        summarized_contexts,
        do_train,
        cfg_name
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store Hyperparameter in text file
    with open(output_dir+'/config','w', encoding='utf8') as wf:
        for i in [
            f"tgt_lang: {tgt_lang}", 
            f"data_path: {data_path}", 
            f"src_context_size: {src_context_size}",  
            f"api: {api}",
            f"model_checkpoint: {model_checkpoint}", 
            f"batch_size: {batch_size}", 
            f"k: {k}",
            f"prompt_type: {prompt_type}", 
            f"max_new_tokens: {max_new_tokens}", 
            f"max_length: {max_length}", 
            f"cfg_name: {cfg_name}",
            f"prompt+source:\n {few_shots}",
            ]:
            wf.write(f"{i}\n")

    # Generate and Evaluate
    evaluate_mt(
        api,
        model_checkpoint, 
        model, 
        tokenizer, 
        batch_size, 
        max_new_tokens, 
        max_length, 
        device, 
        tgt_lang,
        sources,
        inputs, 
        labels, 
        output_dir,
        prompt_type,
        do_train,
        metrics
        )
    print ("Evaluation Successful")

if __name__ == "__main__":
    main()
