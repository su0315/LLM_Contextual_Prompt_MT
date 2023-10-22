import evaluate # Huggingface evaluatetokenizer
import numpy as np
import preprocess
import json

"""
def postprocess_text(preds, labels, input_ids, model_checkpoint, api, prompt_type):

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    input_ids = [input_id.strip() for input_id in input_ids]
    
    # Llama post process
    if "llama" in model_checkpoint: # only original llama, not using "###User:"
        preds = [pred.split("\n")[0] for pred in preds] # Extract only the first prediction 
    
    if prompt_type ==1 and api:
        tgt_preds = []
        
        for pred in preds:
            if "\n" in pred:
                print ("break token founded")
                tgt_pred = pred.split("\n")[-1]
                print ("split1", tgt_pred)
                tgt_preds.append(tgt_pred)
            else:
                print ("break token not founded")
                tgt_preds.append(pred)
        preds = tgt_preds

    elif prompt_type == 3 and api:
        tgt_preds = []
        break_token = "<#b#>"
        
        for pred in preds:
            if break_token in pred:
                print ("break token founded")
                tgt_pred = pred.split(break_token)[1]
                tgt_preds.append(tgt_pred)
            else:
                if "\n" in pred:
                    print ("back-n founded")
                    tgt_pred = pred.split("\n")[-1]
                    tgt_preds.append(tgt_pred)
                
                else:
                    print ("break token not founded")
                    tgt_preds.append(pred)
        preds = tgt_preds
    return preds, labels, input_ids
"""

def compute_metrics(api, model_checkpoint, output_dir, eval_preds):
    preds, decoded_labels, input_ids = eval_preds
    decoded_input_ids = input_ids
    
    # Preds
    decoded_preds = preds
    decoded_input_ids = input_ids

    if isinstance(preds, tuple):
        preds = preds
        
    metric = evaluate.load('rouge')
    #metric2 =  evaluate.load("comet")

    print ("ROUGE preds", decoded_preds, "ROUGE-references", decoded_labels)
    
    # rouge
    rouge = metric.compute(predictions=decoded_preds, references=decoded_labels,use_stemmer=True )
    result = {"rouge": rouge}

    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #result["gen_len"] = np.mean(prediction_lens)
    result = {k: np.round(v, 4) for k, v in rouge.items()}
    print(result)

    return result, decoded_preds, decoded_labels, decoded_input_ids