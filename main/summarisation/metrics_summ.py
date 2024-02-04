import evaluate # Huggingface evaluatetokenizer
import numpy as np
import json


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