import evaluate # Huggingface evaluatetokenizer
import numpy as np
import preprocess
import json

def postprocess_text(preds, labels, input_ids, model_checkpoint):

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    input_ids = [input_id.strip() for input_id in input_ids]
    
    # Llama post process
    if "llama" or "Llama" in model_checkpoint:
        preds = [pred.split("\n")[0] for pred in preds] # Extract only the first prediction 
        #input_ids = [input_id.split("\n")[-1][:-2] for input_id in input_ids] # Extract the input from examples + input [:-2] works for removing "=>" ?
        
    #if "xglm" in model_checkpoint:
        #input_ids = [input_id[:-1] for input_id in input_ids] # Extract the input from examples + input [:-1] works for removing "=" ?

    return preds, labels, input_ids


def compute_metrics(dataset, api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_preds):
    preds, decoded_labels, input_ids = eval_preds
    decoded_input_ids = input_ids
    
    # Preds
    if api is True:
        decoded_preds = preds
        decoded_input_ids = input_ids
        
    else:
        sep = tokenizer.sep_token_id
        split_id = tokenizer.encode("=")[-1]

        if isinstance(preds, tuple):
            preds = preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Input_ids
        #input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
        
        #if "xglm" in model_checkpoint:
            #input_ids = [ np.array_split(item, np.where(item == sep)[-1])[-1] for item in input_ids ] 
        
        #decoded_input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
   
    decoded_preds, decoded_labels, decoded_input_ids = postprocess_text(decoded_preds, decoded_labels, decoded_input_ids,  model_checkpoint)
    
    metric1 = evaluate.load("sacrebleu")
    metric2 =  evaluate.load("comet")

    print ("BLEU-preds", decoded_preds, "BLEU-references", decoded_labels)
    # bleu
    if tgt_lang == "ja":
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels, tokenize='ja-mecab')
    elif tgt_lang == "ko":
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels, tokenize='ko-mecab')
    elif tgt_lang == "zh":
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels, tokenize='zh')
    elif tgt_lang == "ar":
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels, tokenize='flores101')
    else: 
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": bleu["score"]}

    # comet    
    comet = metric2.compute(predictions=decoded_preds, references=[item for decoded_label in decoded_labels for item in decoded_label], sources = decoded_input_ids)
    result["comet"] =  np.mean(comet["scores"])
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)

    return result, decoded_preds, decoded_labels, decoded_input_ids