import evaluate # Huggingface evaluatetokenizer
import numpy as np
import preprocess
import json

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
            print ("pred", pred)
            if "\n" in pred:
                print ("break token founded")
                tgt_pred = pred.split("\n")[-1]
                print ("split1", tgt_pred)
                tgt_preds.append(tgt_pred)
            else:
                print ("break token not founded")
                tgt_preds.append(pred)
        preds = tgt_preds
        print (tgt_preds)

    elif prompt_type == 3 and api:
        tgt_preds = []
        break_token = "<#b#>"
        
        for pred in preds:
            print ("pred", pred)
            if break_token in pred:
                print ("break token founded")
                tgt_pred = pred.split(break_token)[1]
                tgt_preds.append(tgt_pred)
                print (tgt_preds)
            else:
                if "\n" in pred:
                    print ("back-n founded")
                    tgt_pred = pred.split("\n")[-1]
                    print ("split1", tgt_pred)
                    tgt_preds.append(tgt_pred)
                
                else:
                    print ("break token not founded")
                    tgt_preds.append(pred)
        preds = tgt_preds
        print (tgt_preds)
    return preds, labels, input_ids


def compute_metrics(api, model_checkpoint, output_dir, tgt_lang, tokenizer, eval_preds, prompt_type):
    preds, decoded_labels, input_ids = eval_preds
    decoded_input_ids = input_ids
    
    # Preds
    if api:
        decoded_preds = preds
        decoded_input_ids = input_ids
        print ("preds before postprocess", decoded_preds)

        
    else:
        sep = tokenizer.sep_token_id
        split_id = tokenizer.encode("=")[-1]

        if isinstance(preds, tuple):
            preds = preds

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print ("preds before postprocess", decoded_preds)
    
    decoded_preds, decoded_labels, decoded_input_ids = postprocess_text(decoded_preds, decoded_labels, decoded_input_ids,  model_checkpoint, api, prompt_type)
    
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