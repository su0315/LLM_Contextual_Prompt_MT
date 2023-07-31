import evaluate # Huggingface evaluatetokenizer
import numpy as np
import preprocess
import json



def postprocess_text(preds, labels, input_ids):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    input_ids = [[input_id.strip()] for input_id in input_ids]

    return preds, labels, input_ids




def compute_metrics(dataset, output_dir, tgt_lang, tokenizer, eval_preds):
    preds, labels, input_ids = eval_preds
    
    #labels = tokenized_datasets["test"]["labels"]  # Why We have to define it again ?? 
    #print ("preds before split:", tokenizer.batch_decode(preds[:5], skip_special_tokens=True))
    #print ("labels:", tokenizer.batch_decode(labels[:5], skip_special_tokens=True))
    #print ()
    #print ("input before split:",tokenizer.batch_decode(input_ids[:5], skip_special_tokens=True))
    #print ()
    
    sep = tokenizer.sep_token_id
    split_id = tokenizer.encode("=")[-1]
    
    # Preds
    if isinstance(preds, tuple):
        preds = preds
    #preds = [ np.array_split(item, np.where(item == sep)[-1])[-1]  for item in preds ]
    preds = [ np.array_split(item, np.where(item == split_id)[-1])[-1]  for item in preds ]
    del_index= [0, 1] # Delete "= " in the beggining of the prediction 
    
    preds =[ np.delete(item, del_index) for item in preds ]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #print ("splited preds: ", decoded_preds[:5])
    print ()
    
    
    # Labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #print ("labels from eval_preds", [tokenizer.decode(i, skip_special_tokens=True) for i in labels])
    #print ()
    #labels= [ np.array_split(item, np.where(item == sep)[-1])[-1]  for item in labels ]
    #print ("checking labels_token:")
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print ("splitted labels: ", decoded_labels[:5])
    
    # Input_ids
    input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
    print ("input with prompts", tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    input_ids = [ np.array_split(item, np.where(item == sep)[-1])[-1]  for item in input_ids ]
    input_ids = [ np.array_split(item, np.where(item == split_id)[-1])[0]  for item in input_ids ] 
    decoded_input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #print ("splited input_ids", decoded_input_ids[:5])
    print ()
    

    decoded_preds, decoded_labels, decoded_input_ids = postprocess_text(decoded_preds, decoded_labels, decoded_input_ids)
    
    metric1 = evaluate.load("sacrebleu")
    metric2 =  evaluate.load("comet")
    #metric1.add_batch(predictions=decoded_preds, references = decoded_labels)
    #metric2.add_batch(predictions=decoded_preds, references=[item for decoded_label in decoded_labels for item in decoded_label], sources = [item for decoded_input_id in decoded_input_ids for item in decoded_input_id])

    # bleu
    if tgt_lang == "ja":
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels, tokenize='ja-mecab')
    else: 
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": bleu["score"]}

    # comet
    print ("COMET", "decoded_input_ids:",  [item for decoded_input_id in decoded_input_ids for item in decoded_input_id][:5], "\ndecoded_preds", decoded_preds[:5], "\ndecoded_labels", [item for decoded_label in decoded_labels for item in decoded_label][:5])
    
    comet = metric2.compute(predictions=decoded_preds, references=[item for decoded_label in decoded_labels for item in decoded_label], sources = [item for decoded_input_id in decoded_input_ids for item in decoded_input_id])
    result["comet"] =  np.mean(comet["scores"])
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)

    return result, decoded_preds, decoded_labels, decoded_input_ids
