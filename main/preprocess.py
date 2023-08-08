
def generate_prompt(data, tgt_lang, model_checkpoint, k, prompt_talk_id):
    #K-shot Prompt
    _prompt = ""

    for doc in data:
        if doc["talk_id"] == prompt_talk_id:
            for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][13:13+k], doc["doc"][tgt_lang][13:13+k]):
                if "xglm" in model_checkpoint:
                    k_shot = f"{en_sent} = {tgt_lang_sent} </s> "

                if "llama" in model_checkpoint:
                    k_shot =f"{en_sent} => {tgt_lang_sent} \n "
                _prompt += k_shot
    return _prompt

def preprocess_function(tgt_lang, model_checkpoint, prompt, prompt_talk_id, max_length, tokenizer, data): # data should be splitted into train / dev / test internally
    

    target_language = {"ja": "Japanese", "de":"German", "fr":"French", "ko": "Korean", "ar": "Arabic", "zh":"Chinese"}
    
    # Identify prompt talk
    prompt_talk_index = data["talk_id"].index(prompt_talk_id)
    
    if "xglm" in model_checkpoint:
        inputs = [prompt + sent + ' = ' for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]] ## [1:] to eliminate Few shot example

    elif "llama" in model_checkpoint:
        inputs = [f"""Translate English to {target_language[tgt_lang]}: \n\n {prompt} {sent} =>""" for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]] ## [1:] to eliminate Few shot example
    
    targets = [sent for doc in data["doc"][prompt_talk_index+1:]for sent in doc[tgt_lang]]

    model_inputs = tokenizer(
        inputs, text_target=targets, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True) #, max_length=500, truncation=True, padding='max_length', return_tensors="pt"
    
    
    #print ("model_inputs_keys", model_inputs.keys())
    #print ('model_inputs["input_ids"]', tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True))
    #print ('model_inputs["labels"]', tokenizer.batch_decode(model_inputs["labels"], skip_special_tokens=True))
    
    #model_inputs = tokenizer(inputs,  max_length=250, truncation=True, padding='max_length')
    #print ("input_ids", model_inputs["input_ids"])
    #print ("labels", model_inputs["labels"])
    
    return model_inputs

def generate_prompt_bsd(data, tgt_lang, k):
    #K-shot Prompt

    _prompt = ""
    
    for sent in data["conversation"][0][:k]:
        en_sent = sent["en_sentence"]
        tgt_lang_sent = sent[f"{tgt_lang}_sentence"]
        k_shot = f"{en_sent} = {tgt_lang_sent} </s> "
        _prompt += k_shot
       
    return _prompt


def preprocess_function_bsd(tgt_lang, prompt, max_length, tokenizer, data): # data should be splitted into train / dev / test internally
    """
    # Identify prompt talk
    prompt_talk_index = data["talk_id"].index(prompt_talk_id)

    inputs = [prompt + sent + ' = ' for doc in data["doc"][prompt_talk_index+1:5] for sent in doc["en"]] ## [1:] to eliminate Few shot example
    targets = [sent for doc in data["doc"][prompt_talk_index+1:5]for sent in doc[tgt_lang]]

    model_inputs = tokenizer(
        inputs, text_target=targets, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True) #, max_length=500, truncation=True, padding='max_length', return_tensors="pt"
    #print ("model_inputs_keys", model_inputs.keys())
    #print ('model_inputs["input_ids"]', tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True))
    #print ('model_inputs["labels"]', tokenizer.batch_decode(model_inputs["labels"], skip_special_tokens=True))
    
    #model_inputs = tokenizer(inputs,  max_length=250, truncation=True, padding='max_length')
    #print ("input_ids", model_inputs["input_ids"])
    #print ("labels", model_inputs["labels"])
    """
    inputs =  [prompt + sent['en_sentence'] + ' = ' for doc in data["conversation"][1:5] for sent in doc] 
    targets = [sent['ja_sentence'] for doc in data["conversation"][1:5] for sent in doc] 
    model_inputs = tokenizer(
         inputs, text_target=targets, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
    
    return model_inputs