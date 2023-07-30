
def generate_prompt(data, tgt_lang, k, prompt_talk_id):
    #K-shot Prompt

    _prompt = ""

    for doc in data["test"]:
        if doc["talk_id"] == prompt_talk_id:
            for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][13:13+k], doc["doc"][tgt_lang][13:13+k]):
                k_shot = f"{en_sent} = {tgt_lang_sent} </s> "
                _prompt += k_shot
    return _prompt

def preprocess_function(tgt_lang, prompt, prompt_talk_id, max_length, tokenizer, data): # data should be splitted into train / dev / test internally
    # Identify prompt talk
    prompt_talk_index = data["talk_id"].index(prompt_talk_id)

    print ("popped_docs", data["talk_id"])

    inputs = [prompt + sent + ' = ' for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]] ## [1:] to eliminate Few shot example
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
