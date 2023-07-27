
def generate_prompt(data, tgt_lang, k, prompt_talk_id):
    #K-shot Prompt
    prompt_talk_id =  1548
    k = 3
    _prompt = ""

    for doc in data["test"]:
        if doc["talk_id"] == prompt_talk_id:
            for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][:k], doc["doc"][tgt_lang][:k]):
                k_shot = f"{en_sent} = {tgt_lang_sent} </s> "
                _prompt += k_shot
    return _prompt

def preprocess_function(tgt_lang, prompt, prompt_talk_id, tokenizer, data): # data should be splitted into train / dev / test internally
    
    """
    print ("what is doc", data["talk_id"])
    for talk_id, doc in zip(data["talk_id"],data["doc"]):
        print ("what is talkid", talk_id, "what is doc", doc)
        if talk_id == int(prompt_talk_id):
            data.pop(doc)
            print ( "popped")
            break 
    """

    inputs = [prompt + sent + ' = ' for doc in data["doc"] for sent in doc["en"]][:50]
    #inputs = ["Translate English to Japanese: " + sent for doc in data["doc"] for sent in doc["en"]][:50]
    print (inputs[0])
    
    targets = [sent for doc in data["doc"] for sent in doc[tgt_lang] ][:50]
    print (targets[0])
    
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=500, truncation=True, padding='max_length', 
    )
    print (model_inputs["labels"])
    
    #model_inputs = tokenizer(inputs,  max_length=250, truncation=True, padding='max_length')
    #print ("input_ids", model_inputs["input_ids"])
    #print ("labels", model_inputs["labels"])
    
    return model_inputs
