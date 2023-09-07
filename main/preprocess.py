import random

def generate_prompt(data, tgt_lang, model_checkpoint, k, prompt_talk_id):

    target_language = {"ja": "Japanese", "de":"German", "fr":"French", "ko": "Korean", "ar": "Arabic", "zh":"Chinese"}

    #K-shot Prompt
    
    if "xglm" in model_checkpoint:
        after_ip = " = "
        sep_token = "</s>"
        _prompt = ""

    elif "llama" or "Llama" in model_checkpoint:
        after_ip = " => "
        sep_token = "\n"
        _prompt = f"""Translate English to {target_language[tgt_lang]}:{sep_token}"""

    random.seed(10)
    # Random prompt doc id
    for i in range(k):
        rd_doc_id = random.randint(0, len(data["talk_id"]))
        print ("rd_doc_id", rd_doc_id)
        talk_id = data["talk_id"][rd_doc_id]
        print ("talk_id", talk_id)
        rd_sent_id =random.randint(0, len(data["doc"][rd_doc_id]))
        print ("rd_sent_id", rd_sent_id)
        en_sent = data["doc"][rd_doc_id]["en"][rd_sent_id]
        tgt_lang_sent =  data["doc"][rd_doc_id][tgt_lang][rd_sent_id]
        k_shot = en_sent + after_ip + tgt_lang_sent + sep_token
        _prompt += k_shot
    
    # 
    # if "xglm" in model_checkpoint:
    #     for doc in data:
    #         if doc["talk_id"] == prompt_talk_id:
    #             for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][13:13+k], doc["doc"][tgt_lang][13:13+k]):
    #                 k_shot = f"{en_sent} = {tgt_lang_sent} </s> "
    #                 _prompt += k_shot
    #             break

    # elif "llama" or "Llama" in model_checkpoint: 
    #     for doc in data:
    #         if doc["talk_id"] == prompt_talk_id:
    #             print ("Prompt found in trainset")
    #             for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][13:13+k], doc["doc"][tgt_lang][13:13+k]):
    #                 k_shot =f"{en_sent} => {tgt_lang_sent}\n"
    #                 _prompt += k_shot
    #             break
    #         else:
    #             print ("No prompt found")
    
    #     _prompt = f"""Translate English to {target_language[tgt_lang]}:\n\n{_prompt}"""
    # 

    return _prompt


def preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, prompt, prompt_talk_id, max_length, tokenizer, data): # data should be splitted into train / dev / test internally

    inputs = []

    if "xglm" in model_checkpoint:
        after_ip = " = "
        sep_token = "</s>"

    elif "llama" or "Llama" in model_checkpoint:
        after_ip = " => "
        sep_token = "\n"
    
    if src_context_size >= 1:
        for doc_idx, doc in enumerate(data["doc"]):
            doc_input = [sent for sent in doc["en"]] # 10 sentences from each document

            for idx, ip in enumerate(doc_input):
                _context = f"Given context:{sep_token}" 

                # Check each context index given the context size and current input index
                for context_window in range(src_context_size, 0, -1):
                    context_idx = idx - context_window
                    # If context idx is not the left side of the beggining of the doc_inputs
                    if context_idx >= 0: 
                        _context += doc_input[context_idx] + sep_token 

                concat_input = _context + sep_token + prompt + ip + after_ip 
                inputs.append(concat_input)

    else:
        if "mbart" in model_checkpoint:
            inputs = [sent for doc in data["doc"] for sent in doc["en"]]
            tokenizer.src_lang = "en_XX"

        else:
            #inputs = [f"Given context:{sep_token}" + prompt + sent + after_ip for doc in data["doc"] for sent in doc["en"]]  
            inputs = [prompt + sent + after_ip for doc in data["doc"] for sent in doc["en"]] # When6 without context prompt 
    
    if api is True:
        model_inputs = inputs

    else:
        model_inputs = tokenizer(
            inputs, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True) #, max_length=500, truncation=True, padding='max_length', return_tensors="pt"
    
    print ("INPUT EXAMPLE 0")
    print (inputs[0])
    print ("INPUT EXAMPLE 1")
    print (inputs[1])
    
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


def preprocess_function_bsd(tgt_lang, api, prompt, max_length, tokenizer, data): # data should be splitted into train / dev / test internally
    inputs =  [prompt + sent['en_sentence'] + ' = ' for doc in data["conversation"] for sent in doc] 
    
    if api is True:
        model_inputs = inputs
    else:
        model_inputs = tokenizer(
            inputs, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
        
    return model_inputs