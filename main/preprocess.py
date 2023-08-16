
def generate_context(src_context_size, prompt_talk_id, data):
    
    # Identify prompt talk and eliminate it from the inputs
    prompt_talk_index = data["talk_id"].index(prompt_talk_id)

    for doc_idx, doc in enumerate(data["doc"]):
        doc_input = [sent for doc in data["doc"][prompt_talk_index+1:3] for sent in doc["en"]] # 10 sentences from each document
        for idx, ip in enumerate(doc_input):
            _context = "Given context:\n\n"

            # Check each context index given the context size and current input index
            for context_window in range(src_context_size, 0, -1):
                context_idx = idx - context_window
                # If context idx is not the left side of the beggining of the doc_inputs
                if context_idx >= 0: 
                    _context += doc_input[context_idx] + "\n"
        
            print ("CONTEXT", idx, _context, ip)
    return _context




def generate_prompt(data, tgt_lang, model_checkpoint, k, prompt_talk_id):

    target_language = {"ja": "Japanese", "de":"German", "fr":"French", "ko": "Korean", "ar": "Arabic", "zh":"Chinese"}

    #K-shot Prompt
    _prompt = ""

    
    if "xglm" in model_checkpoint:
        for doc in data:
            if doc["talk_id"] == prompt_talk_id:
                for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][13:13+k], doc["doc"][tgt_lang][13:13+k]):
                    k_shot = f"{en_sent} = {tgt_lang_sent} </s> "
                    _prompt += k_shot
                break

    elif "llama" in model_checkpoint:
        for doc in data:
            if doc["talk_id"] == prompt_talk_id:
                for en_sent, tgt_lang_sent in zip(doc["doc"]["en"][13:13+k], doc["doc"][tgt_lang][13:13+k]):
                    k_shot =f"{en_sent} => {tgt_lang_sent}\n"
                    _prompt += k_shot
                break
        _prompt = f"""Translate English to {target_language[tgt_lang]}:\n\n{_prompt}"""
            
    print("PROMPT", _prompt)
    return _prompt


def preprocess_function(src_context_size, tgt_lang, model_checkpoint, prompt, prompt_talk_id, max_length, tokenizer, data): # data should be splitted into train / dev / test internally
    
    # Identify prompt talk and eliminate it from the inputs
    #prompt_talk_index = data["talk_id"].index(prompt_talk_id)
    inputs = []
    

    if "xglm" in model_checkpoint:
        after_ip = " = "

    elif "llama" in model_checkpoint:
        after_ip = " => "

    print ("Yes")
    if src_context_size >= 1:
        print ("Yes2")
        for doc_idx, doc in enumerate(data["doc"]):
            doc_input = [sent for sent in doc["en"]] # 10 sentences from each document

            for idx, ip in enumerate(doc_input):
                _context = "Given context:\n\n"

                # Check each context index given the context size and current input index
                for context_window in range(src_context_size, 0, -1):
                    context_idx = idx - context_window
                    # If context idx is not the left side of the beggining of the doc_inputs
                    if context_idx >= 0: 
                        _context += doc_input[context_idx] + "\n"

                concat_input = _context + "\n" + prompt + ip + after_ip
                inputs.append(concat_input)

    else:
        inputs = [prompt + ip + after_ip for doc in data["doc"][:] for sent in doc["en"]] 

        #if "xglm" in model_checkpoint:
            #doc_input = [_context + prompt + ip + ' = ' for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]] ## [1:] to eliminate Few shot example

        #elif "llama" in model_checkpoint:
            #doc_input = [_context + prompt + ip + " => " for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]]

    #print ("INPUTS", inputs)
    
    #if "xglm" in model_checkpoint:
        #inputs = [prompt + sent + ' = ' for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]] ## [1:] to eliminate Few shot example

    #elif "llama" in model_checkpoint:
        #inputs = [f"""Translate English to {target_language[tgt_lang]}: \n\n {prompt} {sent} =>""" for doc in data["doc"][prompt_talk_index+1:] for sent in doc["en"]] ## [1:] to eliminate Few shot example
    

    targets = [sent for doc in data["doc"] for sent in doc[tgt_lang]]
    
    print ("INPUT EXAMPLE 0")
    print (inputs[0])
    print ("INPUT EXAMPLE 1")
    print (inputs[1])
    model_inputs = tokenizer(
        inputs, text_target=targets, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True) #, max_length=500, truncation=True, padding='max_length', return_tensors="pt"
    
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
    inputs =  [prompt + sent['en_sentence'] + ' = ' for doc in data["conversation"][1:5] for sent in doc] 
    targets = [sent['ja_sentence'] for doc in data["conversation"][1:5] for sent in doc] 
    model_inputs = tokenizer(
         inputs, text_target=targets, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
    
    return model_inputs