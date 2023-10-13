import random
import json

def generate_few_shots(data, src_context_size, tgt_lang, model_checkpoint, k, prompt_type):

    target_language = {"ja": "Japanese", "de":"German", "fr":"French", "ko": "Korean", "ar": "Arabic", "zh":"Chinese"}

    break_token = "<#b#>"

    #K-shot Prompt
    if "xglm" in model_checkpoint:
        after_ip = " = "
        sep_token = "</s>"
        _few_shots = ""

    elif "llama" or "Llama" in model_checkpoint:
        after_ip = " => "
        sep_token = "\n"
        _few_shots = f"""Translate English to {target_language[tgt_lang]}:{sep_token}"""

    # Determine few-shot example 
    seed = 10 #12 #11 # 10
    random.seed(seed)
    #np.random.seed(10)
    # Random prompt doc id
    #for i in range(k):
    for i in range(k):
        rd_doc_id = random.randint(0, len(data["talk_id"])-1)
        print (len(data["talk_id"]))
        print ("rd_doc_id", rd_doc_id)
        print (data["talk_id"])
        talk_id = data["talk_id"][rd_doc_id]
        print ("talk_id", talk_id)
        rd_sent_id =random.randint(0, len(data["doc"][rd_doc_id]))
        print ("rd_sent_id", rd_sent_id)
        if prompt_type == 1 or src_context_size == 0:
            en_sent = data["doc"][rd_doc_id]["en"][rd_sent_id]
            tgt_lang_sent =  data["doc"][rd_doc_id][tgt_lang][rd_sent_id]
            
        if prompt_type == 2:
            en_shot_context = select_context(k, data["doc"][rd_doc_id]["en"], rd_sent_id, sep_token, prompt_type) # context_size = K
            en_sent = en_shot_context + break_token + data["doc"][rd_doc_id]["en"][rd_sent_id] 
            tgt_lang_sent = data["doc"][rd_doc_id][tgt_lang][rd_sent_id]

        if prompt_type == 3:
            en_shot_context = select_context(k, data["doc"][rd_doc_id]["en"], rd_sent_id, sep_token, prompt_type) # context_size = K
            tgt_shot_context = select_context(k, data["doc"][rd_doc_id][tgt_lang], rd_sent_id, sep_token, prompt_type)
            en_sent = en_shot_context + break_token + data["doc"][rd_doc_id]["en"][rd_sent_id] 
            tgt_lang_sent = tgt_shot_context + break_token + data["doc"][rd_doc_id][tgt_lang][rd_sent_id]
        
        k_shot = en_sent + after_ip + tgt_lang_sent + sep_token
        _few_shots += k_shot
    
    return _few_shots

def select_context(context_size, doc_input, current_idx, sep_token, prompt_type):
    #context_list = []
    _context = ""
    
    # Check each context index given the context size and current input index
    for context_window in range(context_size, 0, -1):
        context_idx = current_idx - context_window
        # If context idx is not the left side of the beggining of the doc_inputs
        
        if context_idx >= 0:     
            #context_list.append(doc_input[context_idx]) ## call summarized context 
            _context += doc_input[context_idx]
            if prompt_type == 1:
                _context += sep_token
    
    return _context 


def preprocess_function(src_context_size, tgt_lang, api, model_checkpoint, few_shots, prompt_type, max_length, tokenizer, data): # data should be splitted into train / dev / test internally

    break_token = "<#b#>"
 
    if "xglm" in model_checkpoint:
        after_ip = " = "
        sep_token = "</s>"

    elif "llama" or "Llama" in model_checkpoint:
        after_ip = " => "
        sep_token = "\n"

    inputs = []

    # Context for source sentence
    if prompt_type == 1:
        context_inst = f"Given context:{sep_token}" 
    elif prompt_type == 2:
        context_inst = ""

    if src_context_size >= 1:
        for doc_idx, doc in enumerate(data["doc"]):
            doc_input = [sent for sent in doc["en"]] 
            
            for idx, ip in enumerate(doc_input):
                _context = select_context(src_context_size, doc_input, idx, sep_token, prompt_type)

                if prompt_type==1:
                    #concat_input = _context + prompt + ip + after_ip 
                    #inputs.append(concat_input)
                    if api is True:
                        print ("api!!!!!!!!!!!!!!!!!!!!1")
                        concat_input = "### User:\n" + context_inst + _context + few_shots + ip + "\n\n### Assistant:\n"
                    else:
                        concat_input = context_inst + _context + few_shots + ip + after_ip # need to check when context size == 0, there are no two times sep_token 

                    
                elif prompt_type ==2:
                    concat_input = few_shots + _context + break_token + ip + after_ip ### Put the special break token before input

                elif prompt_type == 3:
                    if _context != "":
                        _context += break_token  # break token only when context exists
                    
                    if api is True:
                        concat_input = "### User:\n" + few_shots + _context + ip + "\n\n### Assistant:\n"

                    else:
                        concat_input = few_shots + _context + ip + after_ip # break token before ip or not ?

                inputs.append(concat_input)

    else:
        if "mbart" in model_checkpoint:
            inputs = [sent for doc in data["doc"] for sent in doc["en"]]
            tokenizer.src_lang = "en_XX"

        else:
            #inputs = [f"Given context:{sep_token}" + prompt + sent + after_ip for doc in data["doc"] for sent in doc["en"]]  

            if api:
                inputs = ["### User:\n" + few_shots + sent + "\n\n### Assistant:\n" for doc in data["doc"] for sent in doc["en"]] 
            else:
                inputs = [few_shots + sent + after_ip for doc in data["doc"] for sent in doc["en"]] # When6 without context prompt 
            
        
    
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

def preprocess_function_contrapro(data_path, tgt_lang, src_context_size, prompt_type, api, max_length):

    target_language = {"ja": "Japanese", "de":"German", "fr":"French", "ko": "Korean", "ar": "Arabic", "zh":"Chinese"}
    with open(f'{data_path}/contrapro.text.en' , 'r') as file:
        src_list = []
        line_counter = 0
        for line in file:
            if line_counter % 3 == 0:
                src_list.append(line.strip())
            line_counter += 1
    print ("src", len(src_list))
    
    with open(f'{data_path}/contrapro.text.{tgt_lang}' , 'r') as file:
        tgt_list = []
        line_counter = 0
        for line in file:
            if line_counter % 3 == 0:
                tgt_list.append(line.strip())
            line_counter += 1
    
    with open('/mnt/data-poseidon/sumire/repos/ContraPro/contrapro.json', "r") as file:
        json_data = json.load(file)

    # Extract and store the "src_segment" values in a list
    print ("json file length", len(json_data))
    tgt_list_json = [item['ref segment'] for item in json_data]
    src_list_json = [item['src segment'] for item in json_data]
    print ("tgt", len(tgt_list_json))

    # take intercept of src examples
    intersection = list(set(src_list).intersection(src_list_json))
    indices_in_src_list = sorted([src_list.index(element) for element in intersection])
    indices_in_json_list = sorted([src_list_json.index(element) for element in intersection])
    #print ("################", len(indices_in_src_list))
    #print (("################", len(indices_in_json_list)))
    #print ("intersection_id_in_src", indices_in_src_list[:100])
    #print ("intersection_id_in_json", indices_in_json_list[:100])

    
    with open(f'{data_path}/contrapro.context.en' , 'r') as file:
        sep_token = "\n"
        context_list = []
        line_counter = 0

        if src_context_size > 0:        
            for line in file:
                if line_counter % (3*src_context_size) == 0:
                    context = ''
        
                if line_counter % (3*src_context_size) <= src_context_size - 1:
                    context += line.strip()
                    if line != '':
                        context += sep_token
                
                if line_counter % (3*src_context_size) == src_context_size - 1:
                    context_list.append(context)
                
                line_counter += 1

    print ("context", len(context_list))
    
    
    inputs = []
    labels = []


    # Context for source sentence
    if prompt_type == 1 and src_context_size >= 1 :
        context_inst = f"Given context:{sep_token}" 
    
    else: 
        context_inst = ""
    
    src_list = [src_list_json[i] for i in indices_in_json_list]
    tgt_list = [tgt_list_json[i] for i in indices_in_json_list]

    if src_context_size > 0:
        context_list_json = [context_list[i] for i in indices_in_src_list]
    
    #print ("src", len(src_list), "tgt", len(tgt_list), "context", len(context_list_json))

    if src_context_size > 0:
        for ip, tgt, _context in zip(src_list, tgt_list, context_list_json):
            concat_input = "### User:\n" + context_inst + _context + f"Translate English to {target_language[tgt_lang]}:{sep_token}" + ip + "\n\n### Assistant:\n"            
            inputs.append(concat_input)
            labels.append(tgt)

    else:
        for ip, tgt in zip(src_list, tgt_list):
            concat_input = "### User:\n" + f"Translate English to {target_language[tgt_lang]}:{sep_token}" + ip + "\n\n### Assistant:\n"            
            inputs.append(concat_input)
            labels.append(tgt)
    sources = src_list
    few_shots = ""
    return inputs, labels, sources, few_shots

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

"""
data_path = "/mnt/data-poseidon/sumire/repos/ContraPro/context_1"
tgt_lang = "de"
api = True
max_length = 1024
src_context_size = 1
prompt_type = 1
inputs, labels = preprocess_function_contrapro(data_path, tgt_lang, src_context_size, prompt_type, api, max_length)
print (labels[-5:], inputs[-5:])
"""