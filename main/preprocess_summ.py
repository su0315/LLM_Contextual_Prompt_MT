import random
import json


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

def sample_example(criteria, original_data, original_indices, n_samples):
    if criteria == "ante==1":
        indices_on_criteria = [i for i, ante in enumerate(original_data) if ante == 1]
    elif criteria == "ante==2":
        indices_on_criteria = [i for i, ante in enumerate(original_data) if ante == 2]
    
    seed = 10 
    random.seed(10)
    sample_size = min(n_samples, len(indices_on_criteria))
    random_samples = random.sample(indices_on_criteria, sample_size)
    dropped_indices = [i for i in indices_on_criteria if i not in random_samples]
    #original_indices = [i for i, item in enumerate(original_data)]
    sampled_indices = [i for i in original_indices if i not in dropped_indices]

    return sampled_indices

def preprocess_function_contrapro(data_path, model_checkpoint, src_context_size, api):

    sep_token = "\n"

    # Take txt file and choose one example out of duplications and store in src_list # 12011
    with open(f'{data_path}/contrapro.text.en' , 'r') as file:
        src_list = []
        line_counter = 0
        for line in file:
            if line_counter % 3 == 0:
                src_list.append(line.strip())
            line_counter += 1
    print ("src", len(src_list)) # 12011
    
    n_sent = len(src_list)

    # Extract and store the "src_segment" values and "ref segment" and "ante distance" from json file to a list
    with open('/mnt/data-poseidon/sumire/repos/ContraPro/contrapro.json', "r") as file:
        json_data = json.load(file)

    print ("json file length", len(json_data))
    #tgt_list_json = [item['ref segment'] for item in json_data]
    src_list_json = [item['src segment'] for item in json_data]
    ante_list = [item['ante distance'] for item in json_data]

    # take intercection of src in text file and src in json file # 11084
    src_intersection_before_sort = list(set(src_list).intersection(src_list_json))
    print ("src_intersec1", len(src_intersection_before_sort)) # 11084
    # Store indices of the intersection both in in text file and json file
    indices_in_src_list = sorted([src_list.index(element) for element in src_intersection_before_sort]) # sorted is needed because list(set().intersection) seems outputs list elements in the different order in each run
    indices_in_json_list = sorted([src_list_json.index(element) for element in src_intersection_before_sort]) # sorted
    # take intersection of antecedent and target in from json file ####################################################################
    ante_intersection = [ante_list[i] for i in indices_in_json_list]
    src_intersection = [src_list_json[i] for i in indices_in_json_list]
    #tgt_intersection = [tgt_list_json[i] for i in indices_in_json_list]

    # Sampling sentences (where antecedent < 2)
    # 1. pop ante 0 : take ante non-zero indices and apply src and tgt 
    print ("ante_intersec", len(ante_intersection)) # 11085
    indices_intersec_non_zero_ante = [i for i, ante in enumerate(ante_intersection) if ante != 0]
    ante_intersec_non_zero = [ante_intersection[i] for i in indices_intersec_non_zero_ante]
    print ("ante_intersec_non_zero", len(ante_intersec_non_zero)) # 8828
    src_intersec_non_zero_ante = [src_intersection[i] for i in indices_intersec_non_zero_ante]
    print ("src_intersec3", len(src_intersec_non_zero_ante)) # 8828
    #tgt_intersec_non_zero_ante = [tgt_intersection[i] for i in indices_intersec_non_zero_ante]

    # subsample 1000 examples indices from antecedent distance = 1  
    sampled_indices = sample_example(criteria="ante==1", original_data=ante_intersec_non_zero, original_indices= [i for i, item in enumerate(ante_intersec_non_zero)], n_samples=1000) # sample 1 (take 1000 of antecedent distance 1)
    sampled_indices = sample_example(criteria="ante==2", original_data=[ante_intersec_non_zero[i] for i in sampled_indices], original_indices= sampled_indices, n_samples=1000) # sample 2 (take 1000 of antecedent distance 2)
    # Apply sampled indices for src and tgt
    sampled_src_intersec = [src_intersec_non_zero_ante[i] for i in sampled_indices]
    #sampled_tgt_intersec = [tgt_intersec_non_zero_ante[i] for i in sampled_indices]
    src_intersec = sampled_src_intersec # sampled
    #tgt_intersec = sampled_tgt_intersec # sammpled
    
    # Take context.txt file and choose one example out of duplications and store in context_list #
    if src_context_size != 0:
        if src_context_size == "ante" or src_context_size == "1-ante":
            max_c = 26 # max context size (max antecedent distant)
        else:
            max_c = src_context_size

        with open(f'{data_path}/contrapro.context.en' , 'r') as file:
            context_list = [line.strip() for line in file]

        contexts = []
        print ("n_sent", n_sent) # 12011
        
        # Choose only one chunk of context out of three chunks repitition 
        #for i, alpha in zip(range(n_sent_intersec)[:5], ante_intersec[:5]):
        for i in range(n_sent): # 12011
            #print (i)
            sentence_context = context_list[i*3*max_c:i*3*max_c+max_c]
            contexts.append(sentence_context)

        print ("contexts length", len(contexts)) # 12011   
        context_chunks_intersection = [contexts[i] for i in indices_in_src_list] # 11084
        print ("context_chunks_intersection", len(context_chunks_intersection))
        
        # Sampling sentences (where antecedent < 2) for context 
        # 1. pop ante 0 : take ante non-zero indices and apply context
        context_chunks_intersec_non_zero_ante = [context_chunks_intersection[i] for i in indices_intersec_non_zero_ante]
        
        # Apply sampled indices for context and antecedent
        sampled_context_chunks_intersec = [context_chunks_intersec_non_zero_ante[i] for i in sampled_indices]
        print (sampled_context_chunks_intersec[:5])
        sampled_ante_intersec = [ante_intersec_non_zero[i] for i in sampled_indices]
        context_chunks_intersec = sampled_context_chunks_intersec # sampled
        ante_intersec_non_zero = sampled_ante_intersec # sampled
        print ("after all sampling", len(src_intersec), len(context_chunks_intersec), len(ante_intersec_non_zero)) # 3290, 3290, 3290, 3290 for sample ante1,  3132 for sample 1+ sample2 
        
        # Context with Antedecedent Distance concatenation
        context_intersec = []
        ante_intersec = []
        """
        if src_context_size == "1-ante":
            # Version 1: Choose only an antecedent sentence as a context sentence
            for sent_context, alpha in zip(context_chunks_intersec, ante_intersec_non_zero):
                ante_context = sent_context[-alpha]
                # if alpha != 0: 
                context_intersec.append(ante_context)
                ante_intersec.append(ante_context)
        """

        if src_context_size == "ante": 
            # Version 2: Choose preceding sentences after antecedent as context sentences
            print ("Version 2!")
            for sent_context, alpha in zip(context_chunks_intersec, ante_intersec_non_zero):
                ante_context = sent_context[-alpha]
                ante_contexts = " ".join(sent_context[-alpha:])
                context_intersec.append(ante_contexts)
                ante_intersec.append(ante_context)

        else: ##### Summarization target ####
            # (Version 3:) Choose all preceding sentences as context sentences

            for sent_context, alpha in zip(context_chunks_intersec, ante_intersec_non_zero):
                #print ("check sent_context", sent_context)
                prec_contexts = " ".join(sent_context)
                context_intersec.append(prec_contexts)
                #print ("alpha", alpha)
                if alpha <= len(sent_context):
                    ante_context = sent_context[-alpha]
                else:
                    ante_context = "Antecedent lies outside the context window"
                ante_intersec.append(ante_context)
            
    ############ common ###################
    inputs = []
    labels = []
    # Context for source sentence

    if "llama" in model_checkpoint or "Llama" in model_checkpoint:
        summ_inst = f"Summarize context:{sep_token}" 

        print (len(context_intersec), len(ante_intersec)) # 3132 for sample 1 + sample 2
        print ("####################", model_checkpoint)
        
        for ip, tgt in zip(context_intersec, ante_intersec):
            concat_input = "### User:\n" + summ_inst + ip + "\n\n### Assistant:\n"            
            inputs.append(concat_input)
            labels.append(tgt)
    
    else: 
        print (len(context_intersec), len(ante_intersec)) # 3132 for sample 1 + sample 2
        print ("####################", model_checkpoint)
        for ip, tgt in zip(context_intersec, ante_intersec):
            concat_input =  ip     
            inputs.append(concat_input)
            labels.append(tgt)

    sources = context_intersec

    return inputs, labels, sources


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

