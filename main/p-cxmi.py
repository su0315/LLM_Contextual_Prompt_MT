import numpy as np
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
import json
import re
import os

def read_args():
    parser = ArgumentParser(description='Read arguments')
    parser.add_argument('--in_path_context', type=str, help='The path for translation with context', required=True)
    parser.add_argument('--in_path_base', type=str, help='The path for translation without context', required=True)
    parser.add_argument('--out_path',  type=str, help = "The path the CXMI socre will be stored", required=True)
    parser.add_argument('--muda_tag_path', type=str, help = "The path the MuDA tagged reference file socre will be stored", required=True)
    args = parser.parse_args()
    in_path_context = args.in_path_context
    in_path_base = args.in_path_base
    out_path = args.out_path
    muda_tag_path = args.muda_tag_path

    return in_path_context, in_path_base, out_path, muda_tag_path

def is_substring(larger_str, smaller_str):
    return smaller_str in larger_str

def take_continuous_correlation(larger_str, smaller_tokens):
    current_continuous_correlation = []
    checked_larger_str = larger_str
    for smaller_item in smaller_tokens:
        for smaller_key, smaller_value in smaller_item.items():
            smaller_str = smaller_key
            smaller_score = smaller_value # Convert to lowercase for case-insensitive comparison            
            if is_substring(checked_larger_str, smaller_str):
                current_continuous_correlation.append({smaller_str: smaller_score})
                checked_larger_str = checked_larger_str[len(smaller_str):]  # Remove matched substring from larger_str
                if current_continuous_correlation and checked_larger_str == "":
                    return current_continuous_correlation
            else:
                #larger_str = original_larger_str  # Reset to the original larger_str
                if current_continuous_correlation and checked_larger_str !="": # if couldn_t complete the larger_str correlation, remove the partial correlation
                    current_continuous_correlation.remove(current_continuous_correlation[-1])
    if checked_larger_str:
        print ("original_larger", larger_str)  # TODO: Thre are lots of missed case already here, why
        print ("smaller", smaller_tokens)
    #print (current_continuous_correlation)
    return current_continuous_correlation if not checked_larger_str else []


def extract_continuous_correlations(larger_tokens, smaller_tokens, phe):
    no_counted=[]
    relevant_items = []
    current_continuous_correlation = []

    for item in larger_tokens:
        for key, value in item.items():
            if phe != "ALL":
                if value and phe in value:
                    ### repetition ###
                    larger_str = key 
                    continuous_correlation = take_continuous_correlation(larger_str, smaller_tokens) ###here
                    if continuous_correlation:
                        if current_continuous_correlation:  # Check if there's a previous correlation
                            relevant_items.append(current_continuous_correlation)
                            current_continuous_correlation = []  # Reset for the next correlation
                        relevant_items.append(continuous_correlation)
                        break  # Exit loop if a relevant item is found
                    else:
                        #print ("not counted tagged token:", larger_str)
                        no_counted_token=larger_str
                        no_counted.append(no_counted_token)
                        current_continuous_correlation.extend(continuous_correlation)
                    ### r ###
            else:
                ### repetition ###
                larger_str = key 
                continuous_correlation = take_continuous_correlation(larger_str, smaller_tokens) ###here
                if continuous_correlation:
                    if current_continuous_correlation:  # Check if there's a previous correlation
                        relevant_items.append(current_continuous_correlation)
                        current_continuous_correlation = []  # Reset for the next correlation
                    relevant_items.append(continuous_correlation)
                    break  # Exit loop if a relevant item is found
                else:
                    no_counted_token=larger_str
                    no_counted.append(no_counted_token)
                    current_continuous_correlation.extend(continuous_correlation)
                ### r ###
    if current_continuous_correlation:  # Check if there's a continuous correlation at the end
        relevant_items.append(current_continuous_correlation)

    return relevant_items, no_counted

def decode_byte(smaller_tokens, output_dir, phe):
    continuous = False
    decoded_smaller_tokens=[]
    for smaller_item in smaller_tokens:
        for smaller_key, smaller_value in smaller_item.items():
            smaller_str = smaller_key
            if '<0x' in smaller_str:
                #print (smaller_str, smaller_value)
                if continuous == False:
                    continuous_hex = []
                    continuous_hex_score = []
                hex = smaller_str.replace('<0x', '').replace('>', '')
                hex = (int(hex, 16))
                hex_score = smaller_value
                continuous_hex.append(hex)
                continuous_hex_score.append(hex_score)
                
                
                continuous = True
            else:
                if continuous == True:
                    continuous=False
                    byte_string = bytes(continuous_hex)
                    decoded_string = byte_string.decode('utf-8')
                    
                    #print (decoded_string)
                    
                    #print (continuous_hex_score)
                    decoded_string_score = sum(continuous_hex_score)
                    #print (decoded_string_score)
                    
                    decoded_item = {decoded_string:decoded_string_score}
                    #smaller_tokens[smaller_tokens.index(smaller_item)] = decoded_item
                    
                    #for i in range(len(continuous_hex)):# Iterate over the left side of the last continuous_hex index
                        #smaller_tokens.pop(smaller_tokens.index(smaller_item)-i)
                        #print (decoded_item)
                        #print (smaller_item)
                        #print (smaller_tokens.index(smaller_item))
                        #print (int(smaller_tokens.index(smaller_item)-i-1)) # current index is after continuous hex
                        #smaller_tokens[smaller_tokens.index(smaller_item)-i-1] = decoded_item
                    decoded_smaller_tokens.append(decoded_item)
                    if phe == "ALL":
                        with open(output_dir+"/mapped_token", "a") as wf:
                            wf.write(f"{decoded_item}\n")
                    decoded_smaller_tokens.append(smaller_item)
                else:
                    decoded_smaller_tokens.append(smaller_item)           
    #print (smaller_tokens)
    return decoded_smaller_tokens

def p_cxmi(in_path_context, in_path_base, out_path, muda_tag_path):
    bc_text_to_score = {} # map base or context
    for in_path in [in_path_context, in_path_base]:
        text_to_score=[]
        with open(f"{in_path}/ref_scores_details.txt", "r") as rf:
            rf = rf.readlines()
            for scores_sent in rf:
                score_detail = scores_sent.strip().split(", ")
                logprob_pattern = re.compile(r'logprob=(-?\d+\.\d+)')
                text_pattern = re.compile(r"text='([^']+)'")
                # Find matches in the content
                logprob_matches = logprob_pattern.findall(scores_sent)
                text_matches = text_pattern.findall(scores_sent)
                # Combine the matches into a list of tuples
                text_to_score_per_sent = []

                for score, text in zip(logprob_matches, text_matches):      
                    text_to_score_per_sent.append({text:float(score)})
                    # text_to_score_per_sent[text] = score
                    #print (score, text)
                    #print (text_to_score_per_sent)
                text_to_score.append(text_to_score_per_sent)
        if in_path == in_path_context:
            bc_text_to_score["context"] = text_to_score
        elif in_path == in_path_base:
            bc_text_to_score["base"] = text_to_score
            

        #base_sent_scores = np.array([score for score in scores_sent.strip().split(", ")])
    
    phenomena = []
    with open(f"{muda_tag_path}", "r") as rf:
        tagged_tokens = []
     
        docs = json.load(rf)
       
        for doc in docs:
            for sent_id, sent in enumerate(doc):
                tagged_token_in_sent = []
                for token_pos, token in enumerate(sent):
                    #if any(token["tags"] for token in sent):
                    #if token["tags"]:
                    tagged_token_in_sent.append({token["token"]:token["tags"]})

                    for phenomenon in token["tags"]:
                        if phenomenon not in phenomena:
                            phenomena.append(phenomenon)
                tagged_tokens.append(tagged_token_in_sent) # List where each element has list of tagged token in sent
        
    phenomena.append("ALL") # calculate per-token cxmi for all tokens in the data
    phe_to_scores = {}
    
    for phe in phenomena:
        no_counted_tokens = []
        bc_p_scores = {}
        for b_or_c in list(bc_text_to_score.keys()):
            all_p_scores = []
            for tagged_token_per_sent, token_or_letter_score in zip(tagged_tokens, bc_text_to_score[b_or_c]): # Check each sent in muda tag data and prediction data
                #no_counted_tokens_per_sent = []
                #if tagged_token_per_sent != []: #
                    #print ("muda tagged token per sent", tagged_token_per_sent)#larger_tokens
                    #print ("llama token per sent", token_or_letter_score)#smaller_tokens
                larger_tokens=tagged_token_per_sent
                smaller_tokens=token_or_letter_score
                #print ("smallerrrrr", smaller_tokens)
                # TODO. Translate smaller_tokens to non-byte encoding sentence
                decoded_smaller_tokens = decode_byte(smaller_tokens, out_path, phe)
                #print (decoded_smaller_tokens)
                
                
                result, no_counted= extract_continuous_correlations(larger_tokens, decoded_smaller_tokens, phe)
                if no_counted:
                    no_counted_tokens.append(no_counted)
                
                for p_token in result: # Sum and Store p_scores per tokens, not per sentence!!!!
                    p_score = (sum([smaller_token_score for dictionary in p_token for smaller_token, smaller_token_score in dictionary.items()]))
                    all_p_scores.append(p_score)

                #print (len(all_p_scores)) . number of phenomena tokens are not matching to muda data, roughly 100-500 smaller
                #no_counted_tokens.append(no_counted_tokens_per_sent)
            bc_p_scores[b_or_c]=all_p_scores
        phe_to_scores[phe] = bc_p_scores
            
    
        # for each phenomena, calculate p_cxmi
        base_sent_scores = np.array([score for score in phe_to_scores[phe]["base"]])
        context_sent_scores = np.array([score for score in phe_to_scores[phe]["context"]])
        
        if len(base_sent_scores) > 0 and len(context_sent_scores) > 0:
            print (f"number of {phe}", (len(base_sent_scores), len(context_sent_scores)))
            p_cxmi = (np.array(base_sent_scores) - np.array(context_sent_scores)) # TODO: ValueError: operands could not be broadcast together with shapes (280,) (281,) 
            max_p_cxmi_id = np.argmax(-p_cxmi)
            max_p_cxmi_score = -p_cxmi[max_p_cxmi_id]
            p_cxmi = - (np.mean(p_cxmi))
            print (phe, p_cxmi)
        else:
            print (f"{phe} has no value")
            print (len(base_sent_scores), len(context_sent_scores))
            p_cxmi = 0
            max_p_cxmi_score = 0
            max_p_cxmi_id = 0

        category_dir = out_path+f"/categorized_test/{phe}"
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        with open(category_dir+f"/p-cxmi.txt", "a") as wf:
            wf.write(f"P-CXMI ({phe}):\n{p_cxmi} (MAX_CXMI: {max_p_cxmi_score} (sent id: {max_p_cxmi_id}))\n")
            wf.write(f"number of {phe}:{(len(base_sent_scores), len(context_sent_scores))} \n")
            wf.write(f"Not Counted Tokens of {phe}: {int(len([no_counted_token for sent in no_counted_tokens for no_counted_token in sent])/2)}\n")
            for no_counted_sent in no_counted_tokens[:int(len(no_counted_tokens)/2)]:
                if no_counted_sent:
                    wf.write(f"{no_counted_sent}\n")
                    
            wf.write("\n")

    return p_cxmi, max_p_cxmi_score, max_p_cxmi_id

        
    
def main():
    in_path_context, in_path_base, out_path, muda_tag_path = read_args()
    p_cxmi(in_path_context, in_path_base, out_path, muda_tag_path)

if __name__ == "__main__":
    main()
    
