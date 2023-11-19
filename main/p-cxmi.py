import numpy as np
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
import json
import re

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
    original_larger_str = larger_str
    for smaller_item in smaller_tokens:
        for smaller_key, smaller_value in smaller_item.items():
            smaller_str = smaller_key  # Convert to lowercase for case-insensitive comparison            
            if is_substring(larger_str, smaller_str):
                current_continuous_correlation.append({smaller_key: smaller_value})
                larger_str = larger_str[len(smaller_str):]  # Remove matched substring from larger_str
                if current_continuous_correlation and larger_str == "":
                    return current_continuous_correlation
            else:
                larger_str = original_larger_str  # Reset to the original larger_str
                if current_continuous_correlation and larger_str !="": # if couldn_t complete the larger_str correlation, remove the partial correlation
                    current_continuous_correlation.remove(current_continuous_correlation[-1])
                
    return current_continuous_correlation if not larger_str else []

def extract_continuous_correlations(larger_tokens, smaller_tokens, phe):
    relevant_items = []
    current_continuous_correlation = []

    for item in larger_tokens:
        for key, value in item.items():
            if value and phe in value:
                larger_str = key  # Convert to lowercase for case-insensitive comparison
                continuous_correlation = take_continuous_correlation(larger_str, smaller_tokens) ###here
                if continuous_correlation:
                    if current_continuous_correlation:  # Check if there's a previous correlation
                        relevant_items.append(current_continuous_correlation)
                        current_continuous_correlation = []  # Reset for the next correlation
                    relevant_items.append(continuous_correlation)
                    break  # Exit loop if a relevant item is found
                else:
                    current_continuous_correlation.extend(continuous_correlation)

    if current_continuous_correlation:  # Check if there's a continuous correlation at the end
        relevant_items.append(current_continuous_correlation)

    return relevant_items


def p_cxmi(in_path_context, in_path_base, out_path, muda_tag_path):
    bc_text_to_score = {} # map base or context
    for in_path in [in_path_context, in_path_base]:
        text_to_score=[]
        with open(f"{in_path}/ref_scores_details.txt", "r") as rf:
            rf = rf.readlines()
            for scores_sent in rf:#TODO
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
       
        for doc in docs:#TODO
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
        

    phe_to_scores = {}
    for phe in phenomena:
        bc_p_scores = {}
        for b_or_c in list(bc_text_to_score.keys()):
            all_p_scores = []
            for tagged_token_per_sent, token_or_letter_score in zip(tagged_tokens, bc_text_to_score[b_or_c]): # Check each sent in muda tag data and prediction data
                #if tagged_token_per_sent != []: #
                    #print ("muda tagged token per sent", tagged_token_per_sent)#larger_tokens
                    #print ("llama token per sent", token_or_letter_score)#smaller_tokens
                larger_tokens=tagged_token_per_sent
                smaller_tokens=token_or_letter_score
                result = extract_continuous_correlations(larger_tokens, smaller_tokens, phe)
                n_smaller_tokens = len([dictionary for list in result for dictionary in list])
                if n_smaller_tokens > 0:
                    p_score = (sum([smaller_token_score for list in result for dictionary in list for smaller_token, smaller_token_score in dictionary.items()])) / n_smaller_tokens
                    #print (len([dictionary for list in result for dictionary in list]))
                    #print (p_score)
                    all_p_scores.append(p_score)

        #print (phe)
        #print (all_p_scores)
        #print (len(all_p_scores)) #TODO. number of phenomena tokens are not matching to muda data, roughly 100-500 smaller

                    """
                    p_sent_str= ""
                    for ref_token_to_score_dict in token_or_letter_score: # TODO # Smaller Unit token (llama tokenizer)
                        ref_token, score = list(ref_token_to_score_dict.items())[0]
                        p_sent_str += ref_token
                        #print (p_sent_str)
                        print ("ref_token", ref_token)
                    
                    for tagged_token_to_tag in tagged_token_per_sent: # Larger Unit token (MuDA Tag)
                        
                        tagged_token, tag = list(tagged_token_to_tag.items())[0]
                        print ("tagged_token", tagged_token)
                        
                        #print (p_sent_str)
                        #print (tagged_token, phe, p_sent_str)
                        #print (phe, tag)

                        # compare tagged_token (MuDA tokenizer) and ref_token (Llma tokenizer)
                        
                        
                        if tagged_token in p_sent_str and  phe in tag:
                            p_scores = 0
                            n_letter_in_token = 0
                            if ref_token in tagged_token:
                                print ("yes1")
                                n_letter_in_token += 1
                                p_scores += score
                                p_scores = p_scores / n_letter_in_token
                                #print (p_scores)
                                all_p_scores.append(p_scores)
                    """         
            #print (tagged_token, phe, p_sent_str)
            bc_p_scores[b_or_c]=all_p_scores
        phe_to_scores[phe] = bc_p_scores
    # print (phe_to_scores)
    #print (len(phe_to_scores["formality"]["base"]), len(phe_to_scores["formality"]["context"]))

    
        # for each phenomena, calculate p_cxmi
        base_sent_scores = np.array([score for score in phe_to_scores[phe]["base"]])
        context_sent_scores = np.array([score for score in phe_to_scores[phe]["context"]])
        
        if len(base_sent_scores) > 0 and len(context_sent_scores) > 0:
            p_cxmi = (np.array(base_sent_scores) - np.array(context_sent_scores))
            max_p_cxmi_id = np.argmax(-p_cxmi)
            max_p_cxmi_score = -p_cxmi[max_p_cxmi_id]
            p_cxmi = - (np.mean(p_cxmi))
            print (phe, p_cxmi)
        else:
            print (f"{phe} has no value")
            p_cxmi = 0
            max_p_cxmi_score = 0
            max_p_cxmi_id = 0
    
        with open(out_path, "a") as wf:
            wf.write(f"P-CXMI ({phe}): {p_cxmi} (MAX_CXMI: {max_p_cxmi_score} (sent id: {max_p_cxmi_id})) \n")
    
    return p_cxmi, max_p_cxmi_score, max_p_cxmi_id
        
    
def main():
    in_path_context, in_path_base, out_path, muda_tag_path = read_args()
    p_cxmi(in_path_context, in_path_base, out_path, muda_tag_path)

if __name__ == "__main__":
    main()
    
