import re
# Function to convert continuous hexadecimal values to UTF-8
def convert_continuous_hex_to_utf8(continuous_hex):
    # Split the continuous_hex string into individual hexadecimal values
    hex_values = [continuous_hex[i:i+4] for i in range(0, len(continuous_hex), 4)]

    # Convert each hexadecimal value to Unicode
    utf8_result = ''
    for hex_value in hex_values:
        if hex_value.startswith('<0x') and hex_value.endswith('>'):
            # Extract the hexadecimal value and convert to Unicode
            unicode_char = chr(int(hex_value[3:-1], 16))
            utf8_result += unicode_char
        else:
            # If not in the expected format, treat as is
            utf8_result += hex_value

    return utf8_result

# Example usage
continuous_hex_values = "<0xE7><0xB5><0x82>"
utf8_result = convert_continuous_hex_to_utf8(continuous_hex_values)

print(utf8_result)

path = "/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/test/en-ja/cxmi-Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/ref_scores_details.txt"
text_to_score=[]
with open(path, "r") as rf:
    rf = rf.readlines()
    for scores_sent in rf:#TODO
        score_detail = scores_sent.strip().split(", ")
        logprob_pattern = re.compile(r'logprob=(-?\d+\.\d+)')
        text_pattern = re.compile(r"text='([^']+)'")
        # Find matches in the content
        logprob_matches = logprob_pattern.findall(scores_sent)
        text_matches = text_pattern.findall(scores_sent)
        #print (text_matches)
        
        """
        # Combine the matches into a list of tuples
        text_to_score_per_sent = []

        for score, text in zip(logprob_matches, text_matches):      
            text_to_score_per_sent.append({text:float(score)})
            # text_to_score_per_sent[text] = score
            #print (score, text)
            #print (text_to_score_per_sent)
        text_to_score.append(text_to_score_per_sent)
        """