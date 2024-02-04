
import numpy as np



comet_without_c_each_path = "/mnt/data-poseidon/sumire/thesis/1-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-1-1/comet_each_score.txt"
comet_without_c_each = []
with open(comet_without_c_each_path, "r") as rf:
   for line in rf:
        score = np.round(float(line.strip()), 4)
        comet_without_c_each.append(score)


comet_with_c_each_path = "/mnt/data-poseidon/sumire/thesis/5-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-5-1/comet_each_score.txt"
comet_with_c_each = []
with open (comet_with_c_each_path, "r") as rf:
    for line in rf:
        score = np.round(float(line.strip()), 4)
        comet_with_c_each.append(score)

# Caluculate score for each tags
formality_count = 0
pro_count = 0
lexical_count = 0
formality_comet_sum = 0
formality_origin_comet_sum = 0
pro_comet_sum = 0
pro_origin_comet_sum = 0
lexical_comet_sum = 0
lexical_origin_comet_sum = 0
all_tags_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/ja/sent_all_tags_3.ja"
with open (all_tags_path, "r") as rf:
    for line, comet_score_with_c, comet_score_without_c in zip(rf, comet_with_c_each, comet_without_c_each):
        if "formality" in line:
            formality_count += 1
            formality_comet_sum += comet_score_with_c
            formality_origin_comet_sum += comet_score_without_c
        #else:
            #formality_comet_sum += comet_score_without_c
        if "pronouns" in line:
            pro_count += 1
            pro_comet_sum += comet_score_with_c
            pro_origin_comet_sum += comet_score_without_c
        #else:
            #pro_comet_sum += comet_score_without_c
        if "lexical_cohesion" in line:
            lexical_count += 1
            lexical_comet_sum += comet_score_with_c
            lexical_origin_comet_sum += comet_score_without_c
        #else:
            #lexical_comet_sum += comet_score_without_c

# formality
formality_comet_avg = formality_comet_sum / formality_count
formality_origin_comet_avg = formality_origin_comet_sum / formality_count

# Pronouns
pro_comet_avg = pro_comet_sum / pro_count
pro_origin_comet_avg = pro_origin_comet_sum / pro_count

# Lexical
lexical_comet_avg = lexical_comet_sum / lexical_count
lexical_origin_comet_avg = lexical_origin_comet_sum / lexical_count


print ("formality 5-1", formality_comet_avg)
print ("formality-1-1", formality_origin_comet_avg)
print ("formal", formality_count)
print ()

print ("Pronoun 5-1", pro_comet_avg)
print ("Pronoun 1-1", pro_origin_comet_avg)
print ("pro", pro_count)
print ()

print ("lexical 5-1", lexical_comet_avg)
print ("lexical 1-1", lexical_origin_comet_avg)
print ("lexical", lexical_count)
print ()
