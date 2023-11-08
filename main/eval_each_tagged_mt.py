
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
"""
tagged_comet_sum = 0
origin_comet_sum = 0
all_tags_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/ja/printsout.ja"
with open (all_tags_path, "r") as rf:
    for line, comet_score_with_c, comet_score_without_c in zip(rf, comet_with_c_each, comet_without_c_each):
        tags = line.strip()
        origin_comet_sum += comet_score_with_c
        if tags != "":
            tagged_comet_sum += comet_score_with_c
            
        else:
            tagged


tagged_comet_avg = tagged_comet_sum / len(comet_with_c_each)
origin_comet_avg = sum(comet_with_c_each) / len(comet_with_c_each)
without_c_comet_avg = sum(comet_without_c_each) / len(comet_without_c_each)
print ("tagged", tagged_comet_avg)
print ("2-1", origin_comet_avg)
print ("1-1", without_c_comet_avg)


sacrebleu_without_c_each_path = "/mnt/data-poseidon/sumire/thesis/1-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-1-1/sacrebleu_each_score.txt"
sacrebleu_without_c_each = []
with open(sacrebleu_without_c_each_path, "r") as rf:
   for line in rf:
        score = np.round(float(line.strip()), 4)
        sacrebleu_without_c_each.append(score)

sacrebleu_with_c_diff = "/mnt/data-poseidon/sumire/thesis/2-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/sacrebleu_score_diff.txt"
sacrebleu_with_c_each = []
with open(sacrebleu_with_c_diff, "r") as rf:
   for line, without_c_score in zip(rf, sacrebleu_without_c_each):
        diff = np.round(float(line.strip()), 4)
        #print (diff)
        with_c_score = without_c_score + diff
        #print (with_c_score)
        sacrebleu_with_c_each.append(with_c_score)

#print (sacrebleu_with_c_each, len(sacrebleu_with_c_each))

sacrebleu_with_c_each_path = "/mnt/data-poseidon/sumire/thesis/2-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/sacrebleu_each_score.txt"
with open(sacrebleu_with_c_each_path, "w") as wf:
    #for line, with_c_score in wf:
    for with_c_score in sacrebleu_with_c_each:
        wf.write(f"{with_c_score}\n")

sacrebleu_sum = 0
num_of_summed = 0
origin_sacrebleu_sum = 0
with open ("/mnt/data-poseidon/sumire/thesis/running/context_classifier/ted-ja/2-1/test/base-uncased-2e-5_ep10-batch-lesszerolabel_2/prediction/prediction.txt", "r") as rf:
    for line, sacrebleu_score in zip(rf, sacrebleu_with_c_each):
        score = int(line.strip())
        origin_sacrebleu_sum += sacrebleu_score
        if score ==1:
            sacrebleu_sum += sacrebleu_score
            num_of_summed += 1

sacrebleu_avg = sacrebleu_sum / num_of_summed
origin_sacrebleu_avg = origin_sacrebleu_sum / len(sacrebleu_with_c_each)
print (sacrebleu_avg)
print (origin_sacrebleu_avg)


comet_without_c_each_path = "/mnt/data-poseidon/sumire/thesis/1-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-1-1/comet_each_score.txt"
comet_without_c_each = []
with open(comet_without_c_each_path, "r") as rf:
   for line in rf:
        score = np.round(float(line.strip()), 4)
        comet_without_c_each.append(score)

sacrebleu_sum = 0
origin_sacrebleu_sum = 0
with open ("/mnt/data-poseidon/sumire/thesis/running/context_classifier/ted-ja/2-1/test/base-uncased-2e-5_ep10-batch-lesszerolabel_2/prediction/prediction.txt", "r") as rf:
    for line, sacrebleu_score_with_c, sacrebleu_score_without_c in zip(rf, sacrebleu_with_c_each, sacrebleu_without_c_each):
        score = int(line.strip())
        origin_sacrebleu_sum += sacrebleu_score_with_c
        if score ==1:
            sacrebleu_sum += sacrebleu_score_with_c
            
        else:
            sacrebleu_sum += sacrebleu_score_without_c


sacrebleu_avg = sacrebleu_sum / len(sacrebleu_with_c_each)
origin_sacrebleu_avg = origin_sacrebleu_sum / len(sacrebleu_with_c_each)
print (sacrebleu_avg)
print (origin_sacrebleu_avg)

"""