
import numpy as np

comet_without_c_each_path = "/mnt/data-poseidon/sumire/thesis/1-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-1-1/comet_each_score.txt"
comet_without_c_each = []
with open(comet_without_c_each_path, "r") as rf:
   for line in rf:
        score = np.round(float(line.strip()), 4)
        comet_without_c_each.append(score)


comet_with_c_each_path = "/mnt/data-poseidon/sumire/thesis/2-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/comet_each_score.txt"
comet_with_c_each = []
with open (comet_with_c_each_path, "r") as rf:
    for line in rf:
        score = np.round(float(line.strip()), 4)
        comet_with_c_each.append(score)

tagged_comet_sum = 0
origin_comet_sum = 0
all_tags_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/ted/ja/sent_all_tags_3.ja"
with open (all_tags_path, "r") as rf:
    for line, comet_score_with_c, comet_score_without_c in zip(rf, comet_with_c_each, comet_without_c_each):
        tags = line.strip()
        origin_comet_sum += comet_score_with_c
        if tags != "":
            tagged_comet_sum += comet_score_with_c
            
        else:
            tagged_comet_sum += comet_score_without_c


tagged_comet_avg = tagged_comet_sum / len(comet_with_c_each)
origin_comet_avg = sum(comet_with_c_each) / len(comet_with_c_each)
without_c_comet_avg = sum(comet_without_c_each) / len(comet_without_c_each)

