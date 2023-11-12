import numpy as np
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)

def read_args():
    parser = ArgumentParser(description='Read arguments')
    parser.add_argument('--in_path_context', type=str, help='The path for translation with context', required=True)
    parser.add_argument('--in_path_base', type=str, help='The path for translation without context', required=True)
    parser.add_argument('--out_path',  type=str, help = "The path the CXMI socre will be stored", required=True)
    args = parser.parse_args()
    in_path_context = args.in_path_context
    in_path_base = args.in_path_base
    out_path = args.out_path

    return in_path_context, in_path_base, out_path

def cxmi(in_path_context, in_path_base, out_path):

    with open(f"{in_path_base}/ref_scores.txt", "r") as rf:
        all_base_scores = []
        for scores_sent in rf:
            base_sent_scores = [float(score) for score in scores_sent.strip().split(", ")]### TODO sum over sent and average all sent
            all_base_scores.append(sum(base_sent_scores))
        all_base_scores = np.array(all_base_scores)
    with open(f"{in_path_context}/ref_scores.txt", "r") as rf:
        all_context_scores = []
        counter = 0
        for scores_sent in rf:
            context_sent_scores = [float(score) for score in scores_sent.strip().split(", ")]
            all_context_scores.append(sum(context_sent_scores))
            counter += 1
        all_context_scores = np.array(all_context_scores)
        print (counter)
    print (len(all_base_scores), len(all_base_scores))
    cxmi_per_sent = all_base_scores - all_context_scores
    print (len(cxmi_per_sent))
    max_cxmi_sent_id = np.argmax(-(cxmi_per_sent))
    max_cxmi_score = -cxmi_per_sent[max_cxmi_sent_id]
    cxmi = - (np.mean(cxmi_per_sent))

    with open(out_path, "w") as wf:
        wf.write(f"CXMI: {cxmi}\nMAX_CXMI: {max_cxmi_score} (sent id: {max_cxmi_sent_id})")

    return cxmi, max_cxmi_score, max_cxmi_sent_id

def main():
    in_path_context, in_path_base, out_path = read_args()
    cxmi(in_path_context, in_path_base, out_path)

if __name__ == "__main__":
    main()
