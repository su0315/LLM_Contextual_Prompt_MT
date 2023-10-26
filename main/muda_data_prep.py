from datasets import load_dataset, concatenate_datasets, load_from_disk

def mudadata_for_ted(data_path, tgt_lang, output_dir):
    data_files = { "train": f"{data_path}train_ted_en-{tgt_lang}",  "test": f"{data_path}test_ted_en-{tgt_lang}"}
    dataset = load_dataset("json", data_files=data_files)

    data = dataset["test"]

    doc_ids = []
    for doc_idx, doc in enumerate(data["doc"]):
        for sent in doc["en"]:
            doc_ids.append(doc_idx)

    #doc_ids = [doc_idx for doc_idx, doc in enumerate(data["doc"])]
    src = [sent for doc in data["doc"] for sent in doc["en"]]
    tgt = [sent for doc in data["doc"] for sent in doc[tgt_lang]]

    with open(output_dir+f'ted/ted_test_src.en','w', encoding='utf8') as wf:
        for item in src:
            wf.write(item+ '\n')

    with open(output_dir+f'ted/ted_test_tgt.{tgt_lang}','w', encoding='utf8') as wf:
        for item in tgt:
            wf.write(item+ '\n')

    with open(output_dir+f'ted/ted_test.docids','w', encoding='utf8') as wf:
        for item in doc_ids:
            print (item)
            wf.write(f"{item}\n")

ted_data_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/"
output_dir = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/muda_tagged/"
mudadata_for_ted(data_path=ted_data_path, tgt_lang="ja", output_dir=output_dir)
