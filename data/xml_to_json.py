import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import json
import jsonlines

def build_dictionary_from_xml(xml_filepath_list, lang):
    """
    Build a dictionary for combined xml files of a language
    """
    files_list = []
    for fp in xml_filepath_list:
        with open(fp, 'r', encoding="utf-8") as f: # , 
            data = f.read()
        bs_data = BeautifulSoup(data, "xml")
        docs = bs_data.find_all('doc')
        files_list.append(docs)

    docs_dict = {}
    for docs in files_list:
        for doc in docs:
        
            #docid = int(doc['docid'])
            talkid = int(doc.find('talkid').get_text())
            seg_elements = doc.find_all('seg')
            segs = [seg.get_text(strip=True) for seg in seg_elements]
            langs_to_segs = {} 
            langs_to_segs[lang] = segs
            docs_dict[talkid] = langs_to_segs
    
    return docs_dict

def get_shared_talkids(lang_dicts):
    shared_keys = set(lang_dicts[0].keys())
    for i in lang_dicts[1:]:
        shared_keys.intersection_update(i.keys())
    return shared_keys


def combine_langs_into_one_dict(lang_dicts, shared_talk_ids):
    
    _multilingual_talks_dict = {}
    doc_dict_list = []

    for talk_id in shared_talk_ids:
        talk_dict = {}
        #talk_dict[talk_id] = {}
    
        for lang_dict in lang_dicts:
            talk_dict.update(lang_dict[talk_id])
        doc_dict = {}
        doc_dict["talk_id"] = talk_id
        doc_dict["doc"] = talk_dict
        
        _multilingual_talks_dict.update(doc_dict)
        doc_dict_list.append(doc_dict)
    
    return doc_dict_list

# Create Parallel json dataset
def generate_parallel_json(parallel_lang_list, data_dir, prompt_talk_id):

    for parallel in parallel_lang_list:
        parallel_lang_dicts = []
        tgt_lang = parallel[1]
        
        for lang in parallel:
            xml_filepath_list = [
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.dev2010.{tgt_lang}-en.{lang}.xml",
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.tst2010.{tgt_lang}-en.{lang}.xml",
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.tst2011.{tgt_lang}-en.{lang}.xml",
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.tst2012.{tgt_lang}-en.{lang}.xml",
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.tst2013.{tgt_lang}-en.{lang}.xml",
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.tst2014.{tgt_lang}-en.{lang}.xml",
                data_dir+f"{tgt_lang}-en/IWSLT17.TED.tst2015.{tgt_lang}-en.{lang}.xml",
                            ]
            docs_dict = build_dictionary_from_xml(xml_filepath_list, lang)
            parallel_lang_dicts.append(docs_dict)
        shared_talk_ids = get_shared_talkids(lang_dicts=parallel_lang_dicts)
        parallel_talks_dict = combine_langs_into_one_dict(lang_dicts = parallel_lang_dicts, shared_talk_ids=shared_talk_ids)    

        with open(data_dir+f"/ted_{parallel[0]}-{parallel[1]}", "w", encoding="utf-8") as outfile:
            json.dump(parallel_talks_dict, outfile, indent=4, ensure_ascii=False)


    # Drop instances of ja/fr/ko/ar/zh not existing in de
    df_ja = pd.read_json(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-ja")
    df_zh = pd.read_json(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-zh")
    df_ko = pd.read_json(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-ko")
    df_ar = pd.read_json(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-ar")
    df_de = pd.read_json(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-de")
    df_fr = pd.read_json(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-fr")
    
    lang_df_list = [df_ja, df_fr, df_ko, df_ar, df_zh] #  "de"
    dropped_df_list = []

    for df in lang_df_list:
        dropped_rows = []

        for talk_id in df["talk_id"].to_numpy():
            if talk_id not in df_de["talk_id"].to_numpy():
                dr_row = df[df["talk_id"]==talk_id].index.item()
                dropped_rows.append(dr_row)
            if talk_id == prompt_talk_id:
                dr_row = df[df["talk_id"]==talk_id].index.item()
                dropped_rows.append(dr_row)

        dropped_df = df.drop(dropped_rows)
        dropped_df.reset_index(inplace=True, drop=True)
        dropped_df_list.append(dropped_df)

    # Drop instance of de not existing in others
    dropped_rows = []

    for talk_id in df_de["talk_id"].to_numpy():
        if talk_id not in df_fr["talk_id"].to_numpy():
            dr_row = df_de[df_de["talk_id"]==talk_id].index.item()
            dropped_rows.append(dr_row)
        
        if talk_id == prompt_talk_id:
                dr_row = df[df["talk_id"]==talk_id].index.item()
                dropped_rows.append(dr_row)

    dropped_df_de = df_de.drop(dropped_rows)
    dropped_df_de.reset_index(inplace=True, drop=True)
    dropped_df_list.append(dropped_df_de)
    print (dropped_df_list)

    lang_list = ["ja", "fr", "ko", "ar","zh","de"] 

    for lang, dropped_df in zip (lang_list, dropped_df_list):
        json_object = dropped_df.to_json(force_ascii=False, orient="records", indent = 4)
        with open(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_en-{lang}", "w", encoding='utf-8') as outfile:
            outfile.write(json_object)

    # Now all of the data has 92 documents 

