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
            docs_dict[f"talk_id {talkid}"] = langs_to_segs
    
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
        talk_dict[talk_id] = {}
    
        for lang_dict in lang_dicts:
            talk_dict[talk_id].update(lang_dict[talk_id])
        print (talk_dict)
        doc_dict = {}
        doc_dict["doc"] = talk_dict

        _multilingual_talks_dict.update(doc_dict)
        doc_dict_list.append({"doc": talk_dict})
    
    return doc_dict_list

# Create Parallel json dataset
def generate_parallel_json(parallel_lang_list, data_dir):

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




