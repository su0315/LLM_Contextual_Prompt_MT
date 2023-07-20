from datasets import load_dataset
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import json
from combine_xml_data import build_dictionary_from_xml, get_shared_talkids, combine_langs_into_one_dict


# Read XML Dataset 
parallel_lang_list = [["en", "ja"],["en", "ar"], ["en",  "fr"], ["en", "de"], ["en", "ko"], ["en", "zh"]]
rel_path = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/"

# Create Parallel json dataset
for parallel in parallel_lang_list:
    parallel_lang_dicts = []
    tgt_lang = parallel[1]
    
    for lang in parallel:
        xml_filepath_list = [
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.dev2010.{tgt_lang}-en.{lang}.xml",
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.tst2010.{tgt_lang}-en.{lang}.xml",
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.tst2011.{tgt_lang}-en.{lang}.xml",
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.tst2012.{tgt_lang}-en.{lang}.xml",
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.tst2013.{tgt_lang}-en.{lang}.xml",
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.tst2014.{tgt_lang}-en.{lang}.xml",
            rel_path+f"{tgt_lang}-en/IWSLT17.TED.tst2015.{tgt_lang}-en.{lang}.xml",
                           ]
        docs_dict = build_dictionary_from_xml(xml_filepath_list, lang)
        parallel_lang_dicts.append(docs_dict)
    shared_talk_ids = get_shared_talkids(lang_dicts=parallel_lang_dicts)
    parallel_talks_dict = combine_langs_into_one_dict(lang_dicts = parallel_lang_dicts, shared_talk_ids=shared_talk_ids)
    with open(f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_{parallel[0]}-{parallel[1]}", "w", encoding="utf-8") as outfile:
        json.dump(parallel_talks_dict, outfile, indent=4, ensure_ascii=False)
