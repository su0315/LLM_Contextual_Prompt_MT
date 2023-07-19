from datasets import load_dataset
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import json
from combine_xml_data import build_dictionary_from_xml, get_shared_talkids, combine_langs_into_one_dict

# Read XML Data 
en_xml_filepath_list = ["/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.dev2010.ja-en.en.xml",
                        "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.tst2010.ja-en.en.xml",
                        "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.tst2011.ja-en.en.xml",
                        "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.tst2012.ja-en.en.xml",
                        "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.tst2013.ja-en.en.xml",
                        "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.tst2014.ja-en.en.xml",
                        "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ja-en/IWSLT17.TED.tst2015.ja-en.en.xml",
                       ]

docs_dict_en = build_dictionary_from_xml(en_xml_filepath_list, lang="en")
lang_list = ["ar", "fr", "zh", "ko", "de", "ja"]
lang_dicts = []
lang_dicts.append(docs_dict_en)
for lang in lang_list:
    xml_filepath_list = [f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.dev2010.{lang}-en.{lang}.xml",
                        f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.tst2010.{lang}-en.{lang}.xml",
                        f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.tst2011.{lang}-en.{lang}.xml",
                        f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.tst2012.{lang}-en.{lang}.xml",
                        f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.tst2013.{lang}-en.{lang}.xml",
                        f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.tst2014.{lang}-en.{lang}.xml",
                        f"/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/{lang}-en/IWSLT17.TED.tst2015.{lang}-en.{lang}.xml",
                       ]
    docs_dict = build_dictionary_from_xml(xml_filepath_list, lang)
    lang_dicts.append(docs_dict)

shared_talk_ids = get_shared_talkids(lang_dicts=lang_dicts)
multilingual_talks_dict = combine_langs_into_one_dict(lang_dicts = lang_dicts, shared_talk_ids=shared_talk_ids)

with open("/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/ted_multilingual", "w", encoding="utf-8") as outfile:
    json.dump(multilingual_talks_dict, outfile, indent=4, ensure_ascii=False)