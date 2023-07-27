from data.xml_to_json import generate_parallel_json

def main():
    parallel_lang_list = [["en", "ja"],["en", "ar"], ["en",  "fr"], ["en", "de"], ["en", "ko"], ["en", "zh"]]
    data_dir = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/"
    prompt_talk_id = 1548
    generate_parallel_json(parallel_lang_list, data_dir, prompt_talk_id )

if __name__ == "__main__":
    main()
    