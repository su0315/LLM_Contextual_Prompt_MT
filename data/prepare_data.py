from xml_to_json import generate_parallel_df

def main():
    parallel_lang_list = [["en", "ja"],["en", "ar"], ["en",  "fr"], ["en", "de"], ["en", "ko"], ["en", "zh"]]
    data_dir = "/home/sumire/thesis/LLM_Contextual_Prompt_MT/data/iwslt_hf/"
    generate_parallel_df(parallel_lang_list, data_dir, "train")
    generate_parallel_df(parallel_lang_list, data_dir, "test")
    generate_parallel_df(parallel_lang_list, data_dir, "val")
    
    
if __name__ == "__main__":
    main()
    