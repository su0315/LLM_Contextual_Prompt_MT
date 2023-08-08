from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, XGLMTokenizerFast, XGLMConfig, LlamaTokenizer, LlamaForCausalLM


model_checkpoint="/mnt/data-artemis/sumire/hf_llama"
max_new_tokens = 2048 #128 
max_length = 271 #512 # 512


tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
tokenizer.add_special_tokens({"pad_token":"<pad>"})
model = LlamaForCausalLM.from_pretrained(model_checkpoint, use_auth_token=True)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

inp = tokenizer(
    "Translate English to Japanese: And they had to face the question  of what to call George Washington, their leader. => 当時直面した問題は 彼らの指導者ジョージ・ワシントンを 何と呼ぶかでした /n They didn't know. => 称号がなかったのです /n What do you call the leader of a republican country? => 共和国の指導者は何と呼ぶべきでしょう? /n  He has the most impeccable memory. =>", 
    return_tensors="pt", 
    max_length=max_length, 
    padding='max_length', 
    truncation=True
    ).input_ids
out = model.generate(inp,  max_new_tokens=max_new_tokens, do_sample=False)
print (tokenizer.batch_decode(out, skip_special_tokens=True))