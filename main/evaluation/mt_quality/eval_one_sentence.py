from transformers import XGLMTokenizer, XGLMForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, XGLMTokenizerFast, XGLMConfig, LlamaTokenizer, LlamaForCausalLM


#model_checkpoint="/mnt/data-artemis/sumire/hf_llama"
model_checkpoint ="/mnt/data/sumire/hf_llama"
max_new_tokens = 128 #128 
max_length = 271 #512 # 512


tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)  # ,  truncation=True, padding='max_length', max_new_tokens=250, return_tensors="pt") # padding_side = 'left',
tokenizer.add_special_tokens({"pad_token":"<pad>"})
model = LlamaForCausalLM.from_pretrained(model_checkpoint, use_auth_token=True)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

inp = tokenizer(
    "Translate English to Japanese: \n\n And they had to face the question  of what to call George Washington, their leader. => 当時直面した問題は 彼らの指導者ジョージ・ワシントンを 何と呼ぶかでした \n They didn't know. => 称号がなかったのです \n What do you call the leader of a republican country? => 共和国の指導者は何と呼ぶべきでしょう? \n And this was debated in Congress for ages and ages. => これが議会で長年に渡り議論されたのです \n The permanent ice is marked in red. =>", 
    return_tensors="pt", 
    max_length=max_length, 
    padding='max_length', 
    truncation=True
    ).input_ids

def postprocess_llama(tokenizer, output):
    decoded_out=tokenizer.batch_decode(output, skip_special_tokens=True)
    decoded_out = [item.split("\n")[0] for item in decoded_out]
    return decoded_out

print (tokenizer.batch_decode(inp, skip_special_tokens=True))
out = model.generate(inp,  max_new_tokens=max_new_tokens, do_sample=False)
print (tokenizer.batch_decode(out, skip_special_tokens=True), tokenizer.batch_decode(out[:,max_length:], skip_special_tokens=True))
post_out=postprocess_llama(tokenizer, out[:, max_length:])
#print ("out", out, "out[max_length:]", out[:, max_length:])
print (post_out)