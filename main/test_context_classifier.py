from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import re
from datasets import Dataset
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from functools import partial
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
"""
def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for evaluating models.")
    parser.add_argument("--cfg", action=ActionConfigFile)
    #parser.add_argument("--generic.summarized_contexts", type=str, required=False, default = None,  help="boolean")
    parser.add_argument("--generic.tgt_lang", required=True, help="target language")
    #parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    parser.add_argument("--generic.data_path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.do_train", type=bool, required=False, default =False,  help="if you do evaluate train data")
    parser.add_argument("--generic.do_train", type=bool, required=False, default =False,  help="if you do evaluate train data")
    parser.add_argument("--generic.src_context", default="src", help="the number of the target context sentence for each input")
    #parser.add_argument("--generic.dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    #parser.add_argument("--generic.speaker", type=bool, default=False)
    #parser.add_argument("--generic.random_context", type=bool, default=False)
    #parser.add_argument("--generic.tag", type=bool, default=False)
    #parser.add_argument("--generic.output_dir", required=True, metavar="DIRECTORY", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.batch_size", type=int, default=0, help="the batch size of evaluation")
    parser.add_argument("--generic.model_checkpoint", required=True, metavar="FILE", help="model_checkpoint")
    parser.add_argument("--generic.k", type=int, default=0, help="the number of few shot")
    parser.add_argument("--generic.prompt_type", type=int, default=1, help="the type of the prompt")
    parser.add_argument("--generic.max_new_tokens", type=int, default=0, help="max_new_tokens")
    parser.add_argument("--generic.max_length", type=int, default=0, help="max_length for input and labels")
    parser.add_argument("--generic.cfg_name", required=True, metavar="FILE", help="config file name")
    parser.add_argument("--generic.api", type=bool, default=False, metavar="FILE", help="Whether using text generation api or not")
    parser.add_argument('--generic.metrics',  type=str, help = "Comma-separated list of strings", default= "sacrebleu,comet", required=False)
    return parser
"""


def concatenate_lines(context_path, src_path, sep_token):
    # Read the lines from both files
    with open(context_path, 'r') as file1:
        lines1 = file1.readlines()
    with open(src_path, 'r') as file2:
        lines2 = file2.readlines()

    # Concatenate lines from the two files to create instances
    instances =  [ line1.strip() + line2.strip() for line1, line2 in zip(lines1, lines2)]

    return instances

def read_label(label_path):
    # Label load
    all_labels = []
    with open (label_path, 'r')as file:
        for line in file:
            score = int(line.strip())
            all_labels.append(score)
    return all_labels

def split_data(split_index, source, target):
    # Split the custom_dataset into training and validation
    train_dataset = Dataset.from_dict({'text': instances[:split_index], 'label':all_labels[:split_index]})
    validation_dataset = Dataset.from_dict({"text": instances[split_index:], 'label':all_labels[split_index:]})
    print ("train", len(train_dataset), train_dataset[-1])
    print("val", len(validation_dataset), validation_dataset[0])
    return train_dataset, validation_dataset

def shuffle_data(train, val):
    # Shuffle Dataset
    train_dataset = train_dataset.shuffle(seed=42) ####
    validation_dataset = validation_dataset.shuffle(seed=42) ####
    # Access the training and validation datasets
    return train_dataset, validation_dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

metric1 = evaluate.load("accuracy")
metric2 = evaluate.load("f1")

def compute_metrics(eval_pred):
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("f1")
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    
    accuracy = metric1.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric2.compute(predictions=preds, references=labels)["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

def main():
    """
    parser = read_arguments()
    cfg = parser.parse_args()
    summarized_contexts = cfg.generic.summarized_contexts
    tgt_lang = cfg.generic.tgt_lang
    data_path = cfg.generic.data_path
    src_context_size = cfg.generic.src_context
    model_checkpoint = cfg.generic.model_checkpoint
    batch_size = cfg.generic.batch_size
    k = cfg.generic.k
    prompt_type =  cfg.generic.prompt_type
    max_new_tokens = cfg.generic.max_new_tokens
    max_length = cfg.generic.max_length
    cfg_name = cfg.generic.cfg_name
    api = cfg.generic.api
    do_train = cfg.generic.do_train
    metrics = cfg.generic.metrics.split(",")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Initialize Model
    model, tokenizer = initialize_model(model_checkpoint, api)

    # Load Dataset
    few_shots, sources, inputs, labels, output_dir  = read_data(
        data_path, 
        tgt_lang, 
        api,
        model_checkpoint, 
        src_context_size,
        k, 
        prompt_type, 
        max_length, 
        tokenizer,
        summarized_contexts,
        do_train,
        cfg_name
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store Hyperparameter in text file
    with open(output_dir+'/config','w', encoding='utf8') as wf:
        for i in [
            f"tgt_lang: {tgt_lang}", 
            f"data_path: {data_path}", 
            f"src_context_size: {src_context_size}",  
            f"api: {api}",
            f"model_checkpoint: {model_checkpoint}", 
            f"batch_size: {batch_size}", 
            f"k: {k}",
            f"prompt_type: {prompt_type}", 
            f"max_new_tokens: {max_new_tokens}", 
            f"max_length: {max_length}", 
            f"cfg_name: {cfg_name}",
            f"prompt+source:\n {few_shots}",
            ]:
            wf.write(f"{i}\n")

    # Generate and Evaluate
    evaluate_mt(
        api,
        model_checkpoint, 
        model, 
        tokenizer, 
        batch_size, 
        max_new_tokens, 
        max_length, 
        device, 
        tgt_lang,
        sources,
        inputs, 
        labels, 
        output_dir,
        prompt_type,
        do_train,
        metrics
        )
    print ("Evaluation Successful")

if __name__ == "__main__":
    main()

# train-eval
file1_path="/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-1-1/source.txt"
file2_path = "/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/prompt+source.txt"
label_path = "/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/comet_binary.txt"
model_checkpoint = "bert-base-uncased" #"bert-base-cased" # "bert-large-cased"

"""
# Test 
file1_path="/mnt/data-poseidon/sumire/thesis/2-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/source.txt"
file2_path = "/mnt/data-poseidon/sumire/thesis/2-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/prompt+source.txt"
label_path = "/mnt/data-poseidon/sumire/thesis/2-1/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/comet_binary.txt"
model_checkpoint = "/mnt/data-poseidon/sumire/thesis/running/classifier/ja/2-1/test/uncased-3e-5_ep5/" #"bert-large-uncased"#"bert-base-cased" # "bert-large-cased"
best_model_checkpoint = "checkpoint-5"

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint+best_model_checkpoint, num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
sep_token = tokenizer.sep_token
print ("sep_token:",sep_token, tokenizer.sep_token_id) # sep_token: [SEP] 102

# Define the index where you want to split the dataset
split_index = 5507  # Replace this with your desired index

# Load and concatenate the data
instances = concatenate_lines(file1_path, file2_path, sep_token)
all_labels = read_label(label_path)
#train_dataset, validation_dataset = split_data(split_index, instances, all_labels)
#train_dataset, validation_dataset = shuffle_data(train_dataset, validation_dataset)

#train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True).select(range(10))
#val_tokenized_datasets = validation_dataset.map(tokenize_function, batched=True).select(range(10))
test_dataset = Dataset.from_dict({"text": instances, 'label':all_labels})
test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True).select(range(10))

output_dir = "/mnt/data-poseidon/sumire/thesis/running/classifier/ja/2-1/test/uncased-3e-5_ep5"
# Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    do_eval= True,
    do_predict= True,
    do_train=True, 
    num_train_epochs=5,#5#10
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch", # "no" was bad with load best model
    save_total_limit = 2,
    load_best_model_at_end=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    metric_for_best_model = "f1",
    learning_rate = 3e-5 #4e-5, #5e-5, 2e-5, 3e-5
    )

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=train_tokenized_datasets,
    #eval_dataset= val_tokenized_datasets,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

"""
# Train
trainer.train()
tokenizer.save_pretrained(output_dir)
print ("Training Done")

# Eval
model.eval()
eval_result = trainer.evaluate()
print (eval_result)

"""

#test
test_result = trainer.predict(test_tokenized_datasets)##############put test data
print (test_result)

with open(output_dir+"model_type.txt", "w", encoding="utf-8") as wf:
    wf.write(f"{model_checkpoint}")

with open(output_dir+"test_score.txt", "w", encoding="utf-8") as wf:
    wf.write(f"{test_result[-1]}")

with open(output_dir+"prediction.txt", "w", encoding="utf-8") as wf:
    wf.write(f"{test_result[-0]}")
