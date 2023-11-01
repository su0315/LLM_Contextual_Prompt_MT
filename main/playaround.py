from datasets import load_dataset
from transformers import AutoTokenizer
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


# train-eval
file1_path="/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-1-1/source.txt"
file2_path = "/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/prompt+source.txt"
label_path = "/mnt/data-poseidon/sumire/thesis/running/ted/eval_mt/train-val/en-ja/Llama-2-70b-instruct-v2-usas-zs-p1-nsplit-ja-2-1/comet_binary.txt"

# Define the index where you want to split the dataset
split_index = 5507  # Replace this with your desired index

def concatenate_lines(file1_path, file2_path):
    # Read the lines from both files
    with open(file1_path, 'r') as file1:
        lines1 = file1.readlines()
    with open(file2_path, 'r') as file2:
        lines2 = file2.readlines()

    # Concatenate lines from the two files to create instances
    instances =  [ line1.strip() + line2.strip() for line1, line2 in zip(lines1, lines2)]

    return instances
# Load and concatenate the data
instances = concatenate_lines(file1_path, file2_path)

# Label load
all_labels = []
with open (label_path, 'r')as file:
    for line in file:
        score = int(line.strip())
        all_labels.append(score)


# Split the custom_dataset into training and validation
train_dataset = Dataset.from_dict({'text': instances[:split_index], 'label':all_labels[:split_index]})
validation_dataset = Dataset.from_dict({"text": instances[split_index:], 'label':all_labels[split_index:]})
print ("train", len(train_dataset), train_dataset[-1])
print("val", len(validation_dataset), validation_dataset[0])


# Shuffle Dataset
#train_dataset = train_dataset.shuffle(seed=42) ####
#validation_dataset = validation_dataset.shuffle(seed=42) ####
# Access the training and validation datasets

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#print ("sep_token:",tokenizer.sep_token, tokenizer.sep_token_id) # sep_token: [SEP] 102

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)#.select(range(3))
val_tokenized_datasets = validation_dataset.map(tokenize_function, batched=True)#.select(range(3))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#small_train_dataset = train_tokenized_datasets.select(range(10)) ##########
#small_eval_dataset = val_tokenized_datasets.select(range(10)) #######

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)


training_args = TrainingArguments(
    output_dir="test_trainer",
    do_eval= True,
    do_predict= True,
    do_train=True, 
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    metric_for_best_model = "f1"
    )

metric1 = evaluate.load("accuracy")
metric2 = evaluate.load("f1")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    
    accuracy = metric1.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric2.compute(predictions=preds, references=labels)["f1"]
    
    return {"accuracy": accuracy, "f1": f1}
 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset= val_tokenized_datasets,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
print ("Training Done")
model.eval()
eval_result = trainer.evaluate()
print (eval_result)

#preds, label_ids, metrics = trainer.predict()##############put test data
