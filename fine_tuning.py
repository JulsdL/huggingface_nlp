import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from termcolor import colored

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")

print(colored("Raw Datasets:\n", 'green'))
print(colored(raw_datasets, 'green'))
print("\n")


raw_train_dataset = raw_datasets["train"]

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")

print(colored("Raw Train Dataset:\n", 'green'))
print(colored(raw_train_dataset, 'green'))
print("\n")

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")

print(colored("Raw Train Dataset Features:\n", 'green'))
print(colored(raw_train_dataset.features, 'green'))
print("\n")

# Print element 15 of the training set
print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")

print(colored("Element 15 of the training set:\n", 'green'))
print(colored(raw_train_dataset[15], 'green'))
print("\n")

# Print element 87 of the validation set
raw_validation_dataset = raw_datasets["validation"]
print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")

print(colored("Element 87 of the validation set:\n", 'green'))
print(colored(raw_validation_dataset[87], 'green'))
print("\n")


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")

print(colored("Tokenized Datasets:\n", 'green'))
print(colored(tokenized_datasets, 'green'))
print("\n")


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

entry_lenght = [len(x) for x in samples["input_ids"]]

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Lenghts of each entry in the batch:\n", 'green'))
print(colored(entry_lenght, 'green'))
print("\n")

batch = data_collator(samples)

batch_detail = {k: v.shape for k, v in batch.items()}

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Batch details:\n", 'green'))
print(colored(batch_detail, 'green'))
print("\n")


from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

metric = evaluate.load("glue", "mrpc")
evaluation_metric = metric.compute(predictions=preds, references=predictions.label_ids)
print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Evaluation Metric:\n", 'green'))
print(colored(evaluation_metric, 'green'))
print("\n")

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
