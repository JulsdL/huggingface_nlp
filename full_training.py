from termcolor import colored
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_column_name = tokenized_datasets["train"].column_names

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Train dataset column name:\n", 'green'))
print(colored(train_column_name, 'green'))
print("\n")


from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Inspect the batch
for batch in train_dataloader:
    print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
    print(colored("Batch:\n", 'green'))
    print(colored({k: v.shape for k, v in batch.items()}, 'green'))
    print("\n")
    break

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

outputs = model(**batch)

print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Outputs:\n", 'green'))
print(colored((outputs.loss, outputs.logits.shape), 'green'))
print("\n")

from accelerate import Accelerator

accelerator = Accelerator()

from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))


print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Number of training steps:\n", 'green'))
print(colored(num_training_steps, 'green'))
print("\n")

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Device:\n", 'green'))
print(colored(device, 'green'))
print("\n")


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])


print("-" * 50 + "\n" + "-" * 50 + "\n" + "-" * 50 + "\n\n")
print(colored("Evaluation Metric:\n", 'green'))
print(colored(metric.compute(), 'green'))
print("\n")
