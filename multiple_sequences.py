import os
import tensorflow as tf
from transformers import pipeline
from termcolor import colored

# Suppress TensorFlow C++ level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: default, 1: no INFO, 2: no WARNING, 3: no WARNING and ERROR

# Suppress TensorFlow Python level logs
tf.get_logger().setLevel('ERROR')


import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# Tokenize the sequence and prepare `input_ids` for the model
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tf.constant([ids])  # Note the additional brackets for batch dimension

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Input IDs:\n", 'green'))
print(colored(input_ids, 'green'))
print("\n")

output = model(input_ids)

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Logits:\n", 'green'))
print(colored(output.logits, 'green'))
print("\n")

batched_ids = [ids, ids]  # Batch size of 2
batched_input_ids = tf.constant(batched_ids)

output = model(batched_input_ids)

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Batched Logits:\n", 'green'))
print(colored(output.logits, 'green'))
print("\n")

padding_id = tokenizer.pad_token_id

bacthed_irregular_ids = [
    [ids, ids, ids],
    [ids, ids, padding_id]
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0] # 0 indicates the model should ignore this "padding_id" token
]

output = model(bacthed_irregular_ids, attention_mask=tf.constant(attention_mask))

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Batched Logits with attention mask:\n", 'green'))
print(colored(output.logits, 'green'))
print("\n")
