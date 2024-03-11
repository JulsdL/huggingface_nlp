import os
import tensorflow as tf
from transformers import pipeline
from termcolor import colored

# Suppress TensorFlow C++ level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: default, 1: no INFO, 2: no WARNING, 3: no WARNING and ERROR

# Suppress TensorFlow Python level logs
tf.get_logger().setLevel('ERROR')


from transformers import BertConfig, TFBertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = TFBertModel(config)

# Print config
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Config:\n", 'green'))
print(colored(config, 'green'))
print("\n")


# Create from a pretrained model
model = TFBertModel.from_pretrained('bert-base-uncased')

model.save_pretrained('models/bert-base-uncased')


sequences = ["Hello!", "Cool.", "Nice!"]

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = tf.convert_to_tensor(encoded_sequences)

output = model(model_inputs)


print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Output:\n", 'green'))
print(colored(output, 'green'))
print("\n")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sequence)

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Tokens:\n", 'green'))
print(colored(tokens, 'green'))
print("\n")

ids = tokenizer.convert_tokens_to_ids(tokens)

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Ids:\n", 'green'))
print(colored(ids, 'green'))
print("\n")

decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Decoded string:\n", 'green'))
print(colored(decoded_string, 'green'))
print("\n")
