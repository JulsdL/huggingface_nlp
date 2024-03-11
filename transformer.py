import os
import tensorflow as tf
from transformers import pipeline
from termcolor import colored

# Suppress TensorFlow C++ level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: default, 1: no INFO, 2: no WARNING, 3: no WARNING and ERROR

# Suppress TensorFlow Python level logs
tf.get_logger().setLevel('ERROR')

# Initialize your pipeline
classifier = pipeline('sentiment-analysis')

# Example usage of the classifier
result = classifier('We are very happy to show you the HuggingFace Transformers library.')

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("\n")
print(colored("Positive/Negative Classifier result:\n", 'green'))
print(colored(result, 'green'))
print("\n")

# Multi sentence example
print("*" * 50)
result = classifier([
    'We are very happy to show you the HuggingFace Transformers library.',
    'I hate this so much!.'
])
print("\n")
print(colored("Positive/Negative Classifier result for multi sentence example:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Zero shot classifier
print("*" * 50)
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print("\n")
print(colored("Zero shot classifier result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Text generation
print("*" * 50)
text_generation_model = "distilgpt2"
generator = pipeline('text-generation', model=text_generation_model)
result = generator('In this course, we will teach you how to', max_length=30, do_sample=False)
print("\n")
print(colored("Text generation result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Mask filling
print("*" * 50)
unmasker = pipeline('fill-mask')
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print("\n")
print(colored("Mask filling result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Named entity recognition (NER)
print("*" * 50)
ner = pipeline('ner', grouped_entities=True)
result = ner("Hugging Face is a company based in New York City.")
print("\n")
print(colored("Named entity recognition result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Question answering
print("*" * 50)
question_answerer = pipeline('question-answering')
result = question_answerer(
    question="Where is Hugging Face based?",
    context="Hugging Face is based in New York City."
)
print("\n")
print(colored("Question answering result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Summarization
print("*" * 50)
summarizer = pipeline('summarization')
result = summarizer("""
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
""")
print("\n")
print(colored("Summarization result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)

# Translation
print("*" * 50)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
print("\n")
print(colored("Translation result:\n", 'green'))
print(colored(result, 'green'))
print("\n")
print("*" * 50)
