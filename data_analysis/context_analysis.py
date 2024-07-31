import json
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import data_analysis.data_loader as data_loader

# This file allows for the analysis of the tokenized context length with the following points for the training and the dev datasets respectively:
# - Distribution of token lengths
# - Relation between too long tokenized contexts and short enough ones 
# - comparison between number of contexts 

# This analysis works for any of the datasets in question, you can replace them in dev_json and train_json
# You can also swap the tokenizer, keep in mind to adapt the max_token_length to the respective models value

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dev_json = "./datasets/SQuAD1.1/dev-v1.1.json"
train_json = "./datasets/SQuAD1.1/train-v1.1.json"

max_token_length = 525

# Create a DataFrame
df_def = data_loader.build_simple_data_frame(dev_json)
print(len(df_def['context']))

# Tokenize contexts using the loaded tokenizer
# Add a new column to the DataFrame for tokenized contexts
df_def['tokenized_context'] =  df_def['context'].apply(lambda context: tokenizer.tokenize(context))

# Now you can analyze the tokenized versions of the contexts
print(df_def['tokenized_context'][0])  # Print the first few tokenized contexts


tokenized_context_lengths = []

for context in df_def['tokenized_context']:
    tokenized_context_lengths.append(len(context))

token_length_distribution = {}
for length in tokenized_context_lengths:
  if length not in token_length_distribution:
    token_length_distribution[length] = 0
  token_length_distribution[length] += 1

print(token_length_distribution)

plt.figure(figsize=(8, 6))
plt.hist(tokenized_context_lengths, bins=20)  # Adjust the number of bins here
plt.xlabel("Token Length")
plt.ylabel("Number of Contexts")
plt.title("Distribution of Token Lengths in Contexts Dev1")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Define max_token_lengths for categories
category_labels = ["<=" + str(max_token_length), ">" + str(max_token_length)]

# Count contexts in each category
category_counts = [0, 0]
for length in tokenized_context_lengths:
  if length <= max_token_length:
    category_counts[0] += 1
  else:
    category_counts[1] += 1

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=category_labels, autopct="%1.1f%%")  # Adjust format string for percentages
plt.title("Distribution of Context Lengths (Categorized) Dev1")
plt.show()



# Create a DataFrame
df_train = data_loader.build_simple_data_frame(train_json)

df_train['tokenized_context'] =  df_train['context'].apply(lambda context: tokenizer.tokenize(context))

tokenized_context_lengths = []

for context in df_train['tokenized_context']:
    tokenized_context_lengths.append(len(context))

token_length_distribution = {}
for length in tokenized_context_lengths:
  if length not in token_length_distribution:
    token_length_distribution[length] = 0
  token_length_distribution[length] += 1

plt.figure(figsize=(8, 6))
plt.hist(tokenized_context_lengths, bins=20)  # Adjust the number of bins here
plt.xlabel("Token Length")
plt.ylabel("Number of Contexts")
plt.title("Distribution of Token Lengths in Contexts Train1")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Count contexts in each category
category_counts = [0, 0]
for length in tokenized_context_lengths:
  if length <= max_token_length:
    category_counts[0] += 1
  else:
    category_counts[1] += 1

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=category_labels, autopct="%1.1f%%")  # Adjust format string for percentages
plt.title("Distribution of Context Lengths (Categorized) Train1")
plt.show()

compare_dev_train_1 = []
compare_dev_train_1.append(len(df_def['context']))
compare_dev_train_1.append(len(df_train['context']))

category_labels = ["Dev1 " + str(len(df_def['context'])), "Train1 " + str(len(df_train['context']))]

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(compare_dev_train_1, labels=category_labels, autopct="%1.1f%%")  # Adjust format string for percentages
plt.title("Comparisons between Dev1 and Train1 number of contexts")
plt.show()

