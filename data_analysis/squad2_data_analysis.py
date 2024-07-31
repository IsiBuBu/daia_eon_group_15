import data_analysis.data_loader as data_loader
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# The major distinction between the SQuAD 2.0 and 1.1 is the introduction of artificial questions
# This file allows for further analysis on the following points:
# - relation between artificial and regular questions 
# - distribution of number of artificial and regular questions

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dev_json = "./datasets/SQuAD2.0/dev-v2.0.json"
train_json = "./datasets/SQuAD2.0/train-v2.0.json"

df_dev = data_loader.build_extended_data_frame_is_impossible(dev_json)
df_train = data_loader.build_extended_data_frame_is_impossible(train_json)

count_impossible_possible = [0, 0]

for is_impossible in df_dev['is_impossible']:
  if is_impossible:
    count_impossible_possible[0] += 1
  else:
    count_impossible_possible[1] += 1

category_labels = ["Impossible: " + str(count_impossible_possible[0]), "Possible: " + str(count_impossible_possible[1])]

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(count_impossible_possible, labels=category_labels, autopct="%1.1f%%")  # Adjust format string for percentages
plt.title("Relation between possible and impossible questions DEV")
plt.show()

count_impossible_possible = [0, 0]

for is_impossible in df_train['is_impossible']:
  if is_impossible:
    count_impossible_possible[0] += 1
  else:
    count_impossible_possible[1] += 1

category_labels = ["Impossible: " + str(count_impossible_possible[0]), "Possible: " + str(count_impossible_possible[1])]

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(count_impossible_possible, labels=category_labels, autopct="%1.1f%%")  # Adjust format string for percentages
plt.title("Relation between possible and impossible questions TRAIN")
plt.show()

def build_is_impossible_distribution(df):
  """Creates dictionaries for is_impossible occurrences per context ID.

  Args:
      df: A Pandas DataFrame containing 'context_id' and 'is_impossible' columns.

  Returns:
      A tuple containing two dictionaries:
          - impossible_counts: Maps context_id to the number of "is_impossible=True" occurrences.
          - possible_counts: Maps context_id to the number of "is_impossible=False" occurrences.
  """
  impossible_counts = {}
  possible_counts = {}
  for index, row in df.iterrows():
    context_id = row['context_id']
    is_impossible = row['is_impossible']

    if is_impossible:
      impossible_counts[context_id] = impossible_counts.get(context_id, 0) + 1
    else:
      possible_counts[context_id] = possible_counts.get(context_id, 0) + 1

  return impossible_counts, possible_counts

def plot_questiontype_distribution_per_context(ids_counts):
  """Creates a histogram for the distribution of "is_impossible" occurrences per context ID.

  Args:
      impossible_counts: A dictionary mapping context_id to the number of "is_(im)possible" occurrences.
  """
  # Extract counts as separate lists
  counts = list(ids_counts.values())

  # Create the histogram
  plt.figure(figsize=(8, 6))
  plt.hist(counts)  # Use counts directly for the histogram
  plt.xlabel("Number of 'is_impossible=True' Occurrences")
  plt.ylabel("Number of Contexts")
  plt.title("Distribution of '-' Occurrences per Context ID")
  plt.xticks(rotation=0)  # Keep labels horizontal
  plt.tight_layout()
  plt.show()

impossible_counts, possible_counts = build_is_impossible_distribution(df_dev)

print("Impossible counts:")
print(impossible_counts)

print("\nPossible counts:")
print(possible_counts)

plot_questiontype_distribution_per_context(impossible_counts)
plot_questiontype_distribution_per_context(possible_counts)


impossible_counts, possible_counts = build_is_impossible_distribution(df_train)

print("Impossible counts:")
print(impossible_counts)

print("\nPossible counts:")
print(possible_counts)

plot_questiontype_distribution_per_context(impossible_counts)
plot_questiontype_distribution_per_context(possible_counts)