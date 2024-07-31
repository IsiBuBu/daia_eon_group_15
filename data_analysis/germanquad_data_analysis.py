import data_analysis.data_loader as data_loader
import matplotlib.pyplot as plt

# While the GermanQuAD dataset also contains the is_impossible key that is also in the SQuAD 2.0 dataset, it makes no use of it - there are 0 occurences of is_impossible = true
# What distincts the GermanQuAD dataset is the answer_category attribute attached to the respective answers 
# This file analyses: 
# - the number of occurences of each answer for both datasets
# - the distribution of the number a answer type per question - this analysis only makes sense for the dev/test dataset, as the train dataset only ever has one answer per question 

dev_json = "./datasets/GermanQuAD/GermanQuAD_test_cleaned.json"
train_json = "./datasets/GermanQuAD/GermanQuAD_train_cleaned.json"

df_dev = data_loader.build_extended_data_frame_answer_category(dev_json)
df_train = data_loader.build_extended_data_frame_answer_category(train_json)

answer_categories = {}

for index, row in df_train.iterrows():
    answer_category = row['answer_category']
    if answer_category not in answer_categories:
        answer_categories[answer_category] = 1
    else:
        answer_categories[answer_category] += 1 

# Create the bar chart with answer category occurences 
plt.figure(figsize=(10, 6))
plt.bar(list(answer_categories.keys()), list(answer_categories.values()))
plt.xlabel("Answer Category")
plt.ylabel("Number of Occurrences")
plt.title("Distribution of Answer Categories in train set")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()

answer_categories = {}

for index, row in df_dev.iterrows():
    answer_category = row['answer_category']
    if answer_category not in answer_categories:
        answer_categories[answer_category] = 1
    else:
        answer_categories[answer_category] += 1 

# Create the bar chart with answer category occurences 
plt.figure(figsize=(10, 6))
plt.bar(list(answer_categories.keys()), list(answer_categories.values()))
plt.xlabel("Answer Category")
plt.ylabel("Number of Occurrences")
plt.title("Distribution of Answer Categories in dev set")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()

answer_category_per_question = {}

for index, row in df_dev.iterrows():
  question_id = row['question_id']
  answer_category = row['answer_category']
  if question_id not in answer_category_per_question:
    answer_category_per_question[question_id] = {}
  answer_category_per_question[question_id][answer_category] = answer_category_per_question[question_id].get(answer_category, 0) + 1

#print(answer_category_per_question)

answer_category_per_question_count_short = {}
answer_category_per_question_count_long = {}

def count_answer_category_per_question(dictionary, key):
    for question in answer_category_per_question.values():
        if key not in question:
            question[key] = 0
        if question[key] in dictionary:
            dictionary[question[key]] += 1
        else:
            dictionary[question[key]] = 1
        

count_answer_category_per_question(answer_category_per_question_count_short, 'SHORT')
count_answer_category_per_question(answer_category_per_question_count_long, 'LONG')

print(answer_category_per_question_count_short)
print(answer_category_per_question_count_long)

plt.figure(figsize=(8, 6))
plt.bar(answer_category_per_question_count_short.keys(), answer_category_per_question_count_short.values())
plt.xlabel("Answer Category")
plt.ylabel("Number of Occurrences")
plt.title(f"Distribution of short answers per question")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.bar(answer_category_per_question_count_long.keys(), answer_category_per_question_count_long.values())
plt.xlabel("Answer Category")
plt.ylabel("Number of Occurrences")
plt.title(f"Distribution of long answers per question")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()