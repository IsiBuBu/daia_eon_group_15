import json
import pandas as pd
import random 

def build_extended_data_frame(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Flatten the data
    contexts = []
    questions = []
    ids = []
    answers = []
    answer_starts = []
    id_counter = 0
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            id = id_counter
            id_counter += 1 
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    answer_text = answer['answer_category']
                    answer_start = answer['answer_start']
                    contexts.append(context)
                    ids.append(id)
                    questions.append(question)
                    answers.append(answer_text)
                    answer_starts.append(answer_start)
    
    # Create a DataFrame
    return pd.DataFrame({
        'context': contexts,
        'context_id': ids,
        'question': questions
    })

def build_extended_data_frame_is_impossible(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Flatten the data
    contexts = []
    questions = []
    is_impossible = []
    ids = []
    id_counter = 0
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            id = id_counter
            id_counter += 1 
            for qa in paragraph['qas']:
                question = qa['question']
                is_impossible_question = qa['is_impossible']
                contexts.append(context)
                ids.append(id)
                questions.append(question)
                is_impossible.append(is_impossible_question)
    
    # Create a DataFrame
    return pd.DataFrame({
        'context': contexts,
        'context_id': ids,
        'question': questions,
        'is_impossible': is_impossible
    })

def build_extended_data_frame_answer_category(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Flatten the data
    contexts = []
    questions = []
    answer_categorys = []
    context_ids = []
    question_ids = []
    context_id_counter = 0
    question_id_counter = 0
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context_id_counter += 1 
            for qa in paragraph['qas']:
                question = qa['question']
                question_id_counter += 1
                for answer in qa['answers']:
                    answer_category = answer['answer_category']
                    contexts.append(context)
                    context_ids.append(context_id_counter)
                    questions.append(question)
                    question_ids.append(question_id_counter)
                    answer_categorys.append(answer_category)
    
    # Create a DataFrame
    return pd.DataFrame({
        'context': contexts,
        'context_id': context_ids,
        'question': questions,
        'question_id': question_ids,
        'answer_category': answer_categorys
    })


def build_simple_data_frame(file_name):
    with open(file_name, 'r') as file:
        dev1 = json.load(file)

    # Flatten the data
    contexts = []

    for article in dev1['data']:
        for paragraph in article['paragraphs']:
            contexts.append(paragraph['context'])

    # Create a DataFrame
    return pd.DataFrame({
        'context': contexts
    })  

def create_data_split_simple(file_name, share):
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    data_structure = {}
    context_id = 0

    #edit here for additional attributes from the json
    for article in data['data']:
        for paragraph in article['paragraphs']:
            data_structure[context_id] = {}
            data_structure[context_id]['context'] = paragraph['context']
            data_structure[context_id]['questions'] = {}
            question_id = 0
            for question in paragraph['qas']:
                data_structure[context_id]['questions'][question_id] = {}
                data_structure[context_id]['questions'][question_id]['question'] = question['question']
                for answer in question['answers']:
                    data_structure[context_id]['questions'][question_id]['answer'] = answer['text']
                    data_structure[context_id]['questions'][question_id]['answer_start'] = answer['answer_start']
                question_id += 1
            context_id += 1
                    
    # Randomly shuffle the contexts (data points)

    number_of_contexts = len(data_structure)
    validation_dataset_target_length = int(share * number_of_contexts)
    training_dataset_target_length = number_of_contexts - validation_dataset_target_length

    # Split the data structure based on the target lengths
    training_data = {}
    validation_data = {}
    context_id_list = list(data_structure.keys())
    random.shuffle(context_id_list)
    for i, context_id in enumerate(context_id_list):
      if i < training_dataset_target_length:
        training_data[context_id] = data_structure[context_id]
      else:
        validation_data[context_id] = data_structure[context_id]

    contexts = []
    questions = []
    answers = []
    answer_starts = []
    
    #edit here for additional attributes from the json
    for context_id in training_data.keys():
        context = training_data[context_id]['context']
        for question_id in training_data[context_id]['questions'].keys():
            questions.append(training_data[context_id]['questions'][question_id]['question'])
            answers.append(training_data[context_id]['questions'][question_id]['answer'])
            answer_starts.append(training_data[context_id]['questions'][question_id]['answer_start'])
            contexts.append(context)
    
    #edit here for additional attributes from the json
    train_df = pd.DataFrame({
        'context': contexts,
        'question': questions,
        'answer': answers,
        'answer_start': answer_starts
    })

    contexts = []
    questions = []
    answers = []
    answer_starts = []
    
    #edit here for additional attributes from the json
    for context_id in validation_data.keys():
        context = validation_data[context_id]['context']
        for question_id in validation_data[context_id]['questions'].keys():
            questions.append(validation_data[context_id]['questions'][question_id]['question'])
            answers.append(validation_data[context_id]['questions'][question_id]['answer'])
            answer_starts.append(validation_data[context_id]['questions'][question_id]['answer_start'])
            contexts.append(context)
    
    #edit here for additional attributes from the json
    val_df = pd.DataFrame({
        'context': contexts,
        'question': questions,
        'answer': answers,
        'answer_start': answer_starts
    })

    return train_df, val_df

def create_data_split_with_possible_impossible_ratio(file_name, share, ratio):
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    data_structure = {}
    context_id = 0
    number_of_datapoints = 0
    #build datastructure - to add new variables look for the same depth in the json 
    for article in data['data']:
        for paragraph in article['paragraphs']:
            data_structure[context_id] = {}
            data_structure[context_id]['context'] = paragraph['context']
            data_structure[context_id]['questions'] = {}
            question_id = 0
            for question in paragraph['qas']:
                data_structure[context_id]['questions'][question_id] = {}
                data_structure[context_id]['questions'][question_id]['question'] = question['question']
                data_structure[context_id]['questions'][question_id]['is_impossible'] = question['is_impossible']
                for answer in question['answers']:
                    data_structure[context_id]['questions'][question_id]['answer'] = answer['text']
                    data_structure[context_id]['questions'][question_id]['answer_start'] = answer['answer_start']
                question_id += 1
                number_of_datapoints += 1
            context_id += 1
                    

    validation_dataset_target_length = int(share * number_of_datapoints)
    validation_dataset_target_length_impossible = validation_dataset_target_length * ratio
    validation_dataset_target_length_possible = validation_dataset_target_length - validation_dataset_target_length_impossible  
    impossible_count = 0
    possible_count = 0

    #create lists for dataframe 
    contexts_train = []
    questions_train = []
    answers_train = []
    answer_start_train = []
    is_impossible_train = []
    contexts_val = []
    questions_val = []
    answers_val = []
    answer_start_val = []
    is_impossible_val = []

    # Randomly shuffle the contexts (data points)
    context_id_list = list(data_structure.keys())
    random.shuffle(context_id_list) 

    for i, context_id in enumerate(context_id_list):
        for j, question_id in enumerate(data_structure[context_id]['questions']):
            if data_structure[context_id]['questions'][question_id]['is_impossible']:
                if impossible_count < validation_dataset_target_length_impossible:
                    impossible_count += 1
                    contexts_val.append(data_structure[context_id]['context'])
                    questions_val.append(data_structure[context_id]['questions'][question_id]['question'])
                    answers_val.append(0)
                    answer_start_val.append(-1)
                    is_impossible_val.append(data_structure[context_id]['questions'][question_id]['is_impossible'])
                else:
                    contexts_train.append(data_structure[context_id]['context'])
                    questions_train.append(data_structure[context_id]['questions'][question_id]['question'])
                    answers_train.append(0)
                    answer_start_train.append(-1)
                    is_impossible_train.append(data_structure[context_id]['questions'][question_id]['is_impossible'])
            else:
                if possible_count < validation_dataset_target_length_possible:
                    possible_count += 1
                    contexts_val.append(data_structure[context_id]['context'])
                    questions_val.append(data_structure[context_id]['questions'][question_id]['question'])
                    answers_val.append(data_structure[context_id]['questions'][question_id]['answer'])
                    answer_start_val.append(data_structure[context_id]['questions'][question_id]['answer_start'])
                    is_impossible_val.append(data_structure[context_id]['questions'][question_id]['is_impossible'])
                else:
                    contexts_train.append(data_structure[context_id]['context'])
                    questions_train.append(data_structure[context_id]['questions'][question_id]['question'])
                    answers_train.append(data_structure[context_id]['questions'][question_id]['answer'])
                    answer_start_train.append(data_structure[context_id]['questions'][question_id]['answer_start'])
                    is_impossible_train.append(data_structure[context_id]['questions'][question_id]['is_impossible'])


    
    train_df = pd.DataFrame({
        'context': contexts_train,
        'question': questions_train,
        'answer': answers_train,
        'answer_start': answer_start_train,
        'is_impossible': is_impossible_train
    })

    
    val_df = pd.DataFrame({
        'context': contexts_val,
        'question': questions_val,
        'answer': answers_val,
        'answer_start': answer_start_val,
        'is_impossible': is_impossible_val    
    })

    return train_df, val_df
