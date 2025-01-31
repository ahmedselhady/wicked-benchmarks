import random
import pandas as pd
import numpy as np
from copy import copy
from transformers import pipeline
import torch
from pprint import pprint

random.seed(5331)
## classification pipeline:
clf_pipeline = pipeline("text-classification", "ahmedselhady/bert-base-uncased-sba-clf", device="cuda") 

def can_be_flipped(question_text:str):
    
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

    res = clf_pipeline(question_text, **tokenizer_kwargs)[0]
    return res['label'].lower() == "negative"


alphabets = [chr(i) for i in range(65,91)]

def doc2text(doc):
    
    prompt = f"Question: {doc['question']}\n"
    for text, label in zip(doc['choices']['text'], doc['choices']['label']):
        prompt += f"{label}. {text}\n"
    prompt += "Answer:"    
    return prompt

def preprocessing(dataset):
    
    def process_doc(doc):
            
        original_doc = copy(doc)
        if can_be_flipped(doc['question']):
            # randomly select an answer to hide:
            answer_to_hide = random.choice(doc['choices']['text'])        
            doc['choices']['text'].remove(answer_to_hide)
            doc['choices']['text'].append("None of the above")
            
            correct_answer = original_doc['choices']['label'].index(original_doc['answerKey'])
            if correct_answer not in doc['choices']['text']:
                correct_answer = "None of the above"
            
            correct_answer_index = doc['choices']['text'].index(correct_answer)
            doc['answerKey'] = doc['choices']['label'][correct_answer_index]
        return doc
    
    
    return dataset.map(process_doc)

