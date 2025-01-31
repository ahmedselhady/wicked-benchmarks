from transformers import pipeline
from functools import partial
import numpy as np
import random
import torch
import copy




import copy
import random
from transformers import pipeline
import torch


random.seed(5331)
## classification pipeline:
clf_pipeline = pipeline("text-classification", "ahmedselhady/bert-base-uncased-sba-clf", device="cuda") # "cuda" if torch.cuda.is_available() else "cpu")
def can_be_flipped(question_text:str):
    
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

    res = clf_pipeline(question_text, **tokenizer_kwargs)[0]
    return res['label'].lower() == "negative"


def preprocessing(dataset):
    
    def process_doc(doc):
        
        original_doc = copy.copy(doc)
        if can_be_flipped(doc['question']):
            # randomly select an answer to hide:
            answer_to_hide = random.choice(doc['choices'])        
            doc['choices'].remove(answer_to_hide)
            doc['choices'].append("None of the above")
            correct_answer = original_doc['choices'][original_doc['answer']]
            if correct_answer not in doc['choices']:
                correct_answer = "None of the above"
            
            doc['answer'] = doc['choices'].index(correct_answer)
        return doc
        
    return dataset.map(process_doc)


