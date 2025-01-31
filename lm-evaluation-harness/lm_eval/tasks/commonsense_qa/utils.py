import copy
import random
from transformers import pipeline
import torch
from pprint import pprint

random.seed(5331)
## classification pipeline:
clf_pipeline = pipeline("text-classification", "ahmedselhady/bert-base-uncased-sba-clf", device="cpu") # "cuda" if torch.cuda.is_available() else "cpu")
def can_be_flipped(question_text:str):
    
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

    res = clf_pipeline(question_text, **tokenizer_kwargs)[0]
    return res['label'].lower() == "negative"


def preprocessing(dataset):
    
    def process_doc(doc):
    
        original_doc = copy.copy(doc)
        if can_be_flipped(doc['question']):

            # randomly select an answer to hide:
            answer_to_hide = random.choice(doc['choices']['text'])      
            answer_key = ord(original_doc['answerKey']) - 65
              
            doc['choices']['text'].remove(answer_to_hide)
            doc['choices']['text'].append("None of the above")
            correct_answer = original_doc['choices']['text'][answer_key]
            if correct_answer not in doc['choices']['text']:
                correct_answer = "None of the above"

            doc['answerKey'] = chr(doc['choices']['text'].index(correct_answer)+65)
        return doc
    
    mapped_ds =  dataset.map(process_doc)
    return mapped_ds
    