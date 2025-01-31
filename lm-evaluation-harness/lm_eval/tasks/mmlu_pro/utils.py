from transformers import pipeline
from functools import partial
import numpy as np
import random
import torch
import copy


random.seed(5331)
## classification pipeline:
clf_pipeline = pipeline("text-classification", "ahmedselhady/bert-base-uncased-sba-clf", device="cpu") # "cuda" if torch.cuda.is_available() else "cpu")
def can_be_flipped(question_text:str):
    
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

    res = clf_pipeline(question_text, **tokenizer_kwargs)[0]
    return res['label'].lower() == "negative"


choices = [chr(i) for i in range(65, 91)]


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def doc2text_mc(doc):
    
    prompt = f"Question: {doc['question']}\n"
    options = doc["options"]
    
    char_idx = 0
    for i, opt in enumerate(options):
    
        if opt != "N/A":
            prompt += "{}. {}\n".format(choices[char_idx], opt)
            char_idx += 1
    
    prompt +="Answer:"
    return prompt


alphabets = choices

def doc2choice(doc):
    return alphabets[: len(doc["options"])]

def doc2tgt(doc):
    return doc['answer_index']

doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)

def process_docs(dataset, subject):
    
    def process_doc(doc):
        
        original_doc = copy.copy(doc)
        if can_be_flipped(doc['question']):
            # randomly select an answer to hide:
            answer_to_hide = random.choice(doc['options'])        
            doc['options'].remove(answer_to_hide)
            doc['options'].append("None of the above")
            correct_answer = original_doc['options'][original_doc['answer_index']]
            if correct_answer not in doc['options']:
                correct_answer = "None of the above"
            
            doc['answer_index'] = doc['options'].index(correct_answer)
            doc['answer'] = alphabets[doc['answer_index']]
        return doc
    # return dataset.filter(lambda x: x["category"] == subject)
    return dataset.map(process_doc).filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")
