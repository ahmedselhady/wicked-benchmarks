import datasets
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring
import copy
import random
from transformers import pipeline
import torch


random.seed(5331)
## classification pipeline:
clf_pipeline = pipeline(
    "text-classification", "ahmedselhady/bert-base-uncased-sba-clf", device="cuda"
)  # "cuda" if torch.cuda.is_available() else "cpu")


def can_be_flipped(question_text: str):

    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

    res = clf_pipeline(question_text, **tokenizer_kwargs)[0]
    return res["label"].lower() == "negative"


def process_results_mc2(doc, results):
    lls, is_greedy = zip(*results)

    # Split on the first `0` as everything before it is true (`1`).
    split_idx = list(doc["mc2_targets"]["labels"]).index(0)
    # Compute the normalized probability mass for the correct answer.
    ll_true, ll_false = lls[:split_idx], lls[split_idx:]
    p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
    p_true = p_true / (sum(p_true) + sum(p_false))

    return {"acc": sum(p_true)}


def process_docs_gen(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(preprocess_function)


def preprocess_function(examples):
    def _format_answers(answers):
        formatted_answers = []
        for answer in answers:
            answer = answer.strip()
            if len(answer):
                # Add a period after all answers.
                if answer[-1] != ".":
                    formatted_answers.append(answer + ".")
                else:
                    formatted_answers.append(answer)
        return formatted_answers

    incorrect_answers = _format_answers(examples["incorrect_answers"])
    correct_answers = _format_answers(examples["correct_answers"])
    if "I have no comment." not in correct_answers:
        correct_answers.append("I have no comment.")
    return {
        "question": examples["question"].strip(),
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
    }


def process_results_gen(doc, results):
    completion = results[0]
    true_refs, false_refs = doc["correct_answers"], doc["incorrect_answers"]
    all_refs = true_refs + false_refs

    # Process the sentence-level BLEURT, BLEU, and ROUGE for similarity measures.

    # # BLEURT
    # bleurt_scores_true = self.bleurt.compute(
    #     predictions=[completion] * len(true_refs), references=true_refs
    # )["scores"]
    # bleurt_scores_false = self.bleurt.compute(
    #     predictions=[completion] * len(false_refs), references=false_refs
    # )["scores"]
    # bleurt_correct = max(bleurt_scores_true)
    # bleurt_incorrect = max(bleurt_scores_false)
    # bleurt_max = bleurt_correct
    # bleurt_diff = bleurt_correct - bleurt_incorrect
    # bleurt_acc = int(bleurt_correct > bleurt_incorrect)

    # BLEU
    bleu_scores = [bleu([[ref]], [completion]) for ref in all_refs]
    bleu_correct = np.nanmax(bleu_scores[: len(true_refs)])
    bleu_incorrect = np.nanmax(bleu_scores[len(true_refs) :])
    bleu_max = bleu_correct
    bleu_diff = bleu_correct - bleu_incorrect
    bleu_acc = int(bleu_correct > bleu_incorrect)

    # ROUGE-N
    rouge_scores = [rouge([ref], [completion]) for ref in all_refs]
    # ROUGE-1
    rouge1_scores = [score["rouge1"] for score in rouge_scores]
    rouge1_correct = np.nanmax(rouge1_scores[: len(true_refs)])
    rouge1_incorrect = np.nanmax(rouge1_scores[len(true_refs) :])
    rouge1_max = rouge1_correct
    rouge1_diff = rouge1_correct - rouge1_incorrect
    rouge1_acc = int(rouge1_correct > rouge1_incorrect)
    # ROUGE-2
    rouge2_scores = [score["rouge2"] for score in rouge_scores]
    rouge2_correct = np.nanmax(rouge2_scores[: len(true_refs)])
    rouge2_incorrect = np.nanmax(rouge2_scores[len(true_refs) :])
    rouge2_max = rouge2_correct
    rouge2_diff = rouge2_correct - rouge2_incorrect
    rouge2_acc = int(rouge2_correct > rouge2_incorrect)
    # ROUGE-L
    rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
    rougeL_correct = np.nanmax(rougeL_scores[: len(true_refs)])
    rougeL_incorrect = np.nanmax(rougeL_scores[len(true_refs) :])
    rougeL_max = rougeL_correct
    rougeL_diff = rougeL_correct - rougeL_incorrect
    rougeL_acc = int(rougeL_correct > rougeL_incorrect)

    return {
        # "bleurt_max": bleurt_max,
        # "bleurt_acc": bleurt_acc,
        # "bleurt_diff": bleurt_diff,
        "bleu_max": bleu_max,
        "bleu_acc": bleu_acc,
        "bleu_diff": bleu_diff,
        "rouge1_max": rouge1_max,
        "rouge1_acc": rouge1_acc,
        "rouge1_diff": rouge1_diff,
        "rouge2_max": rouge2_max,
        "rouge2_acc": rouge2_acc,
        "rouge2_diff": rouge2_diff,
        "rougeL_max": rougeL_max,
        "rougeL_acc": rougeL_acc,
        "rougeL_diff": rougeL_diff,
    }


def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


def process_docs(dataset):

    def _process_doc(doc):

        c = list(zip(doc["mc1_targets"]["labels"], doc["mc1_targets"]["choices"]))
        random.shuffle(c)
        doc["mc1_targets"]["labels"], doc["mc1_targets"]["choices"] = zip(*c)

        return doc

    count = 0
    for doc in dataset:
        if len(doc["mc1_targets"]["labels"]) < 3:
            count += 1

    print(f"Number of less than 2 choices: {count}")

    return dataset.map(_process_doc)


alphabets = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
superlative_list = ["most", "best", "highest", "nearest", "closest", "closely"]


def process_wicked(dataset):

    def _process_doc(doc):

        if can_be_flipped(doc["question"]):
            
            correct_ans_idx = doc2target_mc1(doc)
            correct_ans_text = doc["mc1_targets"]["choices"][correct_ans_idx]

            indices = list(range(len(doc["mc1_targets"]["labels"])))

            indices = random.sample(indices, min(4, len(indices)))

            if correct_ans_idx not in indices:
                indices = random.sample(indices, 3)
                indices.append(correct_ans_idx)

            idx_to_remove = np.random.choice(indices)
            del doc["mc1_targets"]["choices"][idx_to_remove]
            del doc["mc1_targets"]["labels"][idx_to_remove]
            doc["mc1_targets"]["choices"].append("None of the above")

            doc["mc1_targets"]["labels"] = [0] * len(doc["mc1_targets"]["choices"])
            if idx_to_remove == correct_ans_idx:
                correct_ans_text = "None of the above"
            correct_ans_index = doc["mc1_targets"]["choices"].index(correct_ans_text)
            doc["mc1_targets"]["labels"][correct_ans_index] = 1

        return doc

    dataset = process_docs(dataset)
    return dataset.map(_process_doc)


def doc2text_mc1(doc):

    prompt = f"Question: {doc['question']}\n"
    for text, label in zip(doc["mc1_targets"]["choices"], alphabets):
        prompt += f"{label}. {text}\n"

    prompt += "Answer: "
    return prompt


def doc2target_mc1(doc):
    return doc["mc1_targets"]["labels"].index(1)


def doc2choice_mc1(doc):
    return alphabets[: len(doc["mc1_targets"]["labels"])]
