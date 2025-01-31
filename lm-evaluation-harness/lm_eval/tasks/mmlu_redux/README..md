# Dataset Card for MMLU-Redux

<!-- Provide a quick summary of the dataset. -->

MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects. 

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

Each data point in MMLU-Redux contains seven columns:
- **question** (`str`): The original MMLU question.
- **choices** (`List[str]`): The original list of four choices associated with the question from the MMLU dataset.
- **answer** (`int`): The MMLU ground truth label in the form of an array index between 0 and 3.
- **error_type** (`str`): The annotated error_type. The values can be one of the six error types proposed in the taxonomy ("ok", "bad_question_clarity", "bad_options_clarity", "no_correct_answer", "multiple_correct_answers", "wrong_groundtruth") and "expert".
- **source** (`str`): The potential source of the question.
- **correct_answer** (`str`): In the case of "no_correct_answer" and "wrong_groundtruth", the annotators can suggest the alternative correct answer.
- **potential_reason** (`str`): A free text column for the annotators to note what they believe to have caused the error.

The question, choices, and answer columns are taken from [cais/mmlu](https://huggingface.co/datasets/cais/mmlu).

- **Dataset Repository:** https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
- **Code Repository:** https://github.com/aryopg/mmlu-redux
- **Alternative Dataset Repository:** https://zenodo.org/records/11624987
- **Paper:** https://arxiv.org/abs/2406.04127
- **Curated by:** Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile van Krieken, Pasquale Minervini
- **Language(s) (NLP):** English
- **License:** CC-BY-4.0

### Taxonomy

![image/png](https://cdn-uploads.huggingface.co/production/uploads/644f895e23d7eb05ca695054/ChI5KZPPnkRQv1olPifef.png)

We develop a hierarchical taxonomy to classify the various errors identified in MMLU into specific error types.
This figure illustrates our taxonomy for categorising MMLU errors.
We categorise errors into two primary groups: samples with errors in the clarity of the questions (Type 1, Question Assessment) and samples with errors in the ground truth answer (Type 2, Ground Truth Verification). While Type 1 only includes Bad Question Clarity, Type 2, is further divided into the more fine-grained error types.

Question Assessment (Type 1):
- **(1a) Bad Question Clarity:** The question is poorly presented in terms of various aspects, such as clarity, grammar, and sufficiency of information. For instance, referring to a previous question.
- **(1b) Bad Options Clarity:** The options are unclear, similar, or irrelevant to the question. Most errors in this category stem from incorrect parsing of the options from the original source. For example, a single option might be incorrectly split into two separate options.

Ground Truth Verification (Type 2):
- **(2a) No Correct Answer:** None of the options correctly answer the question. This error can, for example, arise when the ground-truth options are omitted to reduce the number of options from five to four.
- **(2b) Multiple Correct Answers:**  More than one option can be selected as the answer to the question. For example, the options contain a synonym of the ground truth label.
- **(2c) Wrong Ground Truth:** The correct answer differs from the ground truth provided in MMLU. This type of error occurs when the annotated label differs from the correct label, which may be caused by a mistake during manual annotation.


### Dataset Sources 

<!-- Provide the basic links for the dataset. -->

The data used to create MMLU-Redux was obtained from [cais/mmlu](https://huggingface.co/datasets/cais/mmlu), which is also utilised in the [lm-eval-harness framework](https://github.com/EleutherAI/lm-evaluation-harness).
To ensure uniformity of our results, the language model (LM) predictions used in our performance analyses were obtained from the [Holistic Evaluation of Language Models (HELM) leaderboard v1.3.0, released on May 15th, 2024](https://crfm.stanford.edu/helm/mmlu/v1.3.0/).

We selected 30 MMLU subjects.
We first chose the 20 subjects with the lowest state-of-the-art accuracy scores on the HELM leaderboard.
These subjects are College Mathematics, Virology, College Chemistry, High School Mathematics, Abstract Algebra, Global Facts, Formal Logic, High School Physics, Professional Law, Machine Learning, High School Chemistry, Econometrics, Professional Accounting, College Physics, Anatomy, College Computer Science, High School Statistics, Electrical Engineering, Public Relations, and College Medicine.
Since there were multiple subjects related to mathematics, we randomly omitted one (Abstract Algebra) and replaced it with the next worst-performing non-mathematical subject (Business Ethics).
The remaining 10 subjects were selected randomly without considering performance: Human Aging, High School Macroeconomics, Clinical Knowledge, Logical Fallacies, Philosophy, Conceptual Physics, High School US History, Miscellaneous, High School Geography, and Astronomy.

We randomly subsampled 100 questions per MMLU subject to be presented to the annotators.
The annotators are instructed to follow the introduced taxonomy by first assessing the question presentation, and then by verifying the ground truth MMLU label.
The annotators were encouraged to perform an exact match search using a search engine to find occurrences of the question and multiple-choice options from credible sources.
If the annotators found an exact match of the question-options pair, the annotators were asked to evaluate the answer provided by the source.
Regardless of whether a label was found in the source, and whether the MMLU label is the same or not, the annotators were asked to decide whether they would follow the label using their expertise.
In the cases where an exact match was not found, the annotators were asked to search for supporting evidence from trusted sources, such as government websites, textbooks, and/or other reputable organisations (*e.g., World Health Organisation (WHO)*).
In cases where the annotators are still unsure, they were asked to annotate the question with "Expert", denoting that the question requires more expertise.

MMLU-Redux comprises subsampled test splits of the aforementioned thirty MMLU subsets.

## Uses

<!-- This section describes suitable use cases for the dataset. -->
To reproduce our results or perform analyses similar to those presented in this study, the user may download the data and utilise all the columns.
MMLU-Redux contains both correct and erroneous instances, so the user should look at the value in column "error_type" to filter samples based on the specific error type.
In those cases where the error is "no_correct_answer", "multiple_correct_answers" or "wrong_groundtruth", the users may examine the suggested answer reported in the "correct_answer" column.
The user should consider that the questions and the options reported are the same as those in the MMLU dataset, and they have not been modified even when affected by bad clarity.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/644f895e23d7eb05ca695054/CXuAtMrd1odrSFhHGuIxO.png)

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@misc{gema2024mmlu,
      title={Are We Done with MMLU?}, 
      author={Aryo Pradipta Gema and Joshua Ong Jun Leang and Giwon Hong and Alessio Devoto and Alberto Carlo Maria Mancino and Rohit Saxena and Xuanli He and Yu Zhao and Xiaotang Du and Mohammad Reza Ghasemi Madani and Claire Barale and Robert McHardy and Joshua Harris and Jean Kaddour and Emile van Krieken and Pasquale Minervini},
      year={2024},
      eprint={2406.04127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


<!-- ## Glossary [optional] -->

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

<!-- [More Information Needed]

## More Information [optional]

[More Information Needed]

## Dataset Card Authors [optional]

[More Information Needed]
 -->
## Dataset Card Contact

- aryo.gema@ed.ac.uk
- p.minervini@ed.ac.uk

## Added thanks to
[@ahmedselhady](https://github.com/ahmedselhady) 
