#!/bin/bash
#SBATCH --job-name=mcq-bench
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/mcq-bench-hard-cot-wicked.out
#SBATCH --error=.slurm/mcq-bench-hard-cot-wicked.err


source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate 

export HF_DATASETS_CACHE="/gscratch5/users/asalem/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

models=(
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b"
    "google/gemma-2-27b-it"
    "Qwen/Qwen2.5-72B"
    "meta-llama/Llama-3.1-70B"
)

tasks_selected=(
    "mmlu__generative_cot_wickd"
)

results_path=/gscratch5/users/asalem/wicked/results

for model in "${models[@]}"; do
    for group_name in "${tasks_selected[@]}"; do

        num_fewshot=0 
        python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model,parallelize=True,attn_implementation=sdpa \
            --tasks $group_name \
            --device cuda \
            --output_path ${results_path}/${model}/${group_name} \
            --num_fewshot ${num_fewshot} \
            --log_samples

    done
    rm -rf /gscratch5/users/asalem/transformers_cache/models-*

done

