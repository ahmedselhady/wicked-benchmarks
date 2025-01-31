#!/bin/bash
#SBATCH --job-name=mcq-bench-org
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/mcq-bench-original1.out
#SBATCH --error=.slurm/mcq-bench-original1.err


source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate 


export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

models=(
    #"Qwen/Qwen2.5-7B"
    #"Qwen/Qwen2.5-7B-Instruct"
    #"Qwen/Qwen2.5-14B"
    #"Qwen/Qwen2.5-14B-Instruct"
    #"meta-llama/Llama-3.1-8B"
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"mistralai/Mistral-7B-Instruct-v0.2"
    #"mistralai/Mistral-7B-v0.3"
    #"google/gemma-2-2b-it"
    #"google/gemma-2-9b"
    #"google/gemma-2-9b-it"
    "google/gemma-2-27b"
    #"google/gemma-2-27b-it"
    #"Qwen/Qwen2.5-72B"
    #"meta-llama/Llama-3.1-70B"
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

tasks_selected=(
    "mmlu"
    #"mmlu_redux"
    "mmlu_pro"
    "arc_challenge"
    "truthfulqa_mc1"
    #"commonsense_qa"
)

results_path=/gscratch5/users/asalem/wicked/results.mcq

for model in "${models[@]}"; do        
    for group_name in "${tasks_selected[@]}"; do

        num_fewshot=3
        python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model,parallelize=True,attn_implementation=sdpa \
            --tasks $group_name \
            --device cuda \
            --output_path ${results_path}/${model}/${group_name}_original_mc \
            --num_fewshot ${num_fewshot} \
            --log_samples
        
    done
    rm -rf /gscratch5/users/asalem/transformers_cache/models-*
done

