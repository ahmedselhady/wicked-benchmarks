tgt_path=$1

cd $tgt_path
dirs=(
    "arc_challenge"
    "commonsense_qa"
    "mmlu"
    "mmlu_pro"
    "mmlu_redux"
    "truthfulqa_mc1"
)

for dir in ${dirs[@]}; do
    mv ${dir}_hard_mc ${dir}_wickd_mc
done