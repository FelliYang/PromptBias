filter_biased_token_nums=$1

model_types=("bert" "bert" "roberta")
model_names=("bert-base-cased" "bert-large-cased" "roberta-large")
inter_vocabs=("common_vocab_cased" "common_vocab_cased" "common_vocab_cased_be_ro_al")
# model_types=("bert")
# model_names=("bert-base-cased")
# inter_vocabs=("common_vocab_cased")

run_sh=/mnt/code/users/xuziyang/PromptBias/code/scripts/run.sh

# export CALIBRATE=True

for ((i=0; i<${#model_types[@]}; i++))
do
    # if [ $i -eq 0 ]; then
    #     continue
    # fi
    model_type=${model_types[$i]}
    model_name=${model_names[$i]}
    inter_vocab=${inter_vocabs[$i]}
    
    echo "当前type: $model_type, 当前name: $model_name, 当前词表: $inter_vocab"
    # bash $run_sh $model_type $model_name $inter_vocab typed_querying LAMA 2 True False 0 $filter_biased_token_nums
    bash $run_sh $model_type $model_name $inter_vocab typed_querying LPAQA 1 True False 0 $filter_biased_token_nums
    # bash $run_sh $model_type $model_name $inter_vocab typed_querying AutoPrompt 2 True False 0 $filter_biased_token_nums
    # bash $run_sh $model_type $model_name $inter_vocab typed_querying optiprompt 0 True False 0 $filter_biased_token_nums
done

# 