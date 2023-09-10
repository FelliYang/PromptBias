model_types=("bert" "bert" "roberta")
model_names=("bert-base-cased" "bert-large-cased" "roberta-large")
inter_vocabs=("common_vocab_cased" "common_vocab_cased" "common_vocab_cased_be_ro_al")
SUPPORT_DATASET=800

run_sh=/mnt/code/users/xuziyang/PromptBias/code/scripts/run.sh

for ((i=0; i<${#model_types[@]}; i++))
do
    # if [ $i -eq 0 ]; then
    #     continue
    # fi
    model_type=${model_types[$i]}
    model_name=${model_names[$i]}
    inter_vocab=${inter_vocabs[$i]}
    
    echo "Eval Prompt Bias: 当前type: $model_type, 当前name: $model_name, 当前词表: $inter_vocab"
    bash $run_sh $model_type $model_name $inter_vocab typed_querying LAMA $((2*i+0)) False True $SUPPORT_DATASET
    bash $run_sh $model_type $model_name $inter_vocab typed_querying LPAQA $((2*i+0)) False True $SUPPORT_DATASET
    bash $run_sh $model_type $model_name $inter_vocab typed_querying AutoPrompt $((2*i+1)) False True $SUPPORT_DATASET
done

# 