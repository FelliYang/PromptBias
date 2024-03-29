MODEL_TYPE=${1:-bert}
FULL_MODEL_NAME=${2:-bert-base-cased}
INTER_VOCAB=${3:-common_vocab_cased}
PROBE_METHOD=${4:-typed_querying}
PROMPT=${5:-LAMA}
CUDA_DEVICE=${6:-0}
DO_DEBIAS=${7:-True}
EVAL_BIAS=${8:-False}
SUPPORT_DATASET=${9:-200}
FILTER_OUT_BIASED_TOKEN=${10:-0}
ABLATION_NO_NORMALIZATION=${11:-False}
ABLATION_NO_RESCALE=${12:-False}

WORKSPACE=/mnt/code/users/xuziyang/PromptBias


# 从MODEL_NAME中提取出来合适的字段
# 原始字符串

# 使用awk提取目标片段
if [[ $FULL_MODEL_NAME == *"/"* ]]; then
    MODEL_NAME=$(echo "$FULL_MODEL_NAME" | awk -F'/' '{print $NF}')
else
    MODEL_NAME=$FULL_MODEL_NAME
fi



# 获取calibrate 环境变量
if [ -z "$CALIBRATE" ] || [ "$CALIBRATE" = "False" ]; then
    DO_CALIBRATE=False
else
    DO_CALIBRATE=True
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

if [ $PROBE_METHOD == typed_querying ]; then
    TYPED_QUERYING=True
else
    TYPED_QUERYING=False
fi


LOG_DIR=${WORKSPACE}/logs/filter_out_${FILTER_OUT_BIASED_TOKEN}_biased_tokens/${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}

if [ $DO_DEBIAS == True ]; then
    LOG_NAME=${LOG_DIR}/${PROMPT}
elif [ $EVAL_BIAS == True ]; then
    LOG_NAME=${LOG_DIR}/${PROMPT}_eval_bias_${SUPPORT_DATASET}
fi

if [ $DO_CALIBRATE == True ]; then
    LOG_NAME=${LOG_NAME}_do_calibrate
    SAVE_PATH=${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/${PROMPT}_result_do_debias.json
    # echo $LOG_NAME
else
    SAVE_PATH=${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/${PROMPT}_result.json
fi

# 如果有消融实验，需要修改logname和savepath
if [ $ABLATION_NO_NORMALIZATION == True -a $ABLATION_NO_RESCALE == True ]; then
    LOG_NAME=${LOG_NAME}/ablations/no_norm_no_rescale
    SAVE_PATH=${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/ablations/${PROMPT}/no_normal_no_rescale_result.json
elif [ $ABLATION_NO_NORMALIZATION == True ]; then
    LOG_NAME=${LOG_NAME}/ablations/no_normalization
    SAVE_PATH=${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/ablations/${PROMPT}/no_normalization_result.json
elif [ $ABLATION_NO_RESCALE == True ]; then
    LOG_NAME=${LOG_NAME}/ablations/no_rescale
    SAVE_PATH=${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/ablations/${PROMPT}/no_rescale_result.json
fi

# exit 0

parent_log_dir="$(dirname "$LOG_NAME")"
if [ ! -d "$parent_log_dir" ]; then
  mkdir -p "$parent_log_dir"
fi

cd ${WORKSPACE}/code

# echo "typed querying setting is "$TYPED_QUERYING
echo "使用显卡${CUDA_DEVICE} 执行如下命令: "
echo "python script.py \
    --model_type $MODEL_TYPE \
    --model_name $FULL_MODEL_NAME \
    --prompt ${PROMPT} \
    --do_debias $DO_DEBIAS \
    --eval_bias $EVAL_BIAS \
    --common_vocab ${INTER_VOCAB}.txt \
    --typed_querying $TYPED_QUERYING \
    --save_path $SAVE_PATH \
    --eval_output_path ${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/${PROMPT}_eval_bias_${SUPPORT_DATASET}.json \
    --do_calibrate $DO_CALIBRATE \
    --filter_biased_token_nums $FILTER_OUT_BIASED_TOKEN \
    --ablation_no_normalization $ABLATION_NO_NORMALIZATION \
    --ablation_no_rescale $ABLATION_NO_RESCALE \
    > $LOG_NAME.log 2>&1 &
"


python script.py \
    --model_type $MODEL_TYPE \
    --model_name $FULL_MODEL_NAME \
    --prompt ${PROMPT} \
    --do_debias $DO_DEBIAS \
    --eval_bias $EVAL_BIAS \
    --common_vocab ${INTER_VOCAB}.txt \
    --typed_querying $TYPED_QUERYING \
    --save_path $SAVE_PATH \
    --eval_output_path ${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/${PROMPT}_eval_bias${SUPPORT_DATASET}.json \
    --support_dataset $SUPPORT_DATASET \
    --do_calibrate $DO_CALIBRATE \
    --filter_biased_token_nums $FILTER_OUT_BIASED_TOKEN \
    --ablation_no_normalization $ABLATION_NO_NORMALIZATION \
    --ablation_no_rescale $ABLATION_NO_RESCALE \
    > $LOG_NAME.log 2>&1 &
