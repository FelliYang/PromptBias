MODEL_TYPE=${1:-bert}
MODEL_NAME=${2:-bert-base-cased}
INTER_VOCAB=${3:-common_vocab_cased}
PROBE_METHOD=${4:-typed_querying}
PROMPT=${5:-LAMA}
CUDA_DEVICE=${6:-0}
WORKSPACE=${7:-/mnt/code/users/xuziyang/PromptBias}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

if [ $PROBE_METHOD == typed_querying ]; then
    TYPED_QUERYING=True
else
    TYPED_QUERYING=False
fi

LOG_DIR=${WORKSPACE}/logs/${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}

cd ${WORKSPACE}/code
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi
# echo "typed querying setting is "$TYPED_QUERYING
echo "使用显卡${CUDA_DEVICE} 执行如下命令: "
echo "python script.py \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --prompt ${PROMPT} \
    --common_vocab ${INTER_VOCAB}.txt \
    --typed_querying $TYPED_QUERYING \
    --save_path ${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/${PROMPT}_result.json \
    > ${LOG_DIR}/${PROMPT}.log 2>&1 &"


python script.py \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --prompt ${PROMPT} \
    --common_vocab ${INTER_VOCAB}.txt \
    --typed_querying $TYPED_QUERYING \
    --save_path ${MODEL_NAME}/${INTER_VOCAB}/${PROBE_METHOD}/${PROMPT}_result.json \
    > ${LOG_DIR}/${PROMPT}.log 2>&1 &
