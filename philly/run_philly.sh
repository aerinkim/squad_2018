#!/usr/bin/env bash
###############################
# train script in Philly
# by yuwfan@microsoft.com
###############################
nvidia-smi

EXTRA_ARGS=""
NAME=NONE
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -n|--name)
    NAME="$2"
    shift
    ;;
    -d|--dataDir)
    DATA_DIR="$2"
    shift
    ;;
    -m|--modelDir)
    MODEL_DIR="$2"
    shift
    ;;
    -l|--logDir)
    LOG_DIR="$2"
    shift
    ;;
    -g|--gpu)
    shift
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift # past argument or value
done


CODE_PATH=/var/storage/shared/pnrsy/yuwfan/mrc/san_mrc
export PYTHONPATH=$CODE_PATH:$PYTHONPATH

echo "Training-execute: DATA_DIR=$DATA_DIR"
echo "Training-execute: LOG_DIR=$LOG_DIR"
echo "Training-execute: MODEL_DIR=$MODEL_DIR"
echo "Training-execute: EXTRA_ARGS=$EXTRA_ARGS"

LOG_FILE=$LOG_DIR/$NAME.log
#python $CODE_PATH/train.py --data_dir $DATA_DIR --log_file $LOG_FILE --model_dir $MODEL_DIR $EXTRA_ARGS
TMP_MODEL=checkpoint
python $CODE_PATH/train.py --data_dir $DATA_DIR --log_file $LOG_FILE  --model_dir $TMP_MODEL $EXTRA_ARGS
cp -r $TMP_MODEL $MODEL_DIR
