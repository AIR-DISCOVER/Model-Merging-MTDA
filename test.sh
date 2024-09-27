CONFIG_FILE=${1:-"configs/hrda/gtaHR2csHR_hrda.py"} # val on cs
CHECKPOINT_FILE=${2:-"LinearMerge_csidd_r101.pth"}
TEST_ROOT="work_dirs"
EVAL_MODEL="teacher"
NORM_SELECT=0
SHOW_DIR="${TEST_ROOT}/preds/"
DATE_TIME=$(date +"%Y%m%d_%H%M%S")
SAVE_FILE=${3:-"${TEST_ROOT}/preds/preds_${DATE_TIME}.log"}

echo 'Config File:' $CONFIG_FILE | tee ${SAVE_FILE}
echo 'Checkpoint File:' $CHECKPOINT_FILE | tee -a ${SAVE_FILE}

python3 -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--eval mIoU --eval_model ${EVAL_MODEL} --show-dir ${SHOW_DIR} \
--opacity 0.5 --norm_select ${NORM_SELECT} \
| tee -a ${SAVE_FILE}