# Evaluate fine-tuned model

## fine-tuned for inference manifest 로 설정
MANIFEST_PATH="${ROOT_DIR}\total_line_stt\data\transcriptions\grapheme_character_spelling_infer"

## fine-tuned checkpoint
CHECKPOINT_PATH="${ROOT_DIR}\total_line_stt\data\save_checkpoint\finetune\checkpoint_best.pt"  ## it is dummy, please modify it

## single model use 'audio_pretraining', multi model use 'audio_multitraining' for task
TASK=audio_multitraining

## we only support 'beam'
DECODER=beam

## length of beam, default:100
BEAM=100

## joint decoding has contirbution weight to balance grapheme and syllable
## put between 0.0~1.0
CONTRIBUTION_WEIGHT=0.5

## w2v_path: pretrained w2v_path
## data: fintuned manifest path
MODEL_OVERRIDES="{'model':{'w2v_path':'${ROOT_DIR_LINUX}/total_line_stt/data/save_checkpoint/pretrain/checkpoint_best.pt','data':'${ROOT_DIR_LINUX}/total_line_stt/data/transcriptions/grapheme_character_spelling'}}"

## SUBSET indiates evaluation set. our manifest include dev, eval_clean, eval_other
## Therefore, modele is evaluated in different subsets.
for SUBSET in dev; do

    ## We report our results with CSV file.
    ## Put csv path and name in EXPERIMENT_DIR
    ## 사용하지 않음.
    EXPERIMENT_DIR="${ROOT_DIR}\total_line_stt\result\experiments_${SUBSET}.csv"

    ## Put path to save log during evaluation.
    ## 예측값이 저장될 경로
    RESULTS_PATH="${ROOT_DIR}\total_line_stt\result"

    python inference/hypo.py ${MANIFEST_PATH} \
       --task ${TASK} \
       --checkpoint-path ${CHECKPOINT_PATH} \
       --gen-subset ${SUBSET} \
       --results-path ${RESULTS_PATH} \
       --decoder ${DECODER} \
       --criterion multi_ctc \
       --labels ltr \
       --post-process letter \
       --beam=${BEAM} \
       --add-weight=${CONTRIBUTION_WEIGHT} \
       --experiments-dir ${EXPERIMENT_DIR} \
       --model_overrides ${MODEL_OVERRIDES}
done