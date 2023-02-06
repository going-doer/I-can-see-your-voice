# Evaluate fine-tuned model

## select MANIFEST PATH used in fine-tuning(for multi model, need to use manifest supporting addtional script)
MANIFEST_PATH="C:\Users\windowadmin6\Documents\minju\kwav2vec_data\transcriptions\inference_graph\grapheme_character_spelling"

## fine-tuned checkpoint
# CHECKPOINT_PATH="C:\Users\windowadmin6\Documents\minju\kwav2vec_data\save_checkpoint\finetune\emotion2_batch8\multi_model\checkpoint_best.pt" 
# CHECKPOINT_PATH="C:\Users\windowadmin6\Documents\minju\kwav2vec_data\hj\save_checkpoint\checkpoint_best.pt" 
CHECKPOINT_PATH="C:\Users\windowadmin6\Documents\minju\kwav2vec_data\save_checkpoint\finetune\ksponspeech1\multi_model\checkpoint_best.pt" ## kspon

## single model use 'audio_pretraining', multi model use 'audio_multitraining' for task
TASK=audio_multitraining

## we only support 'beam'
DECODER=beam

## length of beam, default:100
BEAM=100

## joint decoding has contirbution weight to balance grapheme and syllable
## put between 0.0~1.0
CONTRIBUTION_WEIGHT=0.5

## SUBSET indiates evaluation set. our manifest include dev, eval_clean, eval_other
## Therefore, modele is evaluated in different subsets.
for SUBSET in eval_other; do

    ## We report our results with CSV file.
    ## Put csv path and name in EXPERIMENT_DIR
    EXPERIMENT_DIR="C:\Users\windowadmin6\Documents\minju\kwav2vec_data\experiments\infer_kspon1_ytaudio\multi_model_${SUBSET}.csv"

    ## Put path to save log during evaluation.
    RESULTS_PATH="C:\Users\windowadmin6\Documents\minju\kwav2vec_data\experiments\infer_kspon1_ytaudio"

    python inference/beam_search.py ${MANIFEST_PATH} \
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
       --experiments-dir ${EXPERIMENT_DIR}
done