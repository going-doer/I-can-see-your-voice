# Preprocess for inference dataset

# All related data must in same directory described below

# -----------------------------------
# wav 저장 폴더
# -----------------------------------
WAV_DIR="${ROOT_DIR}\total_line_korean\split_file"

# -----------------------------------
# script 폴더
# -----------------------------------
SCRIPT_PATH="${ROOT_DIR}\total_line_korean\script_file"


# Select for preprocess output
# You can choose either 'grapheme' or 'character'
# Here, character means syllable block which is korean basic character
# If you want to run multi-task model, OUTPUT_UNIT must 'grapheme' and ADD_OUTPUT_UNIT must 'character'
OUTPUT_UNIT=grapheme
ADD_OUTPUT_UNIT=character

# if length of script is over limit, it is exculded as described in https://arxiv.org/abs/2009.03092
LIMIT=200

# Ksponspeech data support dual transcriptions including phonetic and orthographic.
# Therefore, select transcription type [phonetic, spelling]
# Here, phonetic : phonetic, orthographic : spelling
#PROCESS_MODE=phonetic
PROCESS_MODE=spelling

# Put your absolute destination path
# -----------------------------------
# fine-tuned manifest for inference (temp)
# -----------------------------------
DESTINATION="${ROOT_DIR}\total_line_stt\temp_data\transcriptions"

## Run preprocess code
python preprocess/make_manifest_for_infer.py \
     --root ${WAV_DIR} \
     --output_unit ${OUTPUT_UNIT} \
     --additional_output_unit ${ADD_OUTPUT_UNIT} \
     --do_remove \
     --preprocess_mode ${PROCESS_MODE} \
     --token_limit ${LIMIT} \
     --dest ${DESTINATION}/${OUTPUT_UNIT}_${ADD_OUTPUT_UNIT}_${PROCESS_MODE} \
     --script_path ${SCRIPT_PATH}

