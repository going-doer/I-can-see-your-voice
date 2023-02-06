import numpy as np
import random
import librosa
import pandas as pd
import os
import soundfile as sf
import codecs


import tensorflow
from pyannote.audio import Pipeline
from pydub import AudioSegment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import AutoConfig, Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers.file_utils import ModelOutput


from dataclasses import dataclass
from typing import Optional, Tuple


from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from wav2vec_classificationlayer import Wav2Vec2ForSpeechClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path_emotion = "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition3"
config_emotion = AutoConfig.from_pretrained(model_name_or_path_emotion)
processor_emotion = Wav2Vec2Processor.from_pretrained(model_name_or_path_emotion)
sampling_rate_emotion = processor_emotion.feature_extractor.sampling_rate
model_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path_emotion).to(device)





import moviepy.editor as mp

name=input('파일 이름을 입력하세요 (mp4제외): ')
clip = mp.VideoFileClip("./mp4_data/{}.mp4".format(name))
clip.audio.write_audiofile("./wav_data/{}.wav".format(name)) #wa파일로 변환



"""## 2. wav -> array wav파일을 array형태로 바꾸기"""

print('data wav파일 sampling시작')
speech_array, sampling_rate = librosa.core.load("./wav_data/{}.wav".format(name), sr=16000) #wav->array

#파일 길이 구하기(초)

wav_sec=len(speech_array)/sampling_rate

#모델 불러오기
from pyannote.audio import Pipeline

print('모델 불러오기시작')
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')

print('모델에 입력하기시작')


diarization = pipeline("./wav_data/{}.wav".format(name))
diarization.for_json()['content']

df=pd.DataFrame(diarization.for_json()['content'])
df['len']=df['segment'].apply(lambda x: x['end']-x['start'])

df=df[df['len']>1]

print('자르기 시작')



"""## 3. 영상 화자별로 자르기

"""

start_sec=0 #시작 
finsh_sec=wav_sec #끝

data={}

wav2_sec=finsh_sec-start_sec

per2_sec = len(speech_array)/wav2_sec


for idx, i in enumerate(df.iterrows()):

  data_tmp={}
  data_tmp['audio']=speech_array[round(per2_sec*i[1][0]['start']):round(per2_sec*i[1][0]['end'])]
  data_tmp['track']=i[1][1]
  data_tmp['label']=i[1][2]

  data['{}'.format(idx)]=data_tmp


#화자별 시간대로 자르기


wav2_sec=finsh_sec-start_sec

per2_sec = len(speech_array)/wav2_sec

wav_data_list=[]

trash=[]

for idx, i in enumerate(df['segment']) :
  if round(per2_sec*i['end'])-round(per2_sec*i['start'])<=1:
    trash.append(idx)
  else:
    globals()["y_{}".format(idx)]=speech_array[round(per2_sec*i['start']):round(per2_sec*i['end'])]
    wav_data_list.append(globals()["y_{}".format(idx)])

df['wav_array']=wav_data_list
df=df.reset_index(drop=True)


print('영상별로 잘랐음')









"""## 4. 각자 자른 wav파일  -> STT"""

#컷파일 올리기
#split_file 파일 만들어주기


file_path=[]
file_name=[] # .trn 파일 용 # 민주


for idx, i in enumerate(df.iterrows()):
    print('파일나누는중')
    sf.write('./split_file/{}_{}.wav'.format(name,idx), i[1][4], 16000, ) 
    file_path.append('./split_file/{}_{}.wav'.format(name,idx))
    file_name.append('{}_{}.wav'.format(name,idx)) # 민주
  

df['path']=file_path
df.to_csv('./split_file/file_information.csv')

# 민주
with open("./script_file/dev.trn","w", encoding="utf-8") as file:
    for line in file_name:
        file.write(f"{line}\n")

print('파일에 나눠서 저장함')





text=[]
## 민주
print()
print()
print()
print("================= 주의 =================")
print(" STT 모델을 실행시키는 과정을 시작합니다. ")
print()
print()
print()
print("1. 새로운 conda cmd 창을 관리자 권한으로 연 뒤 새로운 환경을 생성 해주세요. ")
print("STT 모델을 실행시키는 과정은 현재 생성한 conda cmd 창에 명령어를 입력해야 합니다.")
print("conda create -n totalkwav2vec python=3.7")
print("conda activate totalkwav2vec")
print("conda install git")
while True:
    isYn = input("완료하셨습니까? (y|n) ")
    if isYn.upper() == 'Y':
        break
print()
print()
print()
print("2. 환경 변수를 설정해 주세요.  ")
print("** ROOT_DIR 은 파일 구분자를 \로 설정해주세요.")
print("** ROOT_DIR_LINUX 은 파일 구분자를 /로 설정해주세요.")
print("반드시 아래의 tree처럼 경로를 설정하셔야 합니다.")
multiline='''
└─I-can-see-you
    ├─total_line_korean
    └─total_line_stt
'''
print(multiline)
print("** 아래처럼 ROOT DIR을 지정하여 명령어를 입력해주세요.")
print("** 반드시 경로를 확인해주세요. ")
print(r"set ROOT_DIR=C:\Users\windowadmin6\Documents\minju\I-can-see-you ")
print(r"set ROOT_DIR_LINUX=C:/Users/windowadmin6/Documents/minju/I-can-see-you ")
while True:
    isYn = input("완료하셨습니까? (y|n) ")
    if isYn.upper() == 'Y':
        break
print()
print()
print()
print("3. 모델을 실행시킵니다.")
print("** 아래 명령어를 입력해주세요.")
print("cd %ROOT_DIR%/total_line_stt/K-wav2vec")
print("pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html")
print("pip install https://github.com/kpu/kenlm/archive/master.zip")
print("pip install pandas")
print("pip install -r ./requirements_update.txt")
print("python setup.py develop")
print("bash script/preprocess/make_infer_script_for_mulitmodel.sh")
print("bash script/preprocess/combine_manifest_for_infer.sh")
print("bash script/inference/evaluate_multimodel_infer.sh")
while True:
    isYn = input("완료하셨습니까? (y|n) ")
    if isYn.upper() == 'Y':
        break

test_dict = {}
with open("../total_line_stt/result/kw2v-dev.csv", "r", encoding='utf-8') as file:
    while True:
        line = file.readline()
        if line == '':
            break
        line_arr = line.strip().split('\t')
        line_filepath = line_arr[0].split('\\')
        num = int(line_filepath[-1].split('.')[0].split('_')[-1])
        test_dict[num] = line_arr[1]
sorted_list = sorted(test_dict)
for idx in sorted_list:
    text.append(test_dict[idx])
# print(text)/
# input("--확인하세요--")
print("============ STT 모델 완료 ============ ")

df['text']=text





"""## 5. wav파일을 통한 감정분석"""

# 하윤
config_emotion.id2label[0] = '기쁨'
config_emotion.id2label[1] = '당황'
config_emotion.id2label[2] = '분노'
config_emotion.id2label[3] = '불안'
config_emotion.id2label[4] = '슬픔'
config_emotion.id2label[5] = '중립'

def speech_file_to_array_fn(path, sampling_rate):

    speech, _ = librosa.core.load(path, sr=16000)

    return speech


def predict(wav, sampling_rate):
    # speech = speech_file_to_array_fn(path, sampling_rate)
    features_emotion = processor_emotion(wav, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features_emotion.input_values.to(device)
    attention_mask = features_emotion.attention_mask.to(device)

    with torch.no_grad():
        logits = model_emotion(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{config_emotion.id2label[i] : round(score * 100, 3)} for i, score in enumerate(scores)]
    print(outputs)

    return outputs


# 감정 추출 후 감정 비율 높은 순으로 정렬
import operator

emotion_list=[]
trash_idx=[]

for idx,i in enumerate(df.iterrows()):
  try:
    emotion_list.append(predict(i[1][4],16000))
  except:
    trash_idx.append(idx)


for i in range(len(emotion_list)):
  emotion_sorted = [] # 감정 비율 높은 순으로 정렬된 리스트
  temp_dict = {}  # 정렬을 위한 임시 저장 딕셔너리

  for emotion in emotion_list[i]:
    print(emotion)
    for k, v in emotion.items():
      temp_dict[k] = v

  emotion_sorted = sorted(temp_dict.items(), key=operator.itemgetter(1),reverse=True)

  emotion_list[i] = emotion_sorted

# 감정 카테고리에 정렬된 값 넣어주기
df['emotion']=emotion_list


# result_name=input('결과물파일 이름을 입력해 주세요: ')

df.to_csv('{}_result.csv'.format(name))


# 데이터 프레임 가져오기
audio_df = df

# 쓸모없는 정보들 삭제
del audio_df['track']
del audio_df['len']
del audio_df['wav_array']

# 인덱스 초기화 - 기존 인덱스 순서대로 안되어있어서 맞춤
audio_df.reset_index(drop = False, inplace = True)

# 기존 인덱스 열 삭제
del audio_df['index']

# 인덱스를 기준으로 딕셔너리 생성
audio_dt = audio_df.to_dict('index')

# 4. ASS
class Assem2Ass(object):
    def __init__(self,assem_dict):
        """
        A class to convert dict_data_assem to ASS format
        """
        # Base screen size for placement calculations.
        # Everything scales according to these vs actual.
        # Font size and Shadow/Outline pixel widths apply to this screen size.
        self.width = 100
        self.height = 100
        # Subtitle events,styles are a dict
        self.events = {}
        self.styles = {}
        # Headers for each section of the ASS file
        # TODO: Add more Script Info
        # ass 자막 필수요소 [Script Info], [V4 Styles], [Events]
        self.Script_Info = "[Script Info]\n" \
        "ScriptType: V4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"\
        "YCbCr Matrix: TV.601\nPlayResX: 1920\nPlayResY: 1080\n"
        self.V4_Styles = "[V4+ Styles]\n"\
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,"\
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing,"\
        "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        self.Events = "[Events]\n"\
        "Format: Layer, Start, End, Style, Name, " \
        "MarginL, MarginR, MarginV, Effect, Text\n"
        #   self.df=pandas.DataFrame(assem_dict)
        self.dt=assem_dict
        self._parse_assem()
        self._convert_to_ass()
    
    # TODO : 감정벡터를 자막 그래픽으로 구현
    # 폰트 색상만 변경 - 구체화 필요
    def emo2vec(self,emotion):
        f_e=emotion[0][0]
        f_e_v=emotion[0][1]
       
        font_size='50'
        outline='2'
        shadow='2'
        font_name='Arial'
        fgColor='&H00FFFFFF'
        bgColor='&H00000000'
        olColor='&H00000000'
        
        if f_e=='중립':
            font_name='나눔고딕'
            fgColor='&H00FFFFFF'    # 흰색
        elif f_e=='기쁨':
            font_name='a고속도로'
            font_size='50'
            fgColor='&H00FFFFFF'    # 노랑
            bgColor='&H0006C8F9'
            olColor='&H0007A6CE'
            outline='5'
            shadow='4'
        elif f_e=='슬픔':
            font_name='a천생연분'
            fgColor='&H00FFFFFF'    # 파랑
            bgColor='&H00BA7D0F'
            olColor='&H00C7AD81'
            outline='5'
            shadow='5'
        elif f_e=='불안':
            font_name='a흑백사진'
            fgColor='&H00FFFFFF'    # 초록
            bgColor='&H007B435F'
            olColor='&H32553243'
            outline='2'
            shadow='5'
        elif f_e=='당황':
            font_name='새굴림'
            fgColor='&H00FFFFFF'    # 보라      
            bgColor='&H00000000'
            olColor='&H00000000'
            outline='2'
            shadow='2'      
        elif f_e=='분노':
            font_name='a옛날이발관L'
            font_size='55'
            fgColor='&H00FFFFFF'    # 빨강
            bgColor='&H000000FF'
            olColor='&H00000000'
            outline='5'
            shadow='5'
        # print(font_name,font_size,fgColor,bgColor,olColor,outline,shadow)    
        return font_name,font_size,fgColor,bgColor,olColor,outline,shadow

    # assem에 있는 text + emotion 데이터 파싱 후 ass 형식으로 바꾸어 저장
    def _parse_assem(self):
        """
        Convert the input assem into separate dicts for
        events (text + duration) and styles to be applied to those events.
        """
        data=self.dt
        for index_dt in range(len(data)):
                t_start=self.t_process(data[index_dt]['segment']['start'])
                t_end=self.t_process(data[index_dt]['segment']['end'])
                text=data[index_dt]['text']

                first_emotion=data[index_dt]['emotion'][0][0]
                
                font_name,font_size,fgColor,bgColor,olColor,outline,shadow=self.emo2vec(data[index_dt]['emotion'])                
                # print(font_name,font_size,fgColor,bgColor,olColor,outline,shadow)
                
                speaker_name = data[index_dt]['label']
                speaker_name_emotion = data[index_dt]['label']+'_'+first_emotion
                
                self.events.update({
                    index_dt: {"Text": self.line_shift(speaker_name+':'+text,font_size), "Start": t_start, "End": t_end,
                          "Style":speaker_name_emotion}
                })
                self.styles.update({
                    speaker_name_emotion: {'Fontname':font_name,'Fontsize':font_size,"PrimaryColour": fgColor,
                                           "BackColour": bgColor, 'Outline':outline, 'Shadow':shadow}
                })
            
                
    def line_shift(self,text,font_size):
        text_list=[]
        max_chars=int(int(font_size))
        result=''
        if int(len(text)/max_chars)>0:
            
            for i in  range(int(len(text)/max_chars)+1):
                 text_list.append(text[(i)*max_chars:(i+1)*max_chars]+'\\N')
                 # print(text_list[i])
                 result+=text_list[i]
        else:
            result=text
        print(result)
        return result


           
    # ass 파일로 변환     
    def _convert_to_ass(self):
        self._write_styles()
        self._write_events()
        # print(self.styles)
    
    def _write_styles(self):
        """
        Write out the style information to self.V4_Styles
        """
        # 변경이 불필요한 기타 정보 초기화
        misc_data = {
            # 'Fontname': 'Arial',
            # 'Fontsize': '100',
            # 'PrimaryColour':'&H00000000',
            'SecondaryColour': '&H000000FF',
            'OutlineColour': '&H00000000',
            # 'BackColour': '&H00000000',
            'Bold': '0',
            'Italic': '0',
            'Underline': '0',
            'StrikeOut': '0',
            'ScaleX': '100',
            'ScaleY': '100',
            'Spacing': '0',
            'Angle': '0',
            'BorderStyle': '1',
            # 'Outline': '2',
            # 'Shadow': '2',
            'Alignment': '2',
            'MarginL': '10',
            'MarginR': '10',
            'MarginV': '10',
            'Encoding': '1',
        }
        
        for (name, data) in self.styles.items():
            data.update(misc_data)
            line = u"Style: {Name},{Fontname},{Fontsize},{PrimaryColour}," \
            "{SecondaryColour},{OutlineColour},{BackColour},{Bold}," \
            "{Italic},{Underline},{StrikeOut},{ScaleX},{ScaleY},{Spacing},{Angle},"\
            "{BorderStyle},{Outline},{Shadow},{Alignment}," \
            "{MarginL},{MarginR},{MarginV},{Encoding}" \
            "\n".format(Name=name, **data)
            self.V4_Styles += line

    def _write_events(self):
        """
        Write out subtitle event information to self.Events
        """
        misc_data = {
            'Layer': '0',
            'Name': '',
            'MarginL': '0',
            'MarginR': '0',
            'MarginV': '0',
            'Effect': '',
        }
        for (num, data) in self.events.items():
            data.update(misc_data)
            line = u"Dialogue: {Layer},{Start},{End},{Style}," \
            "{Name},{MarginL},{MarginR},{MarginV},{Effect},{Text}" \
            "\n".format(**data)
            self.Events += line

    def save(self, filename):
        with codecs.open(filename, 'w', encoding='utf8') as f:
            f.write(self.Script_Info)
            f.write("\n")
            f.write(self.V4_Styles)
            f.write("\n")
            f.write(self.Events)
            f.write("\n")  
        
            
    def check_ass(self):
        print(self.Script_Info)
        print(self.V4_Styles)
        print(self.Events)

    def t_process(self,idx_time):
    
        time=float(idx_time)
        hour=int(time/3600)
        minute=int(time%3600/60)
        second=int(time%3600%60)
        h_s=(time%3600%60-second)*100
        h1=str(hour).zfill(2)
        m1=str(minute).zfill(2)
        s1=str(second).zfill(2)
        f1=str(int(h_s)).zfill(2)

        return '{0}:{1}:{2}.{3}'.format(h1,m1,s1,f1)



# ass 자막 저장할 파일 이름
filename = 'test1.mp4'

# 실행 코드 (audio_dt가 위에서 생성한 데이터 딕셔너리에요)
ass=Assem2Ass(audio_dt)
save_dir = './ass_data/'
ass.save(os.path.join(save_dir, filename.replace(".mp4", "") + ".ass"))
