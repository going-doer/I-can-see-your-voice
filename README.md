# 너의 목소리가 보여: 감정 자막 생성기
## 2022 데이터청년캠퍼스 고려대학교 과정 12조
### 팀 소개
- 김영걸
- 박하윤
- 서민주
- 이하진
- 정종호

### 개발 배경
우리는 생활 속에서 다양한 콘텐츠를 소비하고 있습니다.
그러나, 만약 소리를 사용할 수 없다면 영상의 분위기와 효과를 어떻게 느낄 수 있을까요?

때때로 우리는 불가피하게 공공장소, 강의실과 같은 곳에서 소리를 사용할 수 없는 상황에 처하게 됩니다.
혹은, 장애 혹은 노화로 인해 소리를 활용하기 힘들어 영상의 자동 자막을 사용하기도 합니다.

아쉽게도 하나의 폰트와 색상으로 구성된 자막에서는 영상의 소리에서 느낄 수 있는 감정을 느끼기가 어렵습니다.
그래서 저희는 이러한 불편함을 해결하고자 '감정 자막 자동 생성기인 너의 목소리가 보여' 서비스를 제공합니다.

### 서비스 소개
- 한국어/ 영어 2가지의 언어를 지원
![](https://github.com/jungjongho/I-can-see-your-voice/blob/main/images/diagram.PNG)

- 한국어 예시 영상
    - https://youtu.be/Qui-eum4lio
    - https://youtu.be/U_Uwyi6TVFM
- 영어 예시 영상
    - https://youtu.be/NJUKLWHT6xk

#### 기대 효과
- 청각 장애인이 영상을 볼 때 분위기 그리고 감정/감성의 이해도가 향상 됩니다.
- 소리를 켤 수 없는 공공장소에서 영상을 시청하는 사용자에게 편의를 제공합니다.
- 자막을 자동으로 생성하고 군집화함에 따라 서비스 제공 시 초보 편집자들 혹은 개인 유튜버의 노동력 감소 효과를 제공합니다.
- 유튜브 같은 경우에도 자동생성의 경우 단색으로밖에 표현되지 않아 화자가 누구인지 불편함을 겪을 때가 있는데, 이를 해결함과 동시에 자막에 감성을 담아 시청자가 영상을 보기에 편안함을 느낄 것 입니다.
- 유튜브의 어린이 동화의 경우 캐릭터들의 대사에 자막을 달게 된다면 어린이의 이해도가 향상되는 것을 기대할 수 있습니다.


### 사용방법
- Windows Server 2019 기준으로 작성되었습니다.
- 만약, checkpoint load 시에 오류가 발생했을 때는 아래 링크에서 checkpoint을 다운로드 받아 교체해주시기 바랍니다.
- https://drive.google.com/file/d/1KQq_MzlQZMRo9HdNsEP2KNF0JXpo_yKK/view?usp=sharing
    - 경로: ${ROOT_DIR}\total_line_stt\data\save_checkpoint

0. 해당 repository를 다운로드합니다.
```bash
git clone https://github.com/jungjongho/I-can-see-your-voice.git
```

1. conda의 가상 환경을 생성 합니다. (관리자 권한으로 실행시켜주세요.)
```bash
conda create -n final python=3.7
conda activate final
```
2. 라이브러리 설치를 진행합니다.
```bash
cd I-can-see-you\total_line_korean
pip install -r requirements.txt
```

3. 자막 생성을 원하는 mp4 영상 파일을 아래 위치에 저장합니다.
```
I-can-see-you\total_line_korean\mp4_data
```

4-1. 한국어 감정 자막 생성을 위한 파일을 실행합니다.
(주의사항: 반드시 안내에 따라 모델을 실행하시길 바랍니다.)
```bash
python total_line_korean.py
```

4-2. 영어 감정 자막 생성을 위한 파일을 실행합니다.
(주의사항: 반드시 안내에 따라 모델을 실행하시길 바랍니다.)
```bash
python total_line_en.py
```

5. 아래 경로에서 만들어진 ass 자막을 확인하시면 됩니다.
```bash
I-can-see-you\total_line_korean\ass_data\test1.ass
```

### 폴더 구조 및 설명
```bash
├─images # README 이미지 저장 폴더
├─total_line_korean  # 한국어 STT 모델을 제외한 모든 모델 
│  ├─ass_data
│  ├─mp4_data
│  ├─script_file
│  ├─split_file
│  └─wav_data
└─total_line_stt # 한국어 STT 모델
    ├─data 
    │  ├─save_checkpoint 
    │  │  ├─finetune
    │  │  └─pretrain
    │  └─transcriptions
    │      └─grapheme_character_spelling
    ├─K-wav2vec # K-wav2vec 실행을 위한 폴더
    │  ├─configs
    │  ├─experiments
    │  ├─fairseq
    │  ├─fairseq_cli
    │  ├─inference
    │  ├─preprocess
    │  └─script
    ├─result # 결과값이 저장되는 폴더
    └─temp_data 
        └─transcriptions
            └─grapheme_character_spelling
```

### Dataset
- ai-hub 한국어 음성(ksponspeech)
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123
    - 데이터 구성: 한국어 대화음성, 라벨링 데이터
    - 데이터 양: 2,000여명이 발성한 한국어 대화음성 1,000시간

- ai-hub 감성대화 말뭉치
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
    - 데이터 구성: 남성 음성데이터, 여성 음성데이터
    - 데이터 양: 10,000문장, 약 16시간

- ai-hub 감성 및 발화 스타일별 음성합성
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=466
    - 데이터 구성: 감성 + 감정X스타일 데이터로 구성
    - 감정데이터: (기쁨, 슬픔, 분노, 불안, 상처, 당황, 중립) 7가지로 구성
    - 감정X스타일 데이터: (중립제외 6X2(구연체,대화체)) 총 12가지로 구성
    - 사용 데이터 선별 기준
        - k-wav2vec 학습을 위해 약 30시간의 감정X발화스타일데이터 추출 
        - 감정 분류학습을 위해 감정데이터만 사용하는 약 40시간 데이터셋과 감정X스타일을 포함한 약 90시간 데이터셋 생성
        - 인간적의 보편적인 감정 분노, 혐오, 두려움, 행복, 슬픔, 놀람중 상처를 포함하는 경우는 없어 상처는 제외하였으며 당황이라는 대분류안에 소분류로 놀람과 혐오가 있어서 당황을 disgust로 해석하여 학습시킴. (기쁨,슬픔,분노,불안,당황,중립) 6가지 분류로 학습

- ai-hub 한국인 대화 음성 데이터
    - https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=130
    - 데이터 구성: 8가지 대주제로 분류되어있음
    - 한국인 대화 음성 데이터
    - 데이터 양: 4000시간
    - 사용 데이터 선별 기준
        - 컴퓨팅 파워를 고려 약 100시간의 데이터 선별
        - 하위 카테고리 중 일상안부 데이터 선택
	    - 성별: 남/여 비율을 맞춤
        - 연령: 일반 성인 선택
        - 방언: 서울/경기 선택
        - 음질: 정상 음질 선택
        - 대략 132시간의 데이터 확보 (남성 발화: 68시간, 여성 발화: 64시간)

### 참고 내용
- K-wav2vec: https://github.com/JoungheeKim/K-wav2vec
