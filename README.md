# Listen Attend And Spell
### (Pytorch Implementation using AIhub data)
#### Paper: https://arxiv.org/abs/1508.01211

----
한국어 음성인식 모델이 종합된 Kospeech의 경우 다양한 모델을 지원함에 따라 코드 복잡도가 높아 단순 모델 학습에는 용의하지 않습니다. 
그래서 본 git은 Kospeech의 모델 코드를 이용하여 몇몇 버그 수정 및 단순화에 집중하였습니다.

DeepSpeech -> **LAS**-> LAS(SpecAgument) -> Transformer -> Conformer -> RnnT순으로 추가예정입니다.

코드 사용을 허락해주신 Kospeech Author 김수환 님에게 감사드립니다.

Original git (Kospeech): https://github.com/sooftware/KoSpeech

### Kospeech Split project

-  DeepSpeech2: https://github.com/switiz/deepspeech2.pytorch

### Note
 - KospoonSpeech preprocess code import

--- 
### Step 1 PreProcess (추가예정)
preprocess 과정의 경우 KospoonSpeech방식을 그대로 이용하였기 때문에 아래 git의 코드를 이용하시면 됩니다.

https://github.com/sooftware/KoSpeech/tree/latest/dataset/kspon

### Step 2 Configuration
data/config.yaml 파일의 내용을 load하여 각 코드에서 사용하고 있기 때문에 custom이 필요시 config.yaml을 변경해주어야합니다.

특히 각각의 PC마다 dataset의 위치가 다르기 때문에 해당 부분을 주요하게 변경해주시면됩니다.

- root : dataset root 디렉토리
- script_data_path : script (kospeech style) 디렉토리
- vocab_path : kosponSpeech-preprocess로 생성된 vocab 파일

``` 
#traiin dataset
root : C:/SpeechRecognitionDataset/Dataset/AI_hub
script_data_path: C:/Users/sanma/PycharmProjects/las.pytorch/data/aihub/transcripts.txt
vocab_path: C:/Users/sanma/PycharmProjects/las.pytorch/data/aihub/aihub_labels.csv

#train
batch_size: 8
epochs: 5
use_cuda: True
cer_every: 500
teacher_forcing_step: 0.005
teacher_forcing_ratio: 0.99
min_teacher_forcing_ratio: 0.7
label_smoothing: 0.1

#model_save_load
resume: False
save_every: 1
#inference
inference: False
use_val_data: True
weight_path: C:/Users/sanma/PycharmProjects/las.pytorch/weights/pretrained.pt

#root : C:/SpeechRecognitionDataset/Dataset/AI_hub
#script_data_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/aihub/transcripts.txt
#vocab_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/aihub/aihub_labels.csv
#model_save_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/runs/train.pt

#validation
validation_every: 1

#input feature
feature: logmelspectrogram
n_mels : 40
use_npy: False
split_balance : 0.1

#ENCODER
encoder_hidden_dim : 256
encoder_bidirectional: True
encoder_rnn_type: LSTM
num_encoder_layers: 3
encoder_dropout_p: 0.1
mask_conv: False

#Decoder
decoder_max_length: 150
decoder_embedding_dim: 512
decoder_hidden_dim: 512
decoder_attn_mechanism: dot
decoder_num_heads: 4
decoder_num_layers: 2
decoder_rnn_type: LSTM
decoder_dropout_p: 0.3
decoder_bidirectional: False
decoder_use_beam: False
```

### Step 3 Train

```
python train.py
```

#### Resume
config.yaml의 resume을 True로 변경하거나 --resume arg를 넣어 동작시켜 줍니다.
가장 마지막모델을 기준으로 load하여 resume이 실행됩니다.

```
python train.py --resume
```

### Step 4 Inference
[pretrained weight](https://drive.google.com/file/d/1wb5E8ViS5WKv1P8ynVXaBb7VTdWVY0iU/view?usp=sharing)

inference.py를 실행하면 되며 현재는 validation data를 이용하여 inference하도록 되어있습니다.

```
python inference.py
```
특정 파일 inference를 위해서는 아래처럼 audio_path argment를 이용하면 됩니다.
```
python inference.py --audio_path 'audio_file'
```
![img.png](img.png)

### Step 5 Result
데이터셋 : AIHUB 1000h Korean speech data corpus

PC사양 : Windows10, AMD 3600, RAM32, RTX 3080, Pytorch 1.8

소요시간 : EPOCH당 8시간

CER : 0.40 (40%)



### Reference

    @ARTICLE{2021-kospeech,
      author    = {Kim, Soohwan and Bae, Seyoung and Won, Cheolhwang},
      title     = {KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition},
      url       = {https://www.sciencedirect.com/science/article/pii/S2665963821000026},
      month     = {February},
      year      = {2021},
      publisher = {ELSEVIER},
      journal   = {SIMPAC},
      pages     = {Volume 7, 100054}
    }