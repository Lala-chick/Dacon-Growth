# Dacon-Growth

# Task Description

## Leader Board

Public LB: 4.88014, Private LB: 4.94946  
전체 참가자 415명 중 5등

## Subject

본 대회는 한 쌍의 이미지를 입력 값으로 받아 작물의 생육 기간 예측 모델 개발을 목표로 합니다. 



## Data

- 학습 데이터: 3280x2464 크기의 BC(bok choy, 청경채)이미지 353장, LT(lattuce, 상추)이미지 400장 
- 테스트 데이터: 3280x2464 크기의 BC(bok choy, 청경채)이미지 139장, LT(lattuce, 상추)이미지 168장  



## Metric

- RMSE(Root Mean Square Error): 추정 값 또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다루는 지표




---



# ⚙Installation

## Basic Settings

```shell
# clone repository
$ git clone https://github.com/Lala-chick/Dacon-Growth.git

# install necessary tools
$ pip install -r requirements.txt
```



## Data Structure

```shell
# Download: https://dacon.io/competitions/official/235851/data
[open]/
├── train_dataset/ # 학습 데이터 입력 이미지
│     ├── BC/
│     └── LT/
├── test_dataset/
│     ├── BC/
│     └── LT/
└── sample_submission.csv
```



## Code Structure

```shell
[code]
├── open/ # 데이터셋 디렉토리
├── data/ # data 처리 관련 모듈 디렉토리
├── networks/ # 모델 아키텍처 관련 모듈 디렉토리
├── schedulers/ # 스케쥴러 모듈 디렉토리
├── utils/ # 유틸리티 관련 모듈 디렉토리
├── README.md
├── requirements.txt
├── train.py
└── inference.py
```



---



# 🕹Command Line Interface

생육 기간 예측 과정은 이미지 pre-resize후 학습을 하게 됩니다. 



## Train

```shell
$ python train.py
```


#### I. Swin 기반 모델 학습

- Swin 기반 Custom Model을 활용하여 학습합니다.
- ***Input***. 대회에서 주어진 학습 데이터의 input 이미지
- ***Label***. 대회에서 주어진 학습 데이터의 label

#### II. ViT 기반 모델 학습

- ViT 기반 Custom Model을 활용하여 학습합니다.
- ***Input***. 대회에서 주어진 학습 데이터의 input 이미지
- ***Label***. 대회에서 주어진 학습 데이터의 label




### Arguments

- `'seed'`: seed
- `'train_path'`: 학습 데이터셋 경로
- `'save_path'`: 모델 저장 경로
- `'batch_size'`: 학습 시 배치사이즈 크기
- `'workers'`: dataloader workers 수
- `'optimizer'`: 학습 시 사용될 optimizer
- `'model'`: 학습 시 사용될 model (swin 혹은 vit)
- `'prtrained'`: pretrain 사용여부
- `'epoch'`: 학습 기간
- `'lr'`: 학습 learning rate
- `'weight_decay'`: 학습 weight_decay
- `'do_resize'`: 학습 전 resize 여부
- `'size'`: resize시 변환 될 이미지 크기
- `'fold'`: 5-Fold에서 학습할 fold의 수



## Inference

학습된 모델을 불러와 추론을 수행합니다. 

```shell
$ python inference.py
```



### Arguments


- `'submission_path'`: sample_submission파일의 경로
- `'save_path'`: 추론 결과 저장 경로
- `'test_path'`: test이미지 경로
- `'workers'`: dataloader workers 수
- `'do_resize'`: 학습 전 resize 여부
- `'size'`: resize시 변환 될 이미지 크기
- `'tta'`: 추론 시 tta 사용 여부
- `'tta_num'`: tta 적용 횟수
- `'swin'`: swin 모델을 이용한 추론 여부
- `'swin_paths'`: 학습된 swin 모델 파일 경로
- `'vit'`: vit 모델을 이용한 추론 여부
- `'vit_paths'`: 학습된 vit 모델 파일 경로

# Best Score Config

## Train

1개의 ViT만 학습

- `'seed'`: 41
- `'batch_size'`: 8
- `'workers'`: 4
- `'optimizer'`: Adam
- `'model'`: vit
- `'prtrained'`: True
- `'epoch'`: 50
- `'lr'`: 1e-4
- `'weight_decay'`: 1e-5
- `'do_resize'`: True
- `'size'`: 224
- `'fold'`: 1

## Inference
1개의 ViT모델만 추론

- `'workers'`: 4
- `'do_resize'`: True
- `'size'`: 224
- `'tta'`: True
- `'tta_num'`: 3
- `'swin'`: False
- `'vit'`: True