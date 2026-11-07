# PyTorch Learning

PyTorch를 사용해 **데이터 준비부터 모델 학습, 평가, 저장, 추론까지** 하나의 머신러닝·딥러닝 시스템을 단계적으로 구현하는 프로젝트입니다.

선형회귀를 시작으로 이진분류, 다중분류, MLP, CNN, 사전 학습 모델까지 확장합니다. 모델마다 학습 프로그램을 새로 작성하는 대신, 공통 학습 구조를 재사용하면서 필요한 구성 요소를 추가합니다.

## 프로젝트 구성

| PART | 영역                | 구현 범위                                                                 |
| ---- | ------------------- | ------------------------------------------------------------------------- |
| 1    | PyTorch 공통 구조   | 데이터 준비, Dataset, DataLoader, 모델, 손실함수, 학습 루프, 평가 및 통합 |
| 2    | 모델 구조 발전      | 선형회귀, 이진분류, 다중분류, MLP, CNN                                    |
| 3    | 학습 성능 개선      | Optimizer, 학습률, 과적합, 정규화, 데이터 증강, Early Stopping            |
| 4    | 사전 학습 모델 활용 | 전이학습, 분류기 교체, 동결, 미세조정                                     |
| 5    | 실험 및 평가 시스템 | 평가 지표, 시각화, 모델 저장, 재현성, 교차검증                            |
| 6    | 통합 미니 프로젝트  | 이미지 분류 학습 파이프라인, 모델 비교, 이미지 추론                       |

**총 6개 PART, 36개 구현 주제**로 구성됩니다.

## 전체 학습 시스템

```text
데이터 준비
    ↓
Dataset → DataLoader
    ↓
Model → 예측
    ↓
Loss → 오차 계산
    ↓
Backward → 기울기 계산
    ↓
Optimizer → 파라미터 갱신
    ↓
Evaluation → 성능 확인
    ↓
Checkpoint → 모델 저장 및 불러오기
    ↓
Inference → 새로운 데이터 예측
```

학습 단계에서는 데이터가 모델로 전달되고, 예측 결과와 정답의 차이를 바탕으로 모델 파라미터를 갱신합니다. 평가 및 추론 단계에서는 학습된 모델을 사용해 결과를 확인합니다.

## 프로젝트 디렉터리

아래는 전체 구현을 위한 **목표 디렉터리 구조**입니다. 구현 단계에 따라 파일이 추가됩니다.

```text
pytorch_learning/
├── configs/
│   └── config.py
├── data/
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── linear_model.py
│   ├── mlp.py
│   └── cnn.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── optimizer.py
├── utils/
│   ├── metrics.py
│   └── checkpoint.py
├── experiments/
├── main.py
└── predict.py
```

| 디렉터리/파일  | 역할                            |
| -------------- | ------------------------------- |
| `configs/`     | 학습 및 실험 설정 관리          |
| `data/`        | Dataset과 데이터 변환 처리      |
| `models/`      | 모델 구조 정의                  |
| `training/`    | 학습, 평가, Optimizer 관련 기능 |
| `utils/`       | 성능 지표와 체크포인트 기능     |
| `experiments/` | 실험 결과 및 비교 자료 관리     |
| `main.py`      | 학습 프로그램 실행 진입점       |
| `predict.py`   | 학습된 모델을 이용한 추론       |

## PART 1 — PyTorch 공통 구조 만들기

모델 종류에 관계없이 반복해서 사용하는 기본 학습 시스템을 구성합니다. 최종적으로 데이터를 준비하고, 클래스 기반 선형회귀 모델을 학습시킨 뒤 예측 결과를 확인합니다.

| 번호 | 구현 주제                 | 주요 코드·개념                        |
| ---- | ------------------------- | ------------------------------------- |
| 01   | 데이터 생성 및 준비       | `torch.tensor`, `dtype`, `shape`      |
| 02   | Dataset 구현              | `TensorDataset`, `Dataset`            |
| 03   | DataLoader 구현           | `DataLoader`, `batch_size`, `shuffle` |
| 04   | 모델 클래스 구현          | `nn.Module`, `__init__`, `forward`    |
| 05   | 손실함수와 Optimizer 설정 | `nn.MSELoss`, `torch.optim`           |
| 06   | 학습 루프 구현            | `zero_grad`, `backward`, `step`       |
| 07   | 평가 및 예측 구현         | `eval`, `no_grad`, `inference_mode`   |
| 08   | 전체 학습 시스템 통합     | `train_one_epoch`, `evaluate`, `main` |

**완성 결과:** 데이터 준비 → 모델 예측 → 손실 계산 → 파라미터 갱신 → 평가·예측이 연결된 선형회귀 학습 프로그램.

## PART 2 — 모델 구조 발전시키기

PART 1의 공통 학습 구조를 유지하면서 모델 구조와 문제 유형을 확장합니다. 회귀와 분류의 출력 형태, 마지막 계층, 손실함수, 예측 방법을 비교합니다.

| 번호 | 구현 주제       | 주요 코드·개념                            |
| ---- | --------------- | ----------------------------------------- |
| 09   | 선형회귀 확장   | 여러 입력 특성, `nn.Linear`               |
| 10   | 이진분류 모델   | `Sigmoid`, `BCEWithLogitsLoss`, Threshold |
| 11   | 다중분류 모델   | Logits, `CrossEntropyLoss`, `argmax`      |
| 12   | MLP 신경망      | 여러 `Linear` 계층, `ReLU`, `Sequential`  |
| 13   | CNN 이미지 분류 | `Conv2d`, `ReLU`, `MaxPool2d`             |
| 14   | CNN 구조 확장   | 채널 수, 특징맵, `Flatten`, 출력 Shape    |

**완성 결과:** 동일한 학습 흐름에서 선형회귀·이진분류·다중분류·MLP·CNN 모델을 구현하고 비교할 수 있는 구조.

## PART 3 — 학습 성능 개선하기

기본 학습 프로그램에서 Optimizer, 학습률, 정규화 및 데이터 처리 설정을 변경하고 학습 결과의 차이를 관찰합니다.

| 번호 | 구현 주제                       | 주요 코드·개념                     |
| ---- | ------------------------------- | ---------------------------------- |
| 15   | Optimizer 비교                  | SGD, Momentum, Adam, AdamW         |
| 16   | 학습률 조절                     | Learning Rate, `StepLR`, Scheduler |
| 17   | 과적합 관찰                     | Train Loss, Validation Loss        |
| 18   | Dropout 적용                    | `nn.Dropout`, `train`, `eval`      |
| 19   | Batch Normalization 적용        | `BatchNorm1d`, `BatchNorm2d`       |
| 20   | Weight Decay 및 Label Smoothing | `weight_decay`, `label_smoothing`  |
| 21   | 데이터 증강 구현                | `transforms`, Weak/Medium/Strong   |
| 22   | Early Stopping 구현             | `patience`, 최적 모델 저장         |

**완성 결과:** 학습 설정과 정규화 기법에 따른 성능 변화를 비교하고 과적합에 대응할 수 있는 실험 구조.

## PART 4 — 사전 학습 모델 활용하기

사전 학습된 모델을 불러와 새로운 분류 문제에 맞게 수정합니다. 학습할 계층을 선택하고 동결·미세조정 전략을 비교합니다.

| 번호 | 구현 주제                  | 주요 코드·개념               |
| ---- | -------------------------- | ---------------------------- |
| 23   | 사전 학습 모델 불러오기    | `models.resnet18`, `weights` |
| 24   | 분류기 교체                | `model.fc`, `nn.Linear`      |
| 25   | Full Freeze 구현           | `requires_grad=False`        |
| 26   | Partial / Full Fine-tuning | `layer4`, 학습 범위 변경     |
| 27   | 차등 학습률 구현           | `optimizer.param_groups`     |
| 28   | Gradual Unfreezing         | Epoch별 동결 해제            |

**완성 결과:** 사전 학습 모델의 분류기를 교체하고 동결 범위와 학습률을 조절하는 전이학습 프로그램.

## PART 5 — 실험 및 평가 시스템 만들기

모델 학습 결과를 측정·시각화·저장하고, 동일한 조건에서 여러 학습 전략을 비교할 수 있도록 구성합니다.

| 번호 | 구현 주제                | 주요 코드·개념                                    |
| ---- | ------------------------ | ------------------------------------------------- |
| 29   | 분류 성능 평가           | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 30   | 학습 결과 시각화         | Matplotlib, Loss/Accuracy 그래프                  |
| 31   | 모델 저장과 불러오기     | `state_dict`, `torch.save`, `torch.load`          |
| 32   | 실험 설정 및 재현성 관리 | Config, Seed, 학습 결과 기록                      |
| 33   | 교차검증 및 전략 비교    | Cross-validation, 동일 조건 실험                  |

**완성 결과:** 모델의 성능과 학습 과정을 기록하고 실험 결과를 비교할 수 있는 평가·관리 시스템.

## PART 6 — 통합 미니 프로젝트

앞에서 구현한 기능들을 하나의 이미지 분류 프로그램으로 통합합니다.

| 번호 | 구현 주제                   | 최종 결과                             |
| ---- | --------------------------- | ------------------------------------- |
| 34   | 이미지 분류 학습 파이프라인 | 데이터 준비부터 학습·평가까지 연결    |
| 35   | CNN과 사전 학습 모델 비교   | 정확도, 손실, 학습 시간 비교          |
| 36   | 새로운 이미지 추론 프로그램 | 저장된 모델을 불러와 실제 이미지 분류 |

**완성 결과:** 이미지 데이터를 학습하고, 모델별 성능을 비교하며, 저장된 모델로 새로운 이미지를 분류하는 통합 프로그램.

## 최종 결과물

- 재사용 가능한 PyTorch 학습 및 평가 구조
- 선형회귀·이진분류·다중분류·MLP·CNN 모델
- Optimizer·학습률·정규화·데이터 증강 실험
- 사전 학습 모델의 전이학습 및 미세조정
- 성능 지표·시각화·모델 저장 및 재현성 관리
- 이미지 분류 학습·비교·추론 통합 프로그램
