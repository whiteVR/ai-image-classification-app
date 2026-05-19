# AI 기초 이미지 분류 프로젝트

이 프로젝트는 AI 입문자가 이미지 분류의 흐름을 직접 경험하도록 만든 교육용 예제입니다.

처음에는 사람이 손으로 쓴 `0`부터 `9`까지의 숫자 이미지를 학습해 새로운 숫자 이미지를 맞히는 모델을 만들고, 이후 같은 원리를 이용해 개와 고양이 사진을 분류하는 프로젝트로 확장합니다.

## 학습 목표

- 이미지가 컴퓨터에서 숫자 배열로 표현된다는 것을 이해합니다.
- 학습 데이터와 테스트 데이터의 차이를 이해합니다.
- CNN 모델이 이미지 특징을 학습하는 방식을 경험합니다.
- 학습된 모델로 새로운 이미지를 예측합니다.
- 숫자 분류에서 개/고양이 분류로 문제를 확장합니다.

## 폴더 구성

```text
.
├── README.md
├── requirements.txt
├── src
│   ├── 01_train_mnist_digits.py
│   ├── 02_predict_digit_image.py
│   └── 03_train_cats_vs_dogs.py
└── data
    └── cats_vs_dogs
        ├── train
        │   ├── cats
        │   └── dogs
        └── validation
            ├── cats
            └── dogs
```

## 설치

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1단계: 손글씨 숫자 분류 모델 학습

MNIST 데이터셋은 `0`부터 `9`까지의 손글씨 숫자 이미지 7만 장으로 구성된 대표적인 입문용 데이터셋입니다.

```powershell
python src\01_train_mnist_digits.py
```

실행하면 다음 내용을 확인할 수 있습니다.

- 학습 이미지 크기: `28 x 28`
- 학습 정확도와 검증 정확도
- 최종 테스트 정확도
- 저장된 모델: `models/mnist_digit_cnn.keras`

## 2단계: 새로운 숫자 이미지 예측

직접 만든 숫자 이미지가 있다면 아래처럼 예측할 수 있습니다.

```powershell
python src\02_predict_digit_image.py path\to\digit.png
```

이미지는 흰 배경에 검은 숫자 또는 검은 배경에 흰 숫자 모두 어느 정도 처리되도록 되어 있습니다. 다만 숫자가 중앙에 있고 한 글자만 들어 있을수록 결과가 좋습니다.

## 3단계: 개와 고양이 분류로 확장

숫자 분류와 개/고양이 분류는 겉보기에는 달라 보이지만, 핵심 흐름은 같습니다.

1. 이미지를 준비합니다.
2. 이미지를 모델에 넣을 수 있는 크기와 숫자 범위로 바꿉니다.
3. 모델이 이미지의 특징을 학습합니다.
4. 새로운 이미지를 보고 클래스를 예측합니다.

개/고양이 이미지는 저작권과 용량 문제로 프로젝트에 포함하지 않았습니다. 아래 폴더에 이미지를 넣어주세요.

```text
data/cats_vs_dogs/train/cats
data/cats_vs_dogs/train/dogs
data/cats_vs_dogs/validation/cats
data/cats_vs_dogs/validation/dogs
```

그 다음 실행합니다.

```powershell
python src\03_train_cats_vs_dogs.py
```

저장되는 모델:

```text
models/cats_vs_dogs_cnn.keras
```

## 추천 수업 흐름

### 1차시: AI 이미지 분류 개념

- 상세 교안: [docs/01_ai_image_classification_concepts.md](docs/01_ai_image_classification_concepts.md)
- AI, 머신러닝, 딥러닝의 차이
- 분류 문제란 무엇인가
- 이미지가 픽셀 숫자로 표현되는 방식
- 학습 데이터와 테스트 데이터

### 2차시: MNIST 숫자 분류

- 상세 교안: [docs/02_mnist_digit_classification.md](docs/02_mnist_digit_classification.md)
- MNIST 데이터셋 살펴보기
- CNN 모델 학습하기
- 정확도와 손실 그래프 읽기
- 틀린 예측 사례 관찰하기

### 3차시: 내 숫자 이미지 예측

- 상세 교안: [docs/03_predict_custom_digit_image.md](docs/03_predict_custom_digit_image.md)
- 직접 쓴 숫자 이미지를 준비하기
- 이미지 크기, 색상, 정규화 처리 이해하기
- 모델이 어떤 숫자로 판단하는지 확인하기
- 틀렸을 때 원인 생각해보기

### 4차시: 개와 고양이 분류

- 상세 교안: [docs/04_cats_vs_dogs_classification.md](docs/04_cats_vs_dogs_classification.md)
- 실제 사진 데이터의 어려움 이해하기
- 데이터 폴더 구조 만들기
- 과적합 개념 소개하기
- 데이터 증강의 필요성 이해하기

### 5단계: 앱 구성과 배포

- 상세 문서: [docs/05_app_solution_and_deployment.md](docs/05_app_solution_and_deployment.md)
- Render 배포 문서: [docs/06_render_deployment_guide.md](docs/06_render_deployment_guide.md)
- FastAPI 백엔드로 Keras 모델 추론 API 만들기
- React TypeScript 프론트엔드로 이미지 업로드 앱 만들기
- Git 관리와 배포 절차 정리

## 앱 실행

백엔드 실행:

```powershell
.\aivenv\Scripts\Activate.ps1
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

프론트엔드 실행:

```powershell
cd app\frontend
npm install
npm run dev
```

브라우저 접속:

```text
http://localhost:5173
```

## 핵심 개념 정리

- `epoch`: 전체 학습 데이터를 몇 번 반복해서 볼지 정하는 값입니다.
- `batch`: 데이터를 한 번에 몇 장씩 나눠 학습할지 정하는 값입니다.
- `accuracy`: 모델이 맞힌 비율입니다.
- `loss`: 모델이 얼마나 틀리고 있는지를 나타내는 값입니다. 낮을수록 좋습니다.
- `CNN`: 이미지에서 선, 모서리, 형태 같은 특징을 단계적으로 찾는 딥러닝 구조입니다.
- `overfitting`: 학습 데이터는 잘 맞히지만 새로운 데이터는 잘 못 맞히는 상태입니다.

## 수업용 질문

- 컴퓨터는 이미지를 어떻게 숫자로 바꿀까요?
- 학습 데이터와 시험 데이터는 왜 나누어야 할까요?
- 숫자 `1`과 `7`을 헷갈리는 이유는 무엇일까요?
- 개와 고양이 사진 분류가 숫자 분류보다 어려운 이유는 무엇일까요?
- 사진을 더 많이 모으면 항상 성능이 좋아질까요?
