# 5단계: 앱 구성, Git 관리, 배포 절차

## 1. 최적 솔루션

이 프로젝트에는 `FastAPI 백엔드 + React TypeScript 프론트엔드` 구성이 가장 적합합니다.

이유는 다음과 같습니다.

| 항목 | 판단 |
| --- | --- |
| 기존 모델 | TensorFlow/Keras `.keras` 파일 |
| 추론 실행 | Python TensorFlow 런타임이 가장 안정적 |
| 사용자 화면 | 이미지 업로드와 결과 시각화가 필요 |
| 배포 | 백엔드와 프론트엔드를 분리하면 확장과 운영이 쉬움 |
| 교육 효과 | 모델 학습 코드, API, 화면 구성을 단계적으로 설명 가능 |

React만으로 브라우저에서 직접 실행하려면 모델을 TensorFlow.js 형식으로 변환해야 합니다. 이 프로젝트는 MobileNetV2 fine-tuning 모델과 기존 Keras 산출물이 이미 있으므로, 변환보다 Python 백엔드를 유지하는 편이 안전합니다.

## 2. 최종 앱 구조

```text
app
├── backend
│   ├── inference.py
│   └── main.py
└── frontend
    ├── package.json
    ├── index.html
    ├── vite.config.ts
    └── src
        ├── App.tsx
        └── styles.css
```

## 3. 백엔드 역할

백엔드는 FastAPI로 구성합니다.

주요 API는 다음과 같습니다.

| API | 역할 |
| --- | --- |
| `GET /api/health` | 서버 상태 확인 |
| `GET /api/models` | 모델 파일 존재 여부 확인 |
| `POST /api/predict/mnist` | 숫자 이미지 예측 |
| `POST /api/predict/cats-dogs` | 고양이/개 이미지 예측 |

백엔드는 다음 모델을 사용합니다.

| 작업 | 모델 |
| --- | --- |
| MNIST 숫자 예측 | `models01/mnist_digit_cnn.keras` |
| Cats/Dogs 기본 CNN | `models_CD/cats_vs_dogs_cnn.keras` |
| Cats/Dogs MobileNetV2 | `models_TFDS_CD/cats_vs_dogs_mobilenetv2.keras` |
| Cats/Dogs fine-tuned | `models_TFDS_CD_FT/cats_vs_dogs_mobilenetv2_finetuned.keras` |

## 4. 프론트엔드 역할

프론트엔드는 React TypeScript와 Vite로 구성합니다.

기능은 다음과 같습니다.

- MNIST / Cats-Dogs 작업 선택
- 이미지 업로드
- 업로드 이미지 미리보기
- Cats/Dogs 모델 variant 선택
- 예측 결과 표시
- 클래스별 확률 막대 표시
- 모델 파일 상태 표시

## 5. 실행 절차

### 5.1 Python 의존성 설치

```powershell
cd C:\Users\white\Documents\Codex\2026-05-09\ai-0-9-ai
.\aivenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.2 백엔드 실행

```powershell
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

정상 실행 후 확인:

```text
http://127.0.0.1:8000/api/health
```

### 5.3 프론트엔드 실행

새 PowerShell을 열고 실행합니다.

```powershell
cd C:\Users\white\Documents\Codex\2026-05-09\ai-0-9-ai\app\frontend
npm install
npm run dev
```

브라우저에서 접속합니다.

```text
http://localhost:5173
```

## 6. Git 관리 방침

현재 프로젝트 폴더는 Git 저장소가 아닙니다. 처음 한 번만 다음 명령을 실행합니다.

```powershell
cd C:\Users\white\Documents\Codex\2026-05-09\ai-0-9-ai
git init
```

상태 확인:

```powershell
git status
```

추가:

```powershell
git add README.md requirements.txt .gitignore src docs app models01 models_CD models_TFDS_CD models_TFDS_CD_FT output01 output_CD output_TFDS_CD output_TFDS_CD_FT predict_outputs predict_output_CD predict_outputs_TFDS_CD predict_outputs_TFDS_CD_FT
```

커밋:

```powershell
git commit -m "Build AI image classification app"
```

주의할 점:

- `aivenv`, `.venv`, `node_modules`, `__pycache__`는 Git에 넣지 않습니다.
- Kaggle 원본 데이터와 TFDS 다운로드 캐시는 용량이 커서 Git에 넣지 않습니다.
- 모델 파일은 100MB 미만이면 일반 Git에 넣을 수 있습니다.
- 100MB 이상 파일이 생기면 Git LFS를 사용해야 합니다.

## 7. 배포 전략

### 7.1 로컬 시연

가장 먼저 권장하는 방식입니다.

```text
FastAPI: http://127.0.0.1:8000
React: http://localhost:5173
```

교육 현장이나 개인 PC 시연에는 이 방식이 가장 단순합니다.

### 7.2 서버 배포

운영 배포는 다음 구조를 권장합니다.

```text
React 정적 빌드
-> Nginx 또는 정적 호스팅

FastAPI 백엔드
-> Python 서버 또는 Docker 컨테이너
```

### 7.3 단일 서버 배포

하나의 서버에서 운영한다면 다음 순서입니다.

```powershell
cd app\frontend
npm install
npm run build
```

생성된 `dist` 폴더를 정적 파일로 제공하고, `/api` 요청은 FastAPI로 전달합니다.

## 8. 최종 산출물 체크리스트

| 산출물 | 위치 |
| --- | --- |
| MNIST 학습/예측 코드 | `src/01_train_mnist_digits.py`, `src/02_predict_digit_image.py` |
| Cats/Dogs 학습/예측 코드 | `src/03_train_cats_vs_dogs.py` ~ `src/08_predict_cats_vs_dogs_finetune.py` |
| 웹 백엔드 | `app/backend` |
| 웹 프론트엔드 | `app/frontend` |
| 수업 문서 | `docs` |
| 앱 실행 문서 | `app/README.md`, `docs/05_app_solution_and_deployment.md` |
| 모델 파일 | `models01`, `models_CD`, `models_TFDS_CD`, `models_TFDS_CD_FT` |
| 결과 이미지 | `output*`, `predict_output*` |

