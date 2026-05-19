# 7단계: Streamlit 버전 앱과 무료 배포

## 1. 목적

Render 배포와 별도로 Streamlit 기반 데모 앱을 추가합니다. Streamlit은 Python만으로 이미지 업로드 UI와 모델 예측 화면을 만들 수 있어 AI 교육용 데모에 적합합니다.

## 2. 추가된 파일

```text
streamlit_app.py
requirements-streamlit.txt
```

`streamlit_app.py`는 기존 FastAPI 백엔드의 추론 함수를 재사용합니다.

```text
app/backend/inference.py
```

따라서 MNIST와 Cats/Dogs의 전처리, 모델 로딩, 예측 결과는 React/FastAPI 앱과 동일합니다.

## 3. 로컬 실행

프로젝트 루트에서 실행합니다.

```powershell
cd C:\Users\white\Documents\Codex\2026-05-09\ai-0-9-ai
.\aivenv\Scripts\Activate.ps1
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

브라우저가 자동으로 열리거나, 다음 주소로 접속합니다.

```text
http://localhost:8501
```

## 4. Streamlit Community Cloud 배포

1. Streamlit Community Cloud 접속

```text
https://share.streamlit.io
```

2. GitHub 계정 연결
3. 저장소 선택

```text
whiteVR/ai-image-classification-app
```

4. Main file path 지정

```text
streamlit_app.py
```

5. Python requirements는 기본적으로 저장소의 `requirements.txt`를 찾습니다. Streamlit Cloud에서 별도 지정이 가능하면 다음 파일을 사용합니다.

```text
requirements-streamlit.txt
```

별도 지정이 어렵다면 `requirements.txt`에 `streamlit`과 `pandas`를 포함시키는 방식도 가능합니다.

## 5. Hugging Face Spaces 배포

Hugging Face Spaces에서도 Streamlit SDK를 선택해 배포할 수 있습니다.

1. Hugging Face에서 New Space 생성
2. SDK 선택

```text
Streamlit
```

3. GitHub 저장소 파일을 Space에 복사하거나 Git remote로 push
4. 필요한 파일

```text
streamlit_app.py
requirements.txt 또는 requirements-streamlit.txt
app/backend/inference.py
models01/mnist_digit_cnn.keras
models_TFDS_CD_FT/cats_vs_dogs_mobilenetv2_finetuned.keras
```

## 6. 주의 사항

- TensorFlow 모델 로딩 때문에 첫 실행이 느릴 수 있습니다.
- 무료 환경에서는 메모리 제한이 있을 수 있습니다.
- Streamlit Cloud가 `requirements-streamlit.txt`를 자동 인식하지 못하면 `requirements.txt`에 Streamlit 의존성을 추가해야 합니다.
- 모델 파일이 100MB를 넘으면 GitHub 일반 Git 대신 Git LFS 또는 외부 모델 저장소를 사용해야 합니다.

