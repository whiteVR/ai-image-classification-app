# 6단계: Render 배포 가이드

## 1. 권장 배포 방식

이 프로젝트는 `FastAPI + React TypeScript + TensorFlow/Keras 모델` 구조입니다. 배포는 **Render Web Service 단일 배포**를 권장합니다.

구조는 다음과 같습니다.

```text
사용자 브라우저
-> Render Web Service
   -> React 정적 파일 제공
   -> /api 요청은 FastAPI가 처리
   -> Keras 모델로 추론
```

프론트엔드와 백엔드를 따로 배포할 수도 있지만, 처음 배포에서는 단일 서비스가 관리하기 쉽습니다.

## 2. 배포 전 확인

GitHub 저장소:

```text
https://github.com/whiteVR/ai-image-classification-app
```

필수 파일:

```text
render.yaml
requirements.txt
app/backend/main.py
app/backend/inference.py
app/frontend/package.json
models01/mnist_digit_cnn.keras
models_TFDS_CD_FT/cats_vs_dogs_mobilenetv2_finetuned.keras
```

## 3. Render에서 배포

1. Render 접속

```text
https://render.com
```

2. GitHub 계정 연결
3. `New +` 선택
4. `Blueprint` 또는 `Web Service` 선택
5. GitHub 저장소 선택

```text
whiteVR/ai-image-classification-app
```

6. `render.yaml`을 인식하면 그대로 배포 진행

## 4. 수동 설정으로 배포할 경우

Render에서 Web Service를 직접 만들 경우 설정값은 다음과 같습니다.

| 항목 | 값 |
| --- | --- |
| Runtime | Python |
| Build Command | 아래 명령 사용 |
| Start Command | 아래 명령 사용 |
| Health Check Path | `/api/health` |

Build Command:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
cd app/frontend
npm install
npm run build
```

Start Command:

```bash
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT
```

## 5. 배포 후 확인

배포가 끝나면 Render가 다음과 같은 URL을 제공합니다.

```text
https://ai-image-classification-app.onrender.com
```

상태 확인:

```text
https://ai-image-classification-app.onrender.com/api/health
```

정상 응답:

```json
{"status":"ok"}
```

앱 화면:

```text
https://ai-image-classification-app.onrender.com
```

## 6. 주의 사항

- TensorFlow 설치와 모델 로딩 때문에 첫 배포 시간이 길 수 있습니다.
- 무료/저사양 인스턴스에서는 메모리 부족이 발생할 수 있습니다.
- 메모리 오류가 발생하면 Render 인스턴스를 상위 플랜으로 올려야 합니다.
- GitHub에는 100MB 이상 단일 파일을 일반 Git으로 올릴 수 없습니다. 현재 모델 파일은 100MB 미만입니다.
- 원본 Kaggle/TFDS 대용량 데이터는 Git에 올리지 않고, 학습된 모델과 결과물만 올리는 방식이 맞습니다.

## 7. 새 변경사항 배포

로컬에서 수정 후:

```powershell
git add .
git commit -m "Update deployment configuration"
git push
```

Render가 GitHub push를 감지하면 자동으로 다시 배포합니다.

