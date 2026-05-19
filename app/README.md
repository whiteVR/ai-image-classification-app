# AI 이미지 분류 앱

이 앱은 기존 프로젝트의 학습된 Keras 모델을 웹 화면에서 실행하기 위한 애플리케이션입니다.

## 구조

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

## 실행

백엔드:

```powershell
cd C:\Users\white\Documents\Codex\2026-05-09\ai-0-9-ai
.\aivenv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

프론트엔드:

```powershell
cd C:\Users\white\Documents\Codex\2026-05-09\ai-0-9-ai\app\frontend
npm install
npm run dev
```

브라우저:

```text
http://localhost:5173
```

## 지원 모델

- MNIST CNN: `models01/mnist_digit_cnn.keras`
- Cats/Dogs Kaggle CNN: `models_CD/cats_vs_dogs_cnn.keras`
- Cats/Dogs MobileNetV2: `models_TFDS_CD/cats_vs_dogs_mobilenetv2.keras`
- Cats/Dogs MobileNetV2 fine-tuned: `models_TFDS_CD_FT/cats_vs_dogs_mobilenetv2_finetuned.keras`

