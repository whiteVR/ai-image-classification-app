from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import model_status, predict_cat_dog, predict_digit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "app" / "frontend" / "dist"

app = FastAPI(
    title="AI Image Classification API",
    description="MNIST 숫자 분류와 Cats/Dogs 분류를 제공하는 FastAPI 백엔드입니다.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_upload(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    return image_bytes


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict:
    return model_status()


@app.post("/api/predict/mnist")
async def predict_mnist_endpoint(file: UploadFile = File(...)) -> dict:
    try:
        image_bytes = await _read_upload(file)
        return predict_digit(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MNIST 예측 실패: {exc}") from exc


@app.post("/api/predict/cats-dogs")
async def predict_cats_dogs_endpoint(
    variant: str = "finetuned",
    file: UploadFile = File(...),
) -> dict:
    try:
        image_bytes = await _read_upload(file)
        return predict_cat_dog(image_bytes, variant=variant)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cats/Dogs 예측 실패: {exc}") from exc


if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )


@app.get("/")
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="프론트엔드 빌드 파일이 없습니다. app/frontend에서 npm run build를 실행하세요.",
        )
    return FileResponse(index_path)
