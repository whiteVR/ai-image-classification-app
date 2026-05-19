import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MNIST_MODEL_PATH = PROJECT_ROOT / "models01" / "mnist_digit_cnn.keras"

CAT_DOG_MODELS = {
    "finetuned": {
        "label": "MobileNetV2 fine-tuned",
        "path": PROJECT_ROOT / "models_TFDS_CD_FT" / "cats_vs_dogs_mobilenetv2_finetuned.keras",
        "class_names": PROJECT_ROOT / "models_TFDS_CD_FT" / "class_names.txt",
    },
    "mobilenetv2": {
        "label": "MobileNetV2 feature extraction",
        "path": PROJECT_ROOT / "models_TFDS_CD" / "cats_vs_dogs_mobilenetv2.keras",
        "class_names": PROJECT_ROOT / "models_TFDS_CD" / "class_names.txt",
    },
    "cnn": {
        "label": "Kaggle CNN baseline",
        "path": PROJECT_ROOT / "models_CD" / "cats_vs_dogs_cnn.keras",
        "class_names": None,
    },
}

DIGIT_IMAGE_SIZE = (28, 28)
CAT_DOG_IMAGE_SIZE = (160, 160)
DEFAULT_CLASS_NAMES = ["cat", "dog"]


def _ensure_model_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")


@lru_cache(maxsize=1)
def load_mnist_model() -> tf.keras.Model:
    _ensure_model_exists(MNIST_MODEL_PATH)
    return tf.keras.models.load_model(MNIST_MODEL_PATH)


@lru_cache(maxsize=3)
def load_cat_dog_model(variant: str) -> tf.keras.Model:
    model_info = CAT_DOG_MODELS.get(variant)
    if model_info is None:
        raise ValueError(f"지원하지 않는 Cats/Dogs 모델 variant입니다: {variant}")

    model_path = model_info["path"]
    _ensure_model_exists(model_path)
    return tf.keras.models.load_model(model_path)


def load_class_names(variant: str) -> list[str]:
    model_info = CAT_DOG_MODELS.get(variant)
    if model_info is None:
        raise ValueError(f"지원하지 않는 Cats/Dogs 모델 variant입니다: {variant}")

    class_names_path = model_info.get("class_names")
    if class_names_path and class_names_path.exists():
        names = [
            line.strip()
            for line in class_names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) >= 2:
            return names[:2]

    return DEFAULT_CLASS_NAMES


def read_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError("이미지 파일을 읽을 수 없습니다.") from exc


def make_digit_strength(image: Image.Image) -> np.ndarray:
    rgba_image = image.convert("RGBA")
    rgba = np.array(rgba_image)

    rgb = rgba[:, :, :3].astype("float32")
    alpha = rgba[:, :, 3].astype("float32")

    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    transparent_ratio = np.mean(alpha < 128)

    if transparent_ratio > 0.05:
        digit_strength = alpha
    else:
        border_rgb = np.concatenate(
            [
                rgb[0, :, :],
                rgb[-1, :, :],
                rgb[:, 0, :],
                rgb[:, -1, :],
            ],
            axis=0,
        )

        background_color = np.median(border_rgb, axis=0)
        background_gray = (
            0.299 * background_color[0]
            + 0.587 * background_color[1]
            + 0.114 * background_color[2]
        )

        color_distance = np.linalg.norm(rgb - background_color, axis=2)
        color_distance = np.clip(color_distance * 1.8, 0, 255)

        if background_gray > 160:
            digit_strength = np.maximum(255 - gray, color_distance)
        elif background_gray < 95:
            digit_strength = np.maximum(gray, color_distance)
        else:
            digit_strength = color_distance

    digit_strength[alpha < 10] = 0
    return np.clip(digit_strength, 0, 255).astype("uint8")


def prepare_digit_image(image: Image.Image) -> np.ndarray:
    digit_strength = make_digit_strength(image)
    threshold = max(20, int(digit_strength.max() * 0.12))
    mask = digit_strength > threshold

    if not mask.any():
        raise ValueError("숫자 영역을 찾지 못했습니다. 이미지 대비를 확인하세요.")

    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cropped = digit_strength[y_min : y_max + 1, x_min : x_max + 1]
    digit_image = Image.fromarray(cropped, mode="L")

    width, height = digit_image.size
    scale = 20 / max(width, height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    digit_image = digit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", DIGIT_IMAGE_SIZE, 0)
    left = (DIGIT_IMAGE_SIZE[0] - new_width) // 2
    top = (DIGIT_IMAGE_SIZE[1] - new_height) // 2
    canvas.paste(digit_image, (left, top))

    pixels = np.array(canvas).astype("float32") / 255.0
    return pixels.reshape(1, 28, 28, 1)


def predict_digit(image_bytes: bytes) -> dict:
    image = read_image_bytes(image_bytes)
    prepared_image = prepare_digit_image(image)
    model = load_mnist_model()

    probabilities = model.predict(prepared_image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return {
        "task": "mnist",
        "prediction": str(predicted_digit),
        "confidence": confidence,
        "probabilities": [
            {"label": str(index), "probability": float(probability)}
            for index, probability in enumerate(probabilities)
        ],
        "modelPath": str(MNIST_MODEL_PATH.relative_to(PROJECT_ROOT)),
    }


def prepare_cat_dog_image(image: Image.Image) -> np.ndarray:
    rgba_image = image.convert("RGBA")
    background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    background.alpha_composite(rgba_image)

    rgb_image = background.convert("RGB")
    rgb_image = rgb_image.resize(CAT_DOG_IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.array(rgb_image).astype("float32")
    return image_array.reshape(1, CAT_DOG_IMAGE_SIZE[0], CAT_DOG_IMAGE_SIZE[1], 3)


def predict_cat_dog(image_bytes: bytes, variant: str = "finetuned") -> dict:
    image = read_image_bytes(image_bytes)
    prepared_image = prepare_cat_dog_image(image)
    model = load_cat_dog_model(variant)
    class_names = load_class_names(variant)

    dog_probability = float(model.predict(prepared_image, verbose=0)[0][0])
    cat_probability = 1.0 - dog_probability
    probabilities = [cat_probability, dog_probability]

    predicted_index = 1 if dog_probability >= 0.5 else 0
    confidence = probabilities[predicted_index]

    model_info = CAT_DOG_MODELS[variant]

    return {
        "task": "cats-vs-dogs",
        "variant": variant,
        "variantLabel": model_info["label"],
        "prediction": class_names[predicted_index],
        "confidence": confidence,
        "probabilities": [
            {"label": class_names[0], "probability": cat_probability},
            {"label": class_names[1], "probability": dog_probability},
        ],
        "modelPath": str(model_info["path"].relative_to(PROJECT_ROOT)),
    }


def model_status() -> dict:
    return {
        "mnist": {
            "path": str(MNIST_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "exists": MNIST_MODEL_PATH.exists(),
        },
        "catsDogs": {
            key: {
                "label": value["label"],
                "path": str(value["path"].relative_to(PROJECT_ROOT)),
                "exists": value["path"].exists(),
            }
            for key, value in CAT_DOG_MODELS.items()
        },
    }
