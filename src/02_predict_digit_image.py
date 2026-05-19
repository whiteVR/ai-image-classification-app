import os
import logging
import argparse
from pathlib import Path

# TensorFlow 경고/정보 메시지 숨기기: 반드시 tensorflow import 전에 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

tf.get_logger().setLevel(logging.ERROR)

MODEL_PATH = Path("models01/mnist_digit_cnn.keras")
OUTPUT_DIR = Path("predict_outputs01")


def make_digit_strength(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGBA")
    rgba = np.array(image)

    rgb = rgba[:, :, :3].astype("float32")
    alpha = rgba[:, :, 3].astype("float32")

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    transparent_ratio = np.mean(alpha < 128)

    if transparent_ratio > 0.05:
        # 투명 배경 PNG는 alpha 채널을 숫자 영역으로 사용
        digit_strength = alpha
    else:
        # 불투명 이미지에서는 가장자리 색을 배경색으로 추정
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
            # 흰 배경 + 검은/컬러 숫자
            digit_strength = np.maximum(255 - gray, color_distance)
        elif background_gray < 95:
            # 검은 배경 + 흰/컬러 숫자
            digit_strength = np.maximum(gray, color_distance)
        else:
            # 중간색 배경은 배경색과의 차이를 사용
            digit_strength = color_distance

    digit_strength[alpha < 10] = 0
    return np.clip(digit_strength, 0, 255).astype("uint8")


def prepare_digit_image(image_path: Path) -> np.ndarray:
    digit_strength = make_digit_strength(image_path)

    threshold = max(20, int(digit_strength.max() * 0.12))
    mask = digit_strength > threshold

    if not mask.any():
        raise ValueError("숫자 영역을 찾지 못했습니다. 이미지 대비를 확인하세요.")

    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cropped = digit_strength[y_min:y_max + 1, x_min:x_max + 1]
    digit_image = Image.fromarray(cropped, mode="L")

    # MNIST와 비슷하게 28x28 안에 약간의 여백을 두고 중앙 배치
    width, height = digit_image.size
    scale = 20 / max(width, height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    digit_image = digit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    canvas.paste(digit_image, (left, top))

    pixels = np.array(canvas).astype("float32") / 255.0
    return pixels.reshape(1, 28, 28, 1)


def save_and_show_prediction(
    prepared_image: np.ndarray,
    predicted_digit: int,
    confidence: float,
    image_path: Path,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    save_path = OUTPUT_DIR / f"{image_path.stem}_prediction.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(prepared_image.reshape(28, 28), cmap="gray")
    plt.title(f"Prediction: {predicted_digit} ({confidence:.2%})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"예측 이미지 저장 완료: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="학습된 MNIST 모델로 숫자 이미지를 예측합니다.")
    parser.add_argument("image_path", type=Path, help="예측할 숫자 이미지 경로")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 파일이 없습니다. 먼저 `python src\\01_train_mnist_digits.py`를 실행하세요."
        )

    if not args.image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {args.image_path}")

    model = tf.keras.models.load_model(MODEL_PATH)
    prepared_image = prepare_digit_image(args.image_path)

    probabilities = model.predict(prepared_image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    print(f"예측 결과: {predicted_digit}")
    print(f"확신도: {confidence:.2%}")
    print("숫자별 확률:")

    for digit, probability in enumerate(probabilities):
        print(f"  {digit}: {probability:.2%}")

    save_and_show_prediction(prepared_image, predicted_digit, confidence, args.image_path)


if __name__ == "__main__":
    main()