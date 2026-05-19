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

MODEL_PATH = Path("models_TFDS_CD_FT/cats_vs_dogs_mobilenetv2_finetuned.keras")
CLASS_NAMES_PATH = Path("models_TFDS_CD_FT/class_names.txt")
OUTPUT_DIR = Path("predict_outputs_TFDS_CD_FT")

IMAGE_SIZE = (160, 160)
DEFAULT_CLASS_NAMES = ["cat", "dog"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        names = [
            line.strip()
            for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) >= 2:
            return names[:2]

    return DEFAULT_CLASS_NAMES


def prepare_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)

    image = background.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.array(image).astype("float32")
    return image_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"이미지 파일 또는 폴더를 찾을 수 없습니다: {path}")


def show_then_save(path: Path, seconds: int = 3, dpi: int = 150) -> None:
    print(f"결과 이미지를 {seconds}초 표시한 뒤 저장합니다: {path}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(seconds)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_prediction_result(
    prepared_image: np.ndarray,
    class_names: list[str],
    predicted_index: int,
    confidence: float,
    dog_probability: float,
    image_path: Path,
    show_seconds: int,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / f"{image_path.stem}_finetuned_catdog_prediction.png"

    cat_probability = 1.0 - dog_probability

    plt.figure(figsize=(8, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(prepared_image[0].astype("uint8"))
    plt.title(f"Prediction: {class_names[predicted_index]}\nConfidence: {confidence:.2%}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.bar(class_names, [cat_probability, dog_probability])
    plt.ylim(0, 1)
    plt.title("Class Probability")
    plt.ylabel("Probability")

    for index, value in enumerate([cat_probability, dog_probability]):
        plt.text(index, min(value + 0.03, 0.97), f"{value:.1%}", ha="center")

    if show_seconds > 0:
        show_then_save(save_path, seconds=show_seconds)
    else:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"예측 이미지 저장 완료: {save_path}")


def predict_one_image(
    model: tf.keras.Model,
    image_path: Path,
    class_names: list[str],
    show_seconds: int,
) -> tuple[str, float, float]:
    prepared_image = prepare_image(image_path)

    dog_probability = float(model.predict(prepared_image, verbose=0)[0][0])
    predicted_index = 1 if dog_probability >= 0.5 else 0
    confidence = dog_probability if predicted_index == 1 else 1.0 - dog_probability

    save_prediction_result(
        prepared_image=prepared_image,
        class_names=class_names,
        predicted_index=predicted_index,
        confidence=confidence,
        dog_probability=dog_probability,
        image_path=image_path,
        show_seconds=show_seconds,
    )

    return class_names[predicted_index], confidence, dog_probability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuned MobileNetV2 모델로 고양이/개 이미지를 예측합니다."
    )
    parser.add_argument("image_path", type=Path, help="예측할 이미지 파일 또는 폴더 경로")
    parser.add_argument(
        "--show-seconds",
        type=int,
        default=3,
        help="결과 이미지를 화면에 표시할 시간. 0이면 화면 표시 없이 저장만 합니다.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 파일이 없습니다. 먼저 `python src\\07_train_cats_vs_dogs_finetune.py`를 실행하세요."
        )

    image_paths = collect_image_paths(args.image_path)
    if not image_paths:
        raise FileNotFoundError(f"예측할 이미지가 없습니다: {args.image_path}")

    class_names = load_class_names()
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"모델 경로: {MODEL_PATH}")
    print(f"클래스 이름: {class_names}")
    print(f"예측 이미지 수: {len(image_paths)}")

    result_rows = ["image_path,prediction,confidence,cat_probability,dog_probability"]

    for image_path in image_paths:
        prediction, confidence, dog_probability = predict_one_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            show_seconds=args.show_seconds,
        )

        cat_probability = 1.0 - dog_probability

        print("-" * 60)
        print(f"입력 이미지: {image_path}")
        print(f"예측 결과: {prediction}")
        print(f"확신도: {confidence:.2%}")
        print(f"{class_names[0]}: {cat_probability:.2%}")
        print(f"{class_names[1]}: {dog_probability:.2%}")

        result_rows.append(
            f"{image_path},{prediction},{confidence:.6f},"
            f"{cat_probability:.6f},{dog_probability:.6f}"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "prediction_summary.csv"
    summary_path.write_text("\n".join(result_rows), encoding="utf-8")

    print(f"예측 요약 CSV 저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
