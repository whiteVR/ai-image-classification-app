
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

MODEL_PATH = Path("models_CD/cats_vs_dogs_cnn.keras")
DATA_DIR = Path("data/cats_vs_dogs")
OUTPUT_DIR = Path("predict_output_CD")

IMAGE_SIZE = (160, 160)
DEFAULT_CLASS_NAMES = ["cats", "dogs"]


def load_class_names() -> list[str]:
    classname_paths = [
        DATA_DIR / "train" / "classname.txt",
        DATA_DIR / "validation" / "classname.txt",
    ]

    for path in classname_paths:
        if path.exists():
            names = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(names) >= 2:
                return names[:2]

    train_dir = DATA_DIR / "train"
    if train_dir.exists():
        folder_names = sorted(
            path.name
            for path in train_dir.iterdir()
            if path.is_dir()
        )
        if len(folder_names) >= 2:
            return folder_names[:2]

    return DEFAULT_CLASS_NAMES


def prepare_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGBA")

    # 투명 배경 PNG 처리: 흰 배경 위에 합성
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)

    image = background.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.array(image).astype("float32")
    return image_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


def show_then_save(path: Path, seconds: int = 3, dpi: int = 150) -> None:
    print(f"결과 이미지를 {seconds}초 표시한 뒤 저장합니다: {path}")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(seconds)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def show_prediction_result(
    prepared_image: np.ndarray,
    class_names: list[str],
    predicted_index: int,
    confidence: float,
    probabilities: np.ndarray,
    image_path: Path,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    save_path = OUTPUT_DIR / f"{image_path.stem}_catdog_prediction.png"

    cat_probability = 1.0 - probabilities[0]
    dog_probability = probabilities[0]

    plt.figure(figsize=(8, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(prepared_image[0].astype("uint8"))
    plt.title(
        f"Prediction: {class_names[predicted_index]}\n"
        f"Confidence: {confidence:.2%}"
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.bar(class_names, [cat_probability, dog_probability])
    plt.ylim(0, 1)
    plt.title("Class Probability")
    plt.ylabel("Probability")

    for index, value in enumerate([cat_probability, dog_probability]):
        plt.text(index, value + 0.03, f"{value:.1%}", ha="center")

    show_then_save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="학습된 cats_vs_dogs 모델로 고양이/개 이미지를 예측합니다."
    )
    parser.add_argument("image_path", type=Path, help="예측할 이미지 경로")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 파일이 없습니다. 먼저 `python src\\03_train_cats_vs_dogs.py`를 실행하세요."
        )

    if not args.image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {args.image_path}")

    class_names = load_class_names()

    model = tf.keras.models.load_model(MODEL_PATH)
    prepared_image = prepare_image(args.image_path)

    probability_dog = float(model.predict(prepared_image, verbose=0)[0][0])
    predicted_index = 1 if probability_dog >= 0.5 else 0
    confidence = probability_dog if predicted_index == 1 else 1.0 - probability_dog

    probabilities = np.array([probability_dog], dtype="float32")

    print(f"입력 이미지: {args.image_path}")
    print(f"클래스 이름: {class_names}")
    print(f"예측 결과: {class_names[predicted_index]}")
    print(f"확신도: {confidence:.2%}")
    print("클래스별 확률:")
    print(f"  {class_names[0]}: {(1.0 - probability_dog):.2%}")
    print(f"  {class_names[1]}: {probability_dog:.2%}")

    show_prediction_result(
        prepared_image=prepared_image,
        class_names=class_names,
        predicted_index=predicted_index,
        confidence=confidence,
        probabilities=probabilities,
        image_path=args.image_path,
    )


if __name__ == "__main__":
    main()
