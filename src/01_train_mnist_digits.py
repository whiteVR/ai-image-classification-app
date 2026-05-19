import os
import logging
from pathlib import Path

# TensorFlow 경고/정보 메시지 숨기기: 반드시 tensorflow import 전에 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models

tf.get_logger().setLevel(logging.ERROR)

OUTPUT_DIR = Path("output01")
MODEL_DIR = Path("models01")
MODEL_PATH = MODEL_DIR / "mnist_digit_cnn.keras"


def show_then_save(path: Path, dpi: int = 150) -> None:
    """그래프를 3초 표시한 뒤 이미지 파일로 저장합니다."""
    print(f"그래프를 3초 표시한 뒤 저장합니다: {path}")
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.savefig(path, dpi=dpi)
    plt.close()


def build_model() -> tf.keras.Model:
    """MNIST 숫자 분류용 CNN 모델을 생성합니다."""
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def show_examples(images: np.ndarray, labels: np.ndarray) -> None:
    """MNIST 예시 이미지 10장을 표시하고 저장합니다."""
    plt.figure(figsize=(10, 4))

    for index in range(10):
        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index], cmap="gray")
        plt.title(f"label: {labels[index]}")
        plt.axis("off")

    show_then_save(OUTPUT_DIR / "mnist_samples.png")


def plot_history(history: tf.keras.callbacks.History) -> None:
    """학습 정확도와 손실 그래프를 각각 저장합니다."""
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "accuracy_graph.png")

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    show_then_save(OUTPUT_DIR / "loss_graph.png")


def show_predictions(
    model: tf.keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    count: int = 15,
) -> None:
    """테스트 이미지 예측 결과를 그림으로 표시하고 저장합니다."""
    predictions = model.predict(test_images[:count])
    predicted_labels = np.argmax(predictions, axis=1)

    print(f"처음 {count}개 테스트 이미지 예측:", predicted_labels.tolist())
    print(f"처음 {count}개 테스트 이미지 정답:", test_labels[:count].tolist())

    plt.figure(figsize=(12, 6))

    for index in range(count):
        plt.subplot(3, 5, index + 1)
        plt.imshow(test_images[index].reshape(28, 28), cmap="gray")
        plt.title(f"True: {test_labels[index]}, Pred: {predicted_labels[index]}")
        plt.axis("off")

    show_then_save(OUTPUT_DIR / "prediction_results.png")


def main() -> None:
    """MNIST 데이터 로드부터 모델 저장까지 전체 흐름을 실행합니다."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    print(f"학습 이미지 shape: {train_images.shape}")
    print(f"테스트 이미지 shape: {test_images.shape}")
    print(f"첫 번째 정답 label: {train_labels[0]}")

    show_examples(train_images, train_labels)

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]

    model = build_model()
    model.summary()

    history = model.fit(
        train_images,
        train_labels,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
    )

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"테스트 loss: {test_loss:.4f}")
    print(f"테스트 accuracy: {test_accuracy:.4f}")

    plot_history(history)
    show_predictions(model, test_images, test_labels, count=15)

    model.save(MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")
    print(f"결과 이미지 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()