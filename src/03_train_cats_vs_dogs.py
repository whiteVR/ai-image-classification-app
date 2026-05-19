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

DATA_DIR = Path("data/cats_vs_dogs")
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"

OUTPUT_DIR = Path("output_CD")
MODEL_DIR = Path("models_CD")
MODEL_PATH = MODEL_DIR / "cats_vs_dogs_cnn.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20


def check_dataset() -> None:
    required_dirs = [
        TRAIN_DIR / "cats",
        TRAIN_DIR / "dogs",
        VALIDATION_DIR / "cats",
        VALIDATION_DIR / "dogs",
    ]

    missing_dirs = [path for path in required_dirs if not path.exists()]
    if missing_dirs:
        paths = "\n".join(f"- {path}" for path in missing_dirs)
        raise FileNotFoundError(f"데이터 폴더가 없습니다.\n{paths}")


def show_then_save(path: Path, dpi: int = 150) -> None:
    print(f"그래프를 3초 표시한 뒤 저장합니다: {path}")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_model() -> tf.keras.Model:
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, 3)),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_dataset.class_names
    return train_dataset, validation_dataset, class_names


def take_images_and_labels(dataset: tf.data.Dataset, count: int) -> tuple[np.ndarray, np.ndarray]:
    images_list = []
    labels_list = []

    for images, labels in dataset:
        images_list.append(images.numpy())
        labels_list.append(labels.numpy().reshape(-1))

        if sum(len(batch) for batch in labels_list) >= count:
            break

    images_array = np.concatenate(images_list, axis=0)[:count]
    labels_array = np.concatenate(labels_list, axis=0)[:count].astype(int)

    return images_array, labels_array


def show_sample_images(
    dataset: tf.data.Dataset,
    class_names: list[str],
    count: int = 10,
) -> None:
    images, labels = take_images_and_labels(dataset, count)

    plt.figure(figsize=(12, 5))

    for index in range(count):
        label_index = int(labels[index])
        label_name = class_names[label_index]

        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index].astype("uint8"))
        plt.title(f"label: {label_name}")
        plt.axis("off")

    show_then_save(OUTPUT_DIR / "sample_10_images.png")


def plot_history(history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Cats vs Dogs Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "accuracy_graph.png")

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Cats vs Dogs Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "loss_graph.png")


def show_prediction_images(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
    count: int = 10,
) -> None:
    images, labels = take_images_and_labels(dataset, count)

    probabilities = model.predict(images, verbose=0).reshape(-1)
    predicted_labels = (probabilities >= 0.5).astype(int)

    print(f"처음 {count}개 검증 이미지 예측:", [class_names[i] for i in predicted_labels])
    print(f"처음 {count}개 검증 이미지 정답:", [class_names[i] for i in labels])

    plt.figure(figsize=(12, 5))

    for index in range(count):
        true_index = int(labels[index])
        pred_index = int(predicted_labels[index])
        confidence = probabilities[index] if pred_index == 1 else 1.0 - probabilities[index]

        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index].astype("uint8"))
        plt.title(
            f"True: {class_names[true_index]}\n"
            f"Pred: {class_names[pred_index]} ({confidence:.1%})"
        )
        plt.axis("off")

    show_then_save(OUTPUT_DIR / "prediction_10_images.png")


def show_evaluation_result(
    test_loss: float,
    test_accuracy: float,
    class_names: list[str],
) -> None:
    plt.figure(figsize=(8, 5))
    plt.axis("off")

    result_text = (
        "Cats vs Dogs Classification Result\n\n"
        f"Validation Loss: {test_loss:.4f}\n"
        f"Validation Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})\n\n"
        f"Class 0: {class_names[0]}\n"
        f"Class 1: {class_names[1]}\n\n"
        "Prediction Rule\n"
        "sigmoid output < 0.5  → class 0\n"
        "sigmoid output >= 0.5 → class 1"
    )

    plt.text(
        0.5,
        0.5,
        result_text,
        ha="center",
        va="center",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="#f5f5f5"),
    )

    show_then_save(OUTPUT_DIR / "evaluation_result.png")


def main() -> None:
    check_dataset()

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    train_dataset, validation_dataset, class_names = load_datasets()
    print(f"클래스 이름: {class_names}")

    show_sample_images(train_dataset, class_names, count=10)

    train_dataset_for_training = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset_for_training = validation_dataset.prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.summary()

    class_weight = {
        0: 1.45,  # cats
        1: 0.76,  # dogs
   }
    
    print(f"class_weight 적용: {class_weight}")

    history = model.fit(
        train_dataset_for_training,
        epochs=EPOCHS,
        validation_data=validation_dataset_for_training,
        class_weight=class_weight,
    )

    test_loss, test_accuracy = model.evaluate(validation_dataset_for_training, verbose=0)
    print(f"검증 loss: {test_loss:.4f}")
    print(f"검증 accuracy: {test_accuracy:.4f}")

    plot_history(history)
    show_prediction_images(model, validation_dataset, class_names, count=10)
    show_evaluation_result(test_loss, test_accuracy, class_names)

    model.save(MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")
    print(f"결과 이미지 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
