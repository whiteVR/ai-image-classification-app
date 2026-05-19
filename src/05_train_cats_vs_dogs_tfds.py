import os
import logging
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
from keras.applications import MobileNetV2

tf.get_logger().setLevel(logging.ERROR)

TFDS_DATA_DIR = Path("data/tfds")
OUTPUT_DIR = Path("output_TFDS_CD")
MODEL_DIR = Path("models_TFDS_CD")
MODEL_PATH = MODEL_DIR / "cats_vs_dogs_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10


def show_then_save(path: Path, seconds: int = 3, dpi: int = 150) -> None:
    print(f"그래프를 {seconds}초 표시한 뒤 저장합니다: {path}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(seconds)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def load_tfds_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    print("TensorFlow Datasets에서 cats_vs_dogs 데이터셋을 불러옵니다.")

    (train_raw, validation_raw), info = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        with_info=True,
        data_dir=str(TFDS_DATA_DIR),
        shuffle_files=True,
    )

    class_names = info.features["label"].names
    print(f"클래스 이름: {class_names}")
    print(f"전체 데이터 수: {info.splits['train'].num_examples}")

    return train_raw, validation_raw, class_names


def resize_for_model(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    if training:
        dataset = dataset.shuffle(1000, seed=SEED)

    dataset = dataset.map(resize_for_model, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def take_resized_images_and_labels(dataset: tf.data.Dataset, count: int) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for image, label in dataset.take(count):
        images.append(tf.image.resize(image, IMAGE_SIZE).numpy().astype("uint8"))
        labels.append(int(label.numpy()))

    return np.array(images), np.array(labels)


def show_sample_images(dataset: tf.data.Dataset, class_names: list[str], count: int = 10) -> None:
    images, labels = take_resized_images_and_labels(dataset, count)

    plt.figure(figsize=(12, 5))
    for index in range(count):
        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index])
        plt.title(f"label: {class_names[int(labels[index])]}")
        plt.axis("off")

    show_then_save(OUTPUT_DIR / "sample_10_images.png")


def build_model() -> tf.keras.Model:
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
        ],
        name="data_augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)

    # Lambda(preprocess_input) 대신 저장/불러오기가 안전한 Rescaling 사용
    # MobileNetV2 preprocess_input과 같은 변환: 0~255 -> -1~1
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="cats_vs_dogs_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Cats vs Dogs Accuracy - MobileNetV2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "accuracy_graph.png")

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Cats vs Dogs Loss - MobileNetV2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "loss_graph.png")


def show_prediction_images(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list[str], count: int = 10) -> None:
    images, labels = take_resized_images_and_labels(dataset, count)
    probabilities = model.predict(images.astype("float32"), verbose=0).reshape(-1)
    predicted_labels = (probabilities >= 0.5).astype(int)

    print(f"처음 {count}개 검증 이미지 예측:", [class_names[i] for i in predicted_labels])
    print(f"처음 {count}개 검증 이미지 정답:", [class_names[i] for i in labels])

    plt.figure(figsize=(12, 5))
    for index in range(count):
        true_index = int(labels[index])
        pred_index = int(predicted_labels[index])
        confidence = probabilities[index] if pred_index == 1 else 1.0 - probabilities[index]

        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index])
        plt.title(f"True: {class_names[true_index]}\nPred: {class_names[pred_index]} ({confidence:.1%})")
        plt.axis("off")

    show_then_save(OUTPUT_DIR / "prediction_10_images.png")


def show_evaluation_result(validation_loss: float, validation_accuracy: float, class_names: list[str]) -> None:
    plt.figure(figsize=(8, 5))
    plt.axis("off")

    result_text = (
        "TFDS Cats vs Dogs - MobileNetV2 Transfer Learning\n\n"
        f"Validation Loss: {validation_loss:.4f}\n"
        f"Validation Accuracy: {validation_accuracy:.4f} ({validation_accuracy:.2%})\n\n"
        f"Class 0: {class_names[0]}\n"
        f"Class 1: {class_names[1]}\n\n"
        "Prediction Rule\n"
        "sigmoid output < 0.5  -> class 0\n"
        "sigmoid output >= 0.5 -> class 1"
    )

    plt.text(
        0.5,
        0.5,
        result_text,
        ha="center",
        va="center",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="#f5f5f5"),
    )

    show_then_save(OUTPUT_DIR / "evaluation_result.png")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    TFDS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, validation_raw, class_names = load_tfds_datasets()
    show_sample_images(train_raw, class_names, count=10)

    train_dataset = make_dataset(train_raw, training=True)
    validation_dataset = make_dataset(validation_raw, training=False)

    model = build_model()
    model.summary()

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
    )

    validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"검증 loss: {validation_loss:.4f}")
    print(f"검증 accuracy: {validation_accuracy:.4f}")

    plot_history(history)
    show_prediction_images(model, validation_raw, class_names, count=10)
    show_evaluation_result(validation_loss, validation_accuracy, class_names)

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"모델 저장 완료: {MODEL_PATH}")
    print(f"클래스 이름 저장 완료: {CLASS_NAMES_PATH}")
    print(f"결과 이미지 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
