import os
import logging
from pathlib import Path

# TensorFlow 경고/정보 메시지 숨기기: 반드시 tensorflow import 전에 설정
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
HARD_CASE_DIR = Path("data/hard_cases")

OUTPUT_DIR = Path("output_TFDS_CD_FT")
MODEL_DIR = Path("models_TFDS_CD_FT")
MODEL_PATH = MODEL_DIR / "cats_vs_dogs_mobilenetv2_finetuned.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

FEATURE_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAST_N_LAYERS = 30
HARD_CASE_REPEAT = 150

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def show_then_save(path: Path, seconds: int = 3, dpi: int = 150) -> None:
    print(f"그래프를 {seconds}초 표시한 뒤 저장합니다: {path}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(seconds)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def ensure_hard_case_folders() -> None:
    (HARD_CASE_DIR / "cats").mkdir(parents=True, exist_ok=True)
    (HARD_CASE_DIR / "dogs").mkdir(parents=True, exist_ok=True)


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


def collect_hard_case_paths() -> tuple[list[str], list[int]]:
    label_map = {
        "cat": 0,
        "cats": 0,
        "dog": 1,
        "dogs": 1,
    }

    image_paths = []
    labels = []

    for folder_name, label in label_map.items():
        folder = HARD_CASE_DIR / folder_name
        if not folder.exists():
            continue

        for file_path in sorted(folder.rglob("*")):
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(file_path))
                labels.append(label)

    return image_paths, labels


def make_hard_case_dataset() -> tuple[tf.data.Dataset | None, int]:
    image_paths, labels = collect_hard_case_paths()

    if not image_paths:
        print(f"하드케이스 이미지 없음: {HARD_CASE_DIR}")
        print("필요하면 실패 이미지를 data/hard_cases/cats 또는 data/hard_cases/dogs에 넣으세요.")
        return None, 0

    print(f"하드케이스 이미지 수: {len(image_paths)}")
    print(f"하드케이스 반복 횟수: {HARD_CASE_REPEAT}")

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return image, tf.cast(label, tf.int64)

    return path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE), len(image_paths)


def resize_for_model(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    if training:
        dataset = dataset.shuffle(2000, seed=SEED)

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


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
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
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="cats_vs_dogs_mobilenetv2_finetuned")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def set_fine_tuning(base_model: tf.keras.Model, train_last_n_layers: int) -> None:
    base_model.trainable = True

    for layer in base_model.layers[:-train_last_n_layers]:
        layer.trainable = False

    # BatchNormalization은 소량 데이터 fine-tuning에서 불안정해질 수 있어 고정
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"MobileNetV2 fine-tuning 활성화 layer 수: {trainable_count}")


def plot_history(feature_history: tf.keras.callbacks.History, fine_history: tf.keras.callbacks.History) -> None:
    acc = feature_history.history["accuracy"] + fine_history.history["accuracy"]
    val_acc = feature_history.history["val_accuracy"] + fine_history.history["val_accuracy"]
    loss = feature_history.history["loss"] + fine_history.history["loss"]
    val_loss = feature_history.history["val_loss"] + fine_history.history["val_loss"]

    fine_tune_start = len(feature_history.history["accuracy"])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(fine_tune_start - 1, linestyle="--", label="Fine-tuning Start")
    plt.title("Cats vs Dogs Accuracy - Fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "accuracy_graph_finetune.png")

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(fine_tune_start - 1, linestyle="--", label="Fine-tuning Start")
    plt.title("Cats vs Dogs Loss - Fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    show_then_save(OUTPUT_DIR / "loss_graph_finetune.png")


def show_prediction_images(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
    count: int = 10,
) -> None:
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

    show_then_save(OUTPUT_DIR / "prediction_10_images_finetune.png")


def show_evaluation_result(validation_loss: float, validation_accuracy: float, class_names: list[str]) -> None:
    plt.figure(figsize=(8, 5))
    plt.axis("off")

    result_text = (
        "TFDS Cats vs Dogs - MobileNetV2 Fine-tuning\n\n"
        f"Validation Loss: {validation_loss:.4f}\n"
        f"Validation Accuracy: {validation_accuracy:.4f} ({validation_accuracy:.2%})\n\n"
        f"Class 0: {class_names[0]}\n"
        f"Class 1: {class_names[1]}\n\n"
        f"Feature Epochs: {FEATURE_EPOCHS}\n"
        f"Fine-tune Epochs: {FINE_TUNE_EPOCHS}\n"
        f"Fine-tuned Last Layers: {FINE_TUNE_LAST_N_LAYERS}\n"
        f"Hard Case Repeat: {HARD_CASE_REPEAT}"
    )

    plt.text(
        0.5,
        0.5,
        result_text,
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="#f5f5f5"),
    )

    show_then_save(OUTPUT_DIR / "evaluation_result_finetune.png")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    TFDS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hard_case_folders()

    train_raw, validation_raw, class_names = load_tfds_datasets()
    hard_case_raw, hard_case_count = make_hard_case_dataset()

    if hard_case_raw is not None:
        train_raw = train_raw.concatenate(hard_case_raw.repeat(HARD_CASE_REPEAT))

    show_sample_images(train_raw, class_names, count=10)

    train_dataset = make_dataset(train_raw, training=True)
    validation_dataset = make_dataset(validation_raw, training=False)

    model, base_model = build_model()
    model.summary()

    print("\n[1단계] MobileNetV2 동결 상태에서 분류층 학습")
    feature_history = model.fit(
        train_dataset,
        epochs=FEATURE_EPOCHS,
        validation_data=validation_dataset,
    )

    print("\n[2단계] MobileNetV2 상위 일부 layer fine-tuning")
    set_fine_tuning(base_model, FINE_TUNE_LAST_N_LAYERS)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00002),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    fine_history = model.fit(
        train_dataset,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_dataset,
    )

    validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"검증 loss: {validation_loss:.4f}")
    print(f"검증 accuracy: {validation_accuracy:.4f}")

    plot_history(feature_history, fine_history)
    show_prediction_images(model, validation_raw, class_names, count=10)
    show_evaluation_result(validation_loss, validation_accuracy, class_names)

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"모델 저장 완료: {MODEL_PATH}")
    print(f"클래스 이름 저장 완료: {CLASS_NAMES_PATH}")
    print(f"결과 이미지 저장 폴더: {OUTPUT_DIR}")
    print(f"사용된 하드케이스 이미지 수: {hard_case_count}")


if __name__ == "__main__":
    main()
