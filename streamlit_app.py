from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from app.backend.inference import CAT_DOG_MODELS, model_status, predict_cat_dog, predict_digit


st.set_page_config(
    page_title="AI Image Classification",
    page_icon="🧠",
    layout="wide",
)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def probability_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "class": item["label"],
                "probability": item["probability"],
                "percent": format_percent(item["probability"]),
            }
            for item in result["probabilities"]
        ]
    )


def render_result(result: dict) -> None:
    st.subheader("예측 결과")

    metric_left, metric_right = st.columns(2)
    metric_left.metric("Prediction", result["prediction"])
    metric_right.metric("Confidence", format_percent(result["confidence"]))

    st.caption(f"Model: `{result['modelPath']}`")
    if result.get("variantLabel"):
        st.caption(f"Variant: `{result['variantLabel']}`")

    df = probability_table(result)
    st.dataframe(
        df[["class", "percent"]],
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(df.set_index("class")["probability"])


def render_model_status() -> None:
    status = model_status()
    rows = [
        {
            "model": "MNIST CNN",
            "path": status["mnist"]["path"],
            "exists": status["mnist"]["exists"],
        }
    ]

    for info in status["catsDogs"].values():
        rows.append(
            {
                "model": info["label"],
                "path": info["path"],
                "exists": info["exists"],
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def predict_uploaded_image(task: str, uploaded_file, variant: str | None = None) -> dict:
    image_bytes = uploaded_file.getvalue()
    if task == "mnist":
        return predict_digit(image_bytes)
    if variant is None:
        raise ValueError("Cats/Dogs 모델 variant가 필요합니다.")
    return predict_cat_dog(image_bytes, variant=variant)


st.title("AI 이미지 분류 실습 앱")
st.write("MNIST 숫자 분류와 Cats/Dogs 분류 모델을 Streamlit에서 실행합니다.")

with st.expander("모델 파일 상태", expanded=False):
    render_model_status()

tab_mnist, tab_cats_dogs = st.tabs(["MNIST 숫자 분류", "Cats / Dogs 분류"])

with tab_mnist:
    st.header("MNIST 숫자 분류와 예측")
    st.write("손글씨 숫자 이미지를 업로드하면 `0`부터 `9`까지의 확률을 계산합니다.")

    uploaded_digit = st.file_uploader(
        "숫자 이미지 업로드",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="mnist-uploader",
    )

    if uploaded_digit:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("입력 이미지")
            st.image(Image.open(uploaded_digit), use_container_width=True)

        with right:
            if st.button("MNIST 예측 실행", type="primary"):
                try:
                    result = predict_uploaded_image("mnist", uploaded_digit)
                    render_result(result)
                except Exception as exc:
                    st.error(f"MNIST 예측 실패: {exc}")

with tab_cats_dogs:
    st.header("Cats / Dogs 분류와 예측")
    st.write("고양이 또는 강아지 이미지를 업로드하면 선택한 모델로 분류합니다.")

    variant_labels = {
        key: value["label"]
        for key, value in CAT_DOG_MODELS.items()
    }
    selected_label = st.selectbox(
        "모델 선택",
        options=list(variant_labels.values()),
        index=0,
    )
    selected_variant = next(
        key for key, label in variant_labels.items() if label == selected_label
    )

    uploaded_animal = st.file_uploader(
        "고양이/개 이미지 업로드",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="catdog-uploader",
    )

    if uploaded_animal:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("입력 이미지")
            st.image(Image.open(uploaded_animal), use_container_width=True)

        with right:
            if st.button("Cats/Dogs 예측 실행", type="primary"):
                try:
                    result = predict_uploaded_image(
                        "cats-dogs",
                        uploaded_animal,
                        variant=selected_variant,
                    )
                    render_result(result)
                except Exception as exc:
                    st.error(f"Cats/Dogs 예측 실패: {exc}")

st.divider()
st.caption(f"Project root: `{Path.cwd()}`")

