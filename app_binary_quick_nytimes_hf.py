from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

import app_binary_cliptext_hf as base


APP_DIR = Path(__file__).resolve().parent
SOURCE_DIR = APP_DIR.parent / "streamlit_demo"
SCRAPED_MODEL_PATH = SOURCE_DIR / "multimodal_roberta_quick_nytimes_binary_3way01" / "best_model.pt"
FAKEDDIT_ROBERTA_CLIP_MODEL_PATH = SOURCE_DIR / "roberta_clip_fusion_fakeddit_v1" / "roberta_clip_fusion.pt"

base.SCRAPED_THRESHOLD = 0.50
base.MODEL_CONFIGS = {
    **base.MODEL_CONFIGS,
    "fakeddit": {
        "display_name": "Trained on Fakeddit",
        "path": FAKEDDIT_ROBERTA_CLIP_MODEL_PATH,
        "hf_repo_id_env": "FAKEDDIT_HF_REPO_ID",
        "hf_filename_env": "FAKEDDIT_HF_FILENAME",
        "hf_token_secret": "HF_TOKEN",
        "architecture": "roberta_clip",
    },
    "scraped": {
        "display_name": "Train on Scraped Data",
        "path": SCRAPED_MODEL_PATH,
        "hf_repo_id_env": "SCRAPED_HF_REPO_ID",
        "hf_filename_env": "SCRAPED_HF_FILENAME",
        "hf_token_secret": "HF_TOKEN",
        "architecture": "roberta_clip",
    },
}


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Binary Demo (Quick NYTimes)",
        page_icon=":mag:",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: #ffffff;
            color: {base.TEXT_COLOR};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .hero {{
            border: 1px solid #dbe2ea;
            border-radius: 20px;
            padding: 1.5rem 1.75rem;
            background: linear-gradient(135deg, #f8fbff 0%, #eef6fb 100%);
            margin-bottom: 1.25rem;
        }}
        .hero h1 {{
            color: {base.ACCENT_COLOR};
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
        }}
        .hero p {{
            margin: 0;
            line-height: 1.6;
            color: {base.TEXT_COLOR};
        }}
        .result-card, .probability-card {{
            border: 1px solid #dbe2ea;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: #fbfdff;
            margin-top: 0.85rem;
        }}
        .model-title {{
            color: {base.ACCENT_COLOR};
            font-weight: 700;
            font-size: 1.05rem;
            margin-top: 0.5rem;
        }}
        .probability-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.2rem 0;
        }}
        div.stButton > button {{
            background: linear-gradient(135deg, #ff7a18 0%, #e4572e 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.6rem 1rem;
            width: 100%;
        }}
        div.stButton > button:hover {{
            background: linear-gradient(135deg, #ff8b35 0%, #ef653a 100%);
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Multimodal Binary Demo</h1>
            <p>Upload an image, enter a news-style caption, and compare predictions from the available models.
            This cloud-ready version pulls checkpoints from Hugging Face when they are not available locally.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        preferred_order = ["scraped", "fakeddit"]
        model_options = [key for key in preferred_order if key in base.MODEL_CONFIGS]
        model_options.extend(key for key in base.MODEL_CONFIGS if key not in model_options)
        model_key = st.selectbox(
            "Model",
            options=model_options,
            format_func=lambda key: base.MODEL_CONFIGS[key]["display_name"],
        )
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        text_input = st.text_area("Caption / text", height=140)
        run_inference = st.button("Run Prediction", type="primary")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        else:
            image = None
            st.info("Upload an image to begin.")

        if text_input.strip():
            st.text_area(
                "Processed text preview",
                value=base.preprocess_text(text_input),
                height=140,
                disabled=True,
            )

    with col2:
        st.subheader("Prediction")
        if not run_inference:
            st.info("Upload an image, enter text, and click Run Prediction.")
        elif image is None or not text_input.strip():
            st.warning("Provide both an image and text to run inference.")
        else:
            processed_text = base.preprocess_text(text_input)
            processed_image = base.preprocess_uploaded_image(image)
            try:
                predicted_label, probabilities, threshold, margin, strength = base.predict(
                    model_key,
                    processed_text,
                    processed_image,
                )
                base.render_prediction_card(
                    base.MODEL_CONFIGS[model_key]["display_name"],
                    predicted_label,
                    probabilities,
                    threshold,
                    margin,
                    strength,
                )
            except Exception as exc:
                st.error(str(exc))
                st.info(
                    "Check that your Hugging Face repo contains the exact checkpoint file expected for the selected model."
                )


if __name__ == "__main__":
    main()
