from __future__ import annotations

import streamlit as st
from PIL import Image

import app_binary_cliptext_hf as base


base.SCRAPED_THRESHOLD = 0.50
PAGE_BG = "#f4f8fb"
TEXT_COLOR = "#153047"
ACCENT_COLOR = "#176087"
CARD_BG = "#ffffff"
CARD_BORDER = "#b8d4e5"
HERO_BG = "linear-gradient(135deg, #eff7fc 0%, #dceefa 100%)"
BUTTON_BG = "linear-gradient(135deg, #1f7aa8 0%, #14597d 100%)"
BUTTON_HOVER_BG = "linear-gradient(135deg, #2a8cbd 0%, #1a6b94 100%)"

base.MODEL_CONFIGS = {
    "scraped": {
        "display_name": "Train on Scraped Data",
        "path": base.SOURCE_DIR / "multimodal_roberta_quick_nytimes_binary_3way01" / "best_model.pt",
        "hf_repo_id_env": "SCRAPED_HF_REPO_ID",
        "hf_filename_env": "SCRAPED_HF_FILENAME",
        "hf_token_secret": "HF_TOKEN",
        "architecture": "roberta_clip",
    },
}


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Binary Demo (Scraped)",
        page_icon=":mag:",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(31, 122, 168, 0.10), transparent 28%),
                linear-gradient(180deg, {PAGE_BG} 0%, #fbfdff 100%);
            color: {TEXT_COLOR};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .hero {{
            border: 1px solid {CARD_BORDER};
            border-radius: 24px;
            padding: 1.6rem 1.85rem;
            background: {HERO_BG};
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 50px rgba(23, 96, 135, 0.10);
        }}
        .hero h1 {{
            color: {ACCENT_COLOR};
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
        }}
        .hero p {{
            margin: 0;
            line-height: 1.6;
            color: {TEXT_COLOR};
        }}
        .result-card, .probability-card {{
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: {CARD_BG};
            margin-top: 0.85rem;
            box-shadow: 0 10px 30px rgba(23, 96, 135, 0.08);
        }}
        .model-title {{
            color: {ACCENT_COLOR};
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
            background: {BUTTON_BG};
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.6rem 1rem;
            width: 100%;
        }}
        div.stButton > button:hover {{
            background: {BUTTON_HOVER_BG};
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Scraped Newsroom Model</h1>
            <p>This version uses the scraped-data checkpoint and a cool editorial palette so it is easy to distinguish
            from the Fakeddit deployment while staying within Streamlit Cloud memory limits.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
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
                    "scraped",
                    processed_text,
                    processed_image,
                )
                base.render_prediction_card(
                    base.MODEL_CONFIGS["scraped"]["display_name"],
                    predicted_label,
                    probabilities,
                    threshold,
                    margin,
                    strength,
                )
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
