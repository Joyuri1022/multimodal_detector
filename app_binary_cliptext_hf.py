from __future__ import annotations

import gc
import os
import re
from pathlib import Path

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizer, CLIPVisionModel


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "streamlit_demo"
TEXT_MODEL_NAME = "roberta-base"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
HIDDEN_DIM = 256
DROPOUT = 0.1
FUSION_TYPE = "concat"
MAX_LENGTH = 128
IMAGE_SIZE = 224
TEXT_COLOR = "#303030"
ACCENT_COLOR = "#0a4c86"
SCRAPED_THRESHOLD = 0.30

MODEL_CONFIGS = {
    "fakeddit": {
        "display_name": "Trained on Fakeddit",
        "path": SOURCE_DIR / "late_fusion_model_v2_1_cleaned_l3l4.pt",
        "architecture": "late_fusion_bert_resnet",
        "tokenizer_name": "bert-base-uncased",
    },
    "scraped": {
        "display_name": "Train on Scraped Data",
        "path": SOURCE_DIR / "roberta_cliptext_main" / "best_model.pt",
        "architecture": "roberta_clip",
    },
}

LABEL_MAP = {
    0: "True",
    1: "Fake",
}

_ACTIVE_MODEL_KEY: str | None = None
_ACTIVE_BUNDLE: dict | None = None


def get_runtime_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


def get_model_config_value(model_config: dict, key: str) -> str | None:
    direct_value = model_config.get(key)
    if direct_value:
        return str(direct_value)

    env_key_name = model_config.get(f"{key}_env")
    if env_key_name:
        return get_runtime_setting(str(env_key_name))

    return None


def resolve_checkpoint_path(model_config: dict) -> Path:
    local_path = model_config.get("path")
    if local_path:
        local_path = Path(local_path)
        if local_path.exists():
            return local_path

    repo_id = get_model_config_value(model_config, "hf_repo_id")
    filename = get_model_config_value(model_config, "hf_filename")
    if repo_id and filename:
        token_name = model_config.get("hf_token_secret")
        token = get_runtime_setting(str(token_name)) if token_name else None
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
            )
        )

    raise FileNotFoundError(
        "Model checkpoint not found locally, and no Hugging Face download configuration was provided."
    )


def get_decision_threshold(model_key: str) -> float:
    return SCRAPED_THRESHOLD if model_key == "scraped" else 0.5


def describe_prediction_strength(predicted_label: int, fake_probability: float, threshold: float) -> tuple[float, str]:
    if predicted_label == 1:
        margin = fake_probability - threshold
    else:
        margin = threshold - fake_probability

    if margin < 0.05:
        level = "Weak"
    elif margin < 0.15:
        level = "Moderate"
    elif margin < 0.30:
        level = "Strong"
    else:
        level = "Very Strong"

    target_name = LABEL_MAP[predicted_label]
    return margin, f"{level} {target_name}"


def preprocess_text(text: str) -> str:
    cleaned = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"&amp;", " and ", cleaned)
    cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if len(words) <= MAX_LENGTH:
        return cleaned
    return " ".join(words[:MAX_LENGTH])


def preprocess_uploaded_image(image: Image.Image) -> Image.Image:
    processed = ImageOps.exif_transpose(image).convert("RGB")
    processed.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    offset_x = (IMAGE_SIZE - processed.width) // 2
    offset_y = (IMAGE_SIZE - processed.height) // 2
    canvas.paste(processed, (offset_x, offset_y))
    return canvas


class RobertaClipClassifier(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        vision_model_name: str,
        num_labels: int,
        hidden_dim: int,
        dropout: float,
        fusion_type: str = "concat",
        classifier_variant: str = "layernorm_linear",
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.image_projection = nn.Linear(self.image_encoder.config.hidden_size, hidden_dim)
        self.fusion_type = fusion_type
        self.classifier_variant = classifier_variant

        if fusion_type == "concat":
            classifier_input_dim = hidden_dim * 2
            self.gate = None
        elif fusion_type == "gated":
            classifier_input_dim = hidden_dim
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        elif fusion_type == "gated_interaction":
            classifier_input_dim = hidden_dim * 3
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        if classifier_variant == "layernorm_linear":
            self.classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        elif classifier_variant == "linear_only":
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            raise ValueError(f"Unsupported classifier_variant: {classifier_variant}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_embedding = self.text_projection(text_outputs.last_hidden_state[:, 0])
        image_embedding = self.image_projection(image_outputs.pooler_output)

        if self.fusion_type == "concat":
            fused = torch.cat([text_embedding, image_embedding], dim=-1)
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([text_embedding, image_embedding], dim=-1))
            fused = gate * text_embedding + (1.0 - gate) * image_embedding
        else:
            gate = self.gate(torch.cat([text_embedding, image_embedding], dim=-1))
            gated_fusion = gate * text_embedding + (1.0 - gate) * image_embedding
            interaction_product = text_embedding * image_embedding
            interaction_diff = torch.abs(text_embedding - image_embedding)
            fused = torch.cat([gated_fusion, interaction_product, interaction_diff], dim=-1)

        return self.classifier(fused)


class LateFusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel(BertConfig())
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()
        self.text_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        text_feat = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        img_feat = self.resnet(image)
        text_feat = self.text_proj(text_feat)
        img_feat = self.image_proj(img_feat)
        fused = torch.cat([text_feat, img_feat], dim=1)
        return self.classifier(fused)


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
    return checkpoint


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int | str]:
    text_hidden_dim = int(state_dict["text_projection.weight"].shape[0])
    image_hidden_dim = int(state_dict["image_projection.weight"].shape[0])
    if text_hidden_dim != image_hidden_dim:
        raise RuntimeError(
            "The checkpoint uses different hidden sizes for text and image projections, "
            "which this demo does not currently support."
        )

    final_classifier_key = max(
        (key for key in state_dict if key.startswith("classifier.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    num_labels = int(state_dict[final_classifier_key].shape[0])

    first_classifier_weight = state_dict.get("classifier.0.weight")
    if first_classifier_weight is None:
        raise RuntimeError("Unable to infer classifier structure from checkpoint.")

    if first_classifier_weight.ndim == 1:
        classifier_variant = "layernorm_linear"
        hidden_dim = int(state_dict["classifier.1.weight"].shape[0])
        layernorm_dim = int(first_classifier_weight.shape[0])
    elif first_classifier_weight.ndim == 2:
        classifier_variant = "linear_only"
        hidden_dim = int(first_classifier_weight.shape[0])
        layernorm_dim = int(first_classifier_weight.shape[1])
    else:
        raise RuntimeError("Unsupported classifier.0.weight shape in checkpoint.")

    if layernorm_dim == text_hidden_dim * 2:
        fusion_type = "concat"
    elif layernorm_dim == text_hidden_dim:
        fusion_type = "gated"
    elif layernorm_dim == text_hidden_dim * 3:
        fusion_type = "gated_interaction"
    else:
        raise RuntimeError(
            f"Unable to infer fusion type from classifier input dimension {layernorm_dim} "
            f"and hidden size {text_hidden_dim}."
        )

    return {
        "hidden_dim": hidden_dim,
        "num_labels": num_labels,
        "classifier_variant": classifier_variant,
        "fusion_type": fusion_type,
    }


def load_inference_bundle(model_key: str):
    global _ACTIVE_MODEL_KEY, _ACTIVE_BUNDLE

    if _ACTIVE_MODEL_KEY == model_key and _ACTIVE_BUNDLE is not None:
        return _ACTIVE_BUNDLE

    if _ACTIVE_BUNDLE is not None:
        _ACTIVE_BUNDLE.clear()
        _ACTIVE_BUNDLE = None
        gc.collect()

    model_config = MODEL_CONFIGS[model_key]
    model_path = resolve_checkpoint_path(model_config)

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    architecture = model_config["architecture"]

    if architecture == "late_fusion_bert_resnet":
        tokenizer = BertTokenizer.from_pretrained(model_config["tokenizer_name"])
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        model = LateFusionModel()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        _ACTIVE_MODEL_KEY = model_key
        _ACTIVE_BUNDLE = {
            "architecture": architecture,
            "tokenizer": tokenizer,
            "image_transform": image_transform,
            "model": model,
        }
        return _ACTIVE_BUNDLE

    if architecture != "roberta_clip":
        raise RuntimeError(f"Unsupported architecture: {architecture}")

    inferred = infer_model_config(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    model = RobertaClipClassifier(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        num_labels=int(inferred["num_labels"]),
        hidden_dim=int(inferred["hidden_dim"]),
        dropout=DROPOUT,
        fusion_type=str(inferred["fusion_type"]),
        classifier_variant=str(inferred["classifier_variant"]),
    )

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = set(incompatible.unexpected_keys)
    allowed_unexpected = {"class_weights", "loss_fn.weight"}
    disallowed_unexpected = unexpected_keys - allowed_unexpected
    if incompatible.missing_keys or disallowed_unexpected:
        raise RuntimeError(
            f"Model checkpoint is not compatible with the binary demo app for model '{model_key}' "
            f"from '{model_path}'. "
            f"Missing keys: {incompatible.missing_keys}; "
            f"Unexpected keys: {sorted(disallowed_unexpected)}"
        )

    model.eval()
    _ACTIVE_MODEL_KEY = model_key
    _ACTIVE_BUNDLE = {
        "architecture": architecture,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "model": model,
    }
    return _ACTIVE_BUNDLE


def predict(model_key: str, text: str, image: Image.Image) -> tuple[int, dict[str, float], float, float, str]:
    bundle = load_inference_bundle(model_key)
    architecture = bundle["architecture"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    threshold = get_decision_threshold(model_key)

    if architecture == "late_fusion_bert_resnet":
        image_transform = bundle["image_transform"]
        encoded_text = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        image_tensor = image_transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(
                input_ids=encoded_text["input_ids"],
                attention_mask=encoded_text["attention_mask"],
                image=image_tensor,
            ).squeeze(-1)
            fake_probability = torch.sigmoid(logits)[0]

        probs = {
            "True": float(1.0 - fake_probability.item()),
            "Fake": float(fake_probability.item()),
        }
        predicted_label = 1 if probs["Fake"] >= threshold else 0
        margin, strength = describe_prediction_strength(
            predicted_label,
            probs["Fake"],
            threshold,
        )
        return predicted_label, probs, threshold, margin, strength

    image_processor = bundle["image_processor"]
    encoded_text = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded_image = image_processor(images=[image], return_tensors="pt")
    model_inputs = {**encoded_text, **encoded_image}

    with torch.no_grad():
        logits = model(**model_inputs)
        probabilities = torch.softmax(logits, dim=-1)[0]

    probs = {
        LABEL_MAP[index]: float(probabilities[index].item())
        for index in range(len(LABEL_MAP))
    }
    predicted_label = 1 if probs["Fake"] >= threshold else 0
    margin, strength = describe_prediction_strength(
        predicted_label,
        probs["Fake"],
        threshold,
    )
    return predicted_label, probs, threshold, margin, strength


def render_prediction_card(
    model_name: str,
    predicted_label: int,
    probabilities: dict[str, float],
    threshold: float,
    margin: float,
    strength: str,
) -> None:
    predicted_name = LABEL_MAP[predicted_label]
    confidence = probabilities[predicted_name]
    fake_probability = probabilities["Fake"]
    probability_rows = "".join(
        f"""
        <div class="probability-row">
            <span>{label_name}</span>
            <strong>{probabilities[label_name]:.4f}</strong>
        </div>
        """
        for label_name in LABEL_MAP.values()
    )

    st.markdown(
        f"""
        <div class="model-title">{model_name}</div>
        <div class="result-card">
            <strong>Predicted label:</strong> {predicted_label}<br/>
            <strong>Meaning:</strong> {predicted_name}<br/>
            <strong>Confidence:</strong> {confidence:.4f}<br/>
            <strong>Decision threshold (Fake):</strong> {threshold:.2f}<br/>
            <strong>Fake probability margin:</strong> {margin:.4f}<br/>
            <strong>Prediction strength:</strong> {strength}<br/>
            <strong>Decision rule:</strong> Fake probability {fake_probability:.4f}
            {">=" if predicted_label == 1 else "<"} {threshold:.2f}
        </div>
        <div class="probability-card">
            {probability_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )
