#%%writefile app.py

import os
import time
from datetime import datetime
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple
import langdetect
# Optional ML imports
try:
    from transformers import pipeline, Pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from pydub import AudioSegment
import altair as alt

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="🕵🏻Speech Threat Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Styling header
st.markdown("""
    <style>
    .big-font { font-size:32px; font-weight:700; }
    .muted { color: #9AA0A6; }
    .card { background: linear-gradient(135deg, rgba(10,25,47,0.95), rgba(23,43,77,0.95)); padding: 18px; border-radius: 12px; color: white; box-shadow: 0 6px 30px rgba(8,10,20,0.45); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="card"><span class="big-font">Speech Threat Detection Dashboard</span> <span class="muted"> — upload audio or paste text </span></div>', unsafe_allow_html=True)

UPLOAD_DIR = Path("uploads")
DB_CSV = Path("db.csv")
UPLOAD_DIR.mkdir(exist_ok=True)
if not DB_CSV.exists():
    pd.DataFrame(columns=["timestamp","filename","mode","transcription","predicted_label","scores"]).to_csv(DB_CSV, index=False)

LABELS = [
    "physical threat",
    "cyber threat",
    "hate speech",
    "political extremist threat",
    "neutral"
]

LABEL_MAP = {
    "LABEL_0": "hate speech",
    "LABEL_1": "self-harm",
    "LABEL_2": "cyber threat",
    "LABEL_3": "neutral / daily life",
    "LABEL_4": "physical threat",
    "LABEL_5": "political extremist threat"
}

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def save_audio_file(uploaded_file) -> Path:
    filename = f"{int(time.time())}_{uploaded_file.name}"
    out_path = UPLOAD_DIR / filename
    with open(out_path, "wb") as f:
        f.write(uploaded_file.read())
    return out_path

def normalize_audio_to_wav(path: Path) -> Path:
    sound = AudioSegment.from_file(path)
    sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    wav_path = path.with_suffix(".wav")
    sound.export(wav_path, format="wav")
    return wav_path

@st.cache_resource(show_spinner=False)
def get_asr_pipeline() -> Tuple[str, "Pipeline"]:
    """Load Hugging Face Whisper ASR model"""
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")
    return ("hf", asr)
# def hf_transcribe_with_pipeline(asr_pipeline: Pipeline, path: Path) -> str:
#     output = asr_pipeline(str(path))
#     return output["text"].strip() if isinstance(output, dict) else str(output).strip()
def get_classifier_pipeline(model_name: str):
    """Load zero-shot or custom classifier"""
    try:
        if model_name == "custom_xlm_roberta":
            classifier = pipeline(
                "text-classification",
                model="/content/drive/MyDrive/xlm_roberta_multilingual_classifier/final",
                tokenizer="/content/drive/MyDrive/xlm_roberta_multilingual_classifier/final",
                return_all_scores=True
            )
        else:
            classifier = pipeline("zero-shot-classification", model=model_name)
        return classifier
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

def hf_transcribe_with_pipeline(asr_pipeline_tuple, path: Path, lang_choice: str = "Auto") -> str:
    """Transcribe audio with Whisper and restrict to English/Urdu"""
    asr_pipeline = asr_pipeline_tuple[1] # Extract the pipeline from the tuple

    lang_token = None
    if lang_choice == "English only":
        lang_token = "<|en|>"
    elif lang_choice == "Urdu only":
        lang_token = "<|ur|>"

    kwargs = {"generate_kwargs": {"language": lang_token}} if lang_token else {}
    output = asr_pipeline(str(path), **kwargs)
    text = output["text"].strip() if isinstance(output, dict) else str(output).strip()

    # Restrict to English or Urdu only
    try:
        detected = langdetect.detect(text)
        if detected not in ["en", "ur"]:
            return "[❌ Unsupported language detected — please use Urdu or English.]"
    except Exception:
        pass

    return text

def classify_text(text: str, classifier: Pipeline, labels: List[str]) -> Dict:
    try:
        if "zero-shot" in classifier.task:
            # For zero-shot models like RoBERTa or BART
            result = classifier(text, labels, multi_label=False, hypothesis_template="This text is about {}.")
            labels_out, scores_out = result["labels"], result["scores"]
        else:
            # For custom fine-tuned text classification models
            outputs = classifier(text)
            # Handle both single and batch outputs
            if isinstance(outputs, list):
                outputs = outputs[0]  # unwrap batch
            if isinstance(outputs, list):
                # Handle return_all_scores=True (list of dicts)
                labels_out = [LABEL_MAP.get(o["label"], o["label"]) for o in outputs]
                scores_out = [o["score"] for o in outputs]
            else:
                # Single dict output
                labels_out = [LABEL_MAP.get(outputs["label"], outputs["label"])]
                scores_out = [outputs["score"]]
        # Pick the top scoring label
        top_label = labels_out[scores_out.index(max(scores_out))]
        return {"label": top_label, "scores": dict(zip(labels_out, scores_out))}
    except Exception as e:
        st.error(f"Classification failed: {e}")
        return {"label": "neutral", "scores": {}}


def log_to_db(record: Dict):
    df = pd.read_csv(DB_CSV)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(DB_CSV, index=False)

# -----------------------------------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

asr_pipeline_tuple = get_asr_pipeline() # Get the tuple

asr_language = st.sidebar.selectbox(
    "Transcription Language Restriction",
    ["Auto", "English only", "Urdu only"],
    help="Restrict transcription to English or Urdu only"
)
if asr_pipeline_tuple[0] == "hf": # Check the method
    st.sidebar.markdown("**ASR Method:** `hf`")
else:
    st.sidebar.markdown("**ASR Method:** `none` (Hugging Face models not available)")

# Model selection mode
model_type = st.sidebar.radio("Select Model Type", ["Zero-shot Models", "Custom Models"])

if model_type == "Zero-shot Models":
    model_choices = {
        "✔Pretrained-RoBERTa": "roberta-large-mnli",
        "✔Pretrained-MultiClassification": "facebook/bart-large-mnli",
        "✔XLM-R": "joeddav/xlm-roberta-large-xnli"
    }
else:
    model_choices = {
        "✔IB: XLM-R-Fine-tuned": "/content/drive/MyDrive/xlm_roberta_multilingual_classifier/final"
    }

model_display = st.sidebar.selectbox("Choose a Model", list(model_choices.keys()))
model_path = model_choices[model_display]

with st.sidebar:
    st.markdown("---")
    st.markdown("**Active Labels:**")
    for lbl in LABELS:
        st.markdown(f"- {lbl}")

# Load classifier
with st.spinner("Loading model..."):
    try:
        if model_type == "Custom Models":
            classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, return_all_scores=True)
        else:
            classifier = pipeline("zero-shot-classification", model=model_path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

# -----------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------
st.markdown("""
    <style>
    .subtitle {color:#999;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="subtitle">Upload or enter text to detect threat categories</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["⏳ Processing", "📊 Analysis"])

# -----------------------------------------------------------
# TAB 1: PROCESSING
# -----------------------------------------------------------
with tab1:
    input_mode = st.radio("Input mode", ["Upload audio", "Paste text"])
    transcription_text = ""
    saved_file_path = None

    if input_mode == "Upload audio":
        uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","m4a","flac","ogg"])
        if uploaded_file:
            saved_path = save_audio_file(uploaded_file)
            saved_file_path = saved_path
            st.audio(saved_path)
            wav_path = normalize_audio_to_wav(saved_path)
            st.info("Transcribing audio...")
            try:
                if asr_pipeline_tuple[0] == "hf": # Check the method
                    transcription_text = hf_transcribe_with_pipeline(asr_pipeline_tuple, wav_path) # Pass the tuple
                else:
                    st.warning("No ASR available.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
    else:
        transcription_text = st.text_area("Enter or paste text", height=180)

    if transcription_text:
        st.markdown("### Transcription")
        txt = st.text_area("Editable text", value=transcription_text, height=180)
        if st.button("📝 Classify"):
            with st.spinner("Analyzing text..."):
                result = classify_text(txt, classifier, LABELS)
                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "filename": saved_file_path.name if saved_file_path else "text_input",
                    "mode": "audio" if saved_file_path else "text",
                    "transcription": txt,
                    "predicted_label": result["label"],
                    "scores": result["scores"]
                }
                log_to_db(record)
            st.success(f"**Predicted Category:** {result['label']}")
            df_scores = pd.DataFrame(result["scores"].items(), columns=["Label", "Score"]).sort_values("Score", ascending=False)
            st.bar_chart(df_scores.set_index("Label"))

# -----------------------------------------------------------
# TAB 2: ANALYSIS
# -----------------------------------------------------------
with tab2:
    st.subheader("📈📊 Analytical Overview")
    if not DB_CSV.exists() or os.path.getsize(DB_CSV) == 0:
        st.info("No data available yet. Run a few classifications first.")
    else:
        df = pd.read_csv(DB_CSV)
        if df.empty:
            st.info("No records yet.")
        else:
            st.metric("Total Records", len(df))
            cat_counts = df["predicted_label"].value_counts().reset_index()
            cat_counts.columns = ["Label", "Count"]
            chart = alt.Chart(cat_counts).mark_bar().encode(
                x="Label:N", y="Count:Q", tooltip=["Label", "Count"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### Recent Entries")
            st.dataframe(df.sort_values("timestamp", ascending=False).head(30))

            st.markdown("### Upload Trends Over Time")
            df["ts_day"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
            ts = df.groupby(["ts_day","predicted_label"]).size().reset_index(name="count")
            line_chart = alt.Chart(ts).mark_line(point=True).encode(
                x="ts_day:T", y="count:Q", color="predicted_label:N"
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Log", csv, "threat_log.csv", "text/csv")

st.markdown("---")
st.caption("© 2025 — Intelligence Threat Detection Suite. Built for multilingual zero-shot NLP analysis.")
