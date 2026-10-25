"""Gradio web app for multilingual fake news detection.

Runs locally (`python app.py`) or on Hugging Face Spaces (the cloud part of the project).

Environment variables:
    MODEL_ID   Hugging Face model id (e.g. "your-username/fakenews-xlmr") or a local
               folder such as "results/xlmr/final". Required.
"""
import os
import re

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "results/xlmr/final")
MAX_LEN = 128   # same truncation length as used in training and evaluation

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

EXAMPLES = [
    ["Scientists confirm that drinking hot water every hour cures all forms of cancer, doctors hate this."],
    ["The Reserve Bank of India kept the repo rate unchanged at its latest monetary policy meeting."],
    ["सरकार ने घोषणा की है कि कल से सभी बैंक एक महीने के लिए बंद रहेंगे, यह संदेश सबको भेजें।"],
]


def detect_lang(text: str) -> str:
    deva = len(re.findall(r"[ऀ-ॿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "Hindi" if deva > latin else "English"


@torch.no_grad()
def predict(text: str):
    text = (text or "").strip()
    if len(text) < 20:
        return "Please enter a longer piece of text (at least 20 characters).", {}, ""
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().tolist()
    scores = {"Real": probs[0], "Fake": probs[1]}
    verdict = "Likely FAKE" if probs[1] >= 0.5 else "Likely REAL"
    return verdict, scores, detect_lang(text)


with gr.Blocks(title="Multilingual Fake News Detector") as demo:
    gr.Markdown(
        "# Multilingual Fake News Detector\n"
        "Paste a news headline or article in **English or Hindi**. "
        "The model (XLM-RoBERTa, fine-tuned) estimates whether it is fake or real.\n\n"
        "*This is a research/academic demo. It judges writing patterns, not facts, "
        "so always verify important news with trusted sources.*"
    )
    inp = gr.Textbox(lines=6, label="News text", placeholder="Paste news text here...")
    btn = gr.Button("Check", variant="primary")
    with gr.Row():
        verdict = gr.Textbox(label="Verdict")
        lang = gr.Textbox(label="Detected language")
    conf = gr.Label(label="Confidence", num_top_classes=2)
    btn.click(predict, inputs=inp, outputs=[verdict, conf, lang])
    gr.Examples(EXAMPLES, inputs=inp)

if __name__ == "__main__":
    demo.launch()
