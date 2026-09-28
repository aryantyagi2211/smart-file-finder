"""
indexing/summarizer.py

Generates a one-paragraph summary of a text document's content using
Qwen3-1.7B, run locally on GPU (4-bit quantized). This is an INDEXING-TIME
only feature -- summaries are generated once when a document is indexed
and stored as an additional searchable entry, never generated at search
time (a single generation call takes several seconds, far too slow for
live search).

This fills a real gap: individual chunks capture specific passages, but
nothing previously represented "what is this whole document about" as a
single searchable unit -- useful for broad/overview-style queries that
don't match any single chunk well.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_model = None
_tokenizer = None

MODEL_NAME = "Qwen/Qwen3-1.7B"


def get_model_and_tokenizer():
    """Load the reasoning model once and reuse it (loading is slow)."""
    global _model, _tokenizer
    if _model is None:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quant_config, device_map="cuda"
        )
    return _model, _tokenizer


def summarize_document(chunks, max_chars_for_context=2000):
    """
    Generate a one-paragraph summary of a document from its chunks.
    Only the first max_chars_for_context worth of content is used as
    input, to keep generation fast and within a reasonable prompt size --
    a document's opening content is usually representative enough for a
    high-level summary.
    """
    if not chunks:
        return ""

    model, tokenizer = get_model_and_tokenizer()

    combined_text = " ".join(chunks)[:max_chars_for_context]

    prompt = (
        f"Summarize what this document is about in one short paragraph "
        f"(2-3 sentences), focusing on its main topic and purpose:\n\n{combined_text}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    del inputs, outputs
    torch.cuda.empty_cache()

    return response.strip()