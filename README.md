Manual_Transcriptions_Ground_Truth: hello..hello … Ankit G, yes speaking. Can u give rate for EUR booking, EUR Rupee. Ok u are calling from which company? Ridhi speciality, I need to know current rate for eur for export, Oh!! its Mitesh on the other side, ok, yes.. Current rate for export EUR EUR current rate is 95.56 ok i want to forward sell 1lac EUR for maturity 31Aug, 31 of Aug, ok. Aug month end last good date is 29Aug, no problem no problem, 65, 65,  let me check the premium whether its voltile today, net rate would be sir 97.15 ok please go ahead. ya.  Also sir one more thing like regarding your forward booking, there some doucmetation is pending at your end which Prasad, your business RM would have sent, we are basically increasing the tenure from 6 to 12months and also limit increase to 15cr, so some finding was pending which Suresh said will get it done after checking with your company secretary, so I am just following up regarding that, I am in europe right now, ohh ok ok let me check with Suresh and come back ok.. this is booked ok ok thank you

Gemini: Yeah. Hello. Ankit ji. Yes, I am speaking. You can tell the Euro exchange rate? Euro-Rupee. Yes, from which company are you speaking, sir? Vidhi Vidhi Specialty. I need to know the current rate for Euro for export. Oh, Mittal sir, okay. Yes. Yes. Yes. Yes, sir. The current rate for export Euro. Euro current rate is ninety-five point sixty-six. Okay, I want to forward sell one lakh Euro. For maturity thirty-first of August. Thirty-first of August, okay. August month's last good date is twenty-ninth actually. Twenty-ninth August. No problem. No problem. Yes. Yes. Let me just check the premium whether it's volatile today. Net it would be sir ninety-seven point one-five. Okay. Please go ahead. Yes. Also, sir, one more thing, like regarding your forward booking, there's some documentation pending at your end which probably your business head would have sent. We are basically increasing the tenure from six to twelve months and also limit increased to fifteen cr. So some signing was pending, yeah, which Suresh said he will get it done after checking with your company secretary. So I just following up regarding that. See, I'm in Europe right now. Oh, okay, okay. Let me check with Suresh and come back to you. Yeah, yeah, okay, sir. This here is booked, sir. Okay. Okay, thanks.

Offline_Whisper: Hello? Hello? Yes, I am speaking. Can you tell me the price of Euro? Euro rupee? Yes, which company are you speaking from, sir? Vidhi, Vidhi Specialty. I need to know the current rate for Euro for export. Oh, you are talking from METEF? Yes. current rate for export euro you know current rate is 95.66. Okay, I want to forward sell 1 lakh Euro for maturity 31st of August. Hello. Sir, yes, tell me. August month end last good date is 29th, 29th of August. No problem. Now I am getting 65, so I am having, I'll just take the premium because it's volatile today. Net rate would be sir 97.15 ok please go ahead also sir one more thing regarding your forward booking, there is some documentation pending at your end which Prasad your business RM would have sent. We are basically increasing the tenor from 6 to 12 months and also limit increased to 15th year. So some signing was pending which Suresh said he will get it done after checking with your company secretary. So I just following up regarding that. See, I am in Europe right now. Oh, okay, okay. Let me check with Suresh and come back to you. Yeah, okay sir. This will be good sir. Okay, yeah. Okay, thank you.

STT _V1: I need to know the current rate for Euro for export. current rate for export euro you know current rate is a 95.66 okay I want to forward sell 1 lakh Euro for maturity 31st of August August month and last good date is 29th actually, 29th August, no problem, no problem, now I 65 Wait, I'll just take the premium with it. It's volatile today Netted would be 97 point one five My okay please go ahead Also sir one more thing like regarding your forward booking, there is some documentation pending at your end which Prasad your business RM would have sent. We are basically increasing the tenure from 6 to 12 months and also limit increase to 15th year so some signing was pending with Suresh said he will get it done after checking See, I am in Europe right now, let me check with Suresh and come back to you.

STT_V2: Hello?   Uh, you look at seeing the best idea. Which company are you talking about sir? Vidhi Specialty. I need to know the current rate for Euro for export Oh, Mitre Sir is speaking. Yes The current trade for export euro. Euro current rate is 95.66 I want to forward sell 1 lakh euro. for maturity 31st of August. The 21st of August, okay. month and last good date is 29th actually 29th hours no problem no problem uh now i think 65 All right. It will just take the premium whether it's volatile today. netted would be 97.15 And I, okay, please go ahead. Also sir one more thing like regarding your forward booking, there is some documentation pending at your end which Prasad your business RM would have sent. We are basically increasing the tenor from 6 to 12 months and also limit increase to 15th year so some signing was pending with Suresh said he will get it done after checking with your company secretary, so I'm just following up regarding that. See, I'm in Europe right now. Oh, okay, okay. Let me check with Suresh and come back to you. Yeah, yeah. Okay, sir. Okay. Thank you.

Generate a python script that calculates the following metrics for all of the rest transcriptions taking Manual_Transcriptions_Ground_Truth as base truth
"BLEU", "WER", "ROUGE-L", "SemanticSim", "BERT-P", "BERT-R", "BERT-F1", "Hybrid"

I'll upload an excel with all the above data, some of columns might be empty.


I have a sampole code below
"""
eval_full_offline.py

Requirements:
    pip install sacrebleu jiwer bert-score rouge-score sentence-transformers tqdm pandas numpy

Place your semantic model at:
    /Users/epfn119476/Documents/HDFC/Models/paraphrase-multilingual-mpnet-base-v2

CSV input: header required
    audio_file,reference,hyp1,hyp2,...

Output:
    metrics_output.csv  (one row per hypothesis)

Notes:
- BERTScore uses model "microsoft/mdeberta-v3-base" (modern, in bert_score registry).
  If running in a restricted network, pre-download/cache that model on a machine with internet.
- Semantic similarity uses the local sentence-transformers model (mpnet) at SEM_MODEL_PATH.
"""

import os
import re
import csv
from typing import List, Tuple
import math
from tqdm import tqdm
import sacrebleu
from jiwer import wer
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# ----------------------------
# Config - change if needed
# ----------------------------
INPUT_CSV = "/Users/epfn119476/Documents/Voicing_Tests_Input.csv"      # change to your CSV filename
OUTPUT_CSV = "/Users/epfn119476/Documents/Metrics_Output.csv"

# Local sentence-transformers model for semantic similarity (user confirmed path)
SEM_MODEL_PATH = "/Users/epfn119476/Documents/HDFC/Models/paraphrase-multilingual-mpnet-base-v2"

# BERTScore model (registered name). bert_score will use its registry.
# If offline, make sure this model is cached in HF cache or run once on an internet machine.
BERT_SCORE_MODEL = "microsoft/mdeberta-v3-base"

# Batching configs
BERT_BATCH_SIZE = 64

# Composite / hybrid metric weights (customize if you want)
HYBRID_WEIGHTS = {
    "semantic": 0.5,
    "wer": 0.3,       # weight applied on (1 - WER)
    "bertf1": 0.2     # weight applied on BERT-F1
}
# hybrid = semantic*W1 + (1-WER)*W2 + bertF1*W3

# ----------------------------
# Normalization utilities
# ----------------------------
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\s\.\-]")  # keep digits, allow dot/hyphen (we will remove dots later)

# Number word maps (pragmatic)
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90
}
_SCALE = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "lac": 100_000,
    "crore": 10_000_00
}

def _text_to_number_simple(tokens: List[str]) -> Tuple[str, int]:
    """
    Convert a stream of tokens starting at index 0 into a numeric string if possible.
    Returns (number_string, tokens_consumed).
    This is a pragmatic parser for common spoken forms including "point", "lakh", "crore".
    """
    i = 0
    total = 0
    current = 0
    consumed = 0
    saw_number = False
    saw_point = False
    decimal_part = ""

    while i < len(tokens):
        tok = tokens[i].lower()
        if tok == "and":
            i += 1
            consumed += 1
            continue
        if tok in _UNITS:
            current += _UNITS[tok]
            saw_number = True
            i += 1
            consumed += 1
            continue
        if tok in _TENS:
            current += _TENS[tok]
            saw_number = True
            i += 1
            consumed += 1
            if i < len(tokens) and tokens[i].lower() in _UNITS:
                current += _UNITS[tokens[i].lower()]
                i += 1
                consumed += 1
            continue
        if tok.isdigit():
            # numeric token
            current = current * (10 ** len(tok)) + int(tok) if saw_number else int(tok)
            saw_number = True
            i += 1
            consumed += 1
            continue
        if tok in _SCALE:
            scale = _SCALE[tok]
            if current == 0:
                current = 1
            total += current * scale
            current = 0
            saw_number = True
            i += 1
            consumed += 1
            continue
        if tok == "point":
            saw_point = True
            i += 1
            consumed += 1
            # parse decimal digits
            while i < len(tokens) and (tokens[i].lower() in _UNITS or re.fullmatch(r"\d+", tokens[i])):
                if re.fullmatch(r"\d+", tokens[i]):
                    decimal_part += tokens[i]
                else:
                    decimal_part += str(_UNITS[tokens[i].lower()])
                i += 1
                consumed += 1
            break
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            consumed += 1
            return (tok, consumed)
        break

    final_num = total + current
    if saw_point:
        if decimal_part == "":
            decimal_part = "0"
        number_str = f"{final_num}.{decimal_part}"
    else:
        if saw_number:
            number_str = str(final_num)
        else:
            number_str = ""
    return (number_str, consumed)

def normalize_numbers_in_text(text: str) -> str:
    """
    Replace spoken-number sequences with numeric representations when possible.
    This is pragmatic and covers common spoken forms (including Indian scales).
    """
    if not text:
        return ""
    # replace commas/colons with spaces (makes tokenization easier)
    raw = re.sub(r"[,:]", " ", text)
    tokens = re.split(r"(\s+)", raw)  # keep whitespace tokens so we can preserve spacing
    out_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isspace():
            out_tokens.append(token)
            i += 1
            continue

        # build word-level lookahead (words/digits) up to a reasonable window
        lookahead = []
        j = 0
        while i + j < len(tokens) and len(lookahead) < 12:
            t = tokens[i + j]
            if t.isspace():
                j += 1
                continue
            words = re.findall(r"[A-Za-z]+|\d+(\.\d+)?", t)
            if not words:
                break
            lookahead.extend(words)
            j += 1

        if not lookahead:
            out_tokens.append(token)
            i += 1
            continue

        conv, consumed_words = _text_to_number_simple(lookahead)
        if consumed_words and conv != "":
            out_tokens.append(conv)
            # advance i by approximately consumed_words word chunks
            remaining = consumed_words
            k = 0
            while remaining > 0 and i + k < len(tokens):
                t = tokens[i + k]
                words = re.findall(r"[A-Za-z]+|\d+(\.\d+)?", t)
                if words:
                    remaining -= len(words)
                k += 1
            i += max(1, k)
        else:
            out_tokens.append(token)
            i += 1

    joined = "".join(out_tokens)
    joined = re.sub(r"\s+", " ", joined)
    return joined.strip()

def normalize_text(text: str) -> str:
    """
    Apply normalization:
    - lowercase
    - collapse newlines
    - normalize numbers (Indian forms)
    - remove punctuation (except digits kept)
    - collapse spaces
    """
    if text is None:
        return ""
    t = text.lower()
    t = t.replace("\n", " ").replace("\r", " ")
    t = normalize_numbers_in_text(t)
    # remove punctuation except alphanumerics and spaces
    t = _RE_PUNCT.sub(" ", t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    return t.strip()

# simple sentence splitter without NLTK
def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]

# ----------------------------
# Metrics wrappers
# ----------------------------
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_bleu(reference: str, hypothesis: str) -> float:
    try:
        return float(sacrebleu.sentence_bleu(hypothesis, [reference], effective_order=True).score)
    except Exception:
        # fallback to corpus BLEU on single example
        return float(sacrebleu.corpus_bleu([hypothesis], [[reference]]).score)

def compute_wer(reference: str, hypothesis: str) -> float:
    try:
        return float(wer(reference, hypothesis))
    except Exception:
        return float('nan')

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    sc = rouge.score(reference, hypothesis)
    return float(sc['rougeL'].fmeasure)

def compute_bertscore_batch(references: List[str], candidates: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute BERTScore in batches. Uses registered model name BERT_SCORE_MODEL.
    Returns lists of P, R, F (floats).
    """
    P_all, R_all, F_all = [], [], []
    n = len(references)
    for i in range(0, n, BERT_BATCH_SIZE):
        chunk_refs = references[i:i+BERT_BATCH_SIZE]
        chunk_cands = candidates[i:i+BERT_BATCH_SIZE]
        try:
            P, R, F = bertscore_score(chunk_cands, chunk_refs,
                                      model_type=BERT_SCORE_MODEL,
                                      lang="en",
                                      rescale_with_baseline=True)
            P_all.extend([float(x) for x in P])
            R_all.extend([float(x) for x in R])
            F_all.extend([float(x) for x in F])
        except Exception as e:
            # on failure, fill NaNs and continue
            print("Warning: BERTScore batch failed:", e)
            P_all.extend([float('nan')] * len(chunk_refs))
            R_all.extend([float('nan')] * len(chunk_refs))
            F_all.extend([float('nan')] * len(chunk_refs))
    return P_all, R_all, F_all

# Semantic similarity (sentence-level average cosine with local model)
def compute_semantic_similarity(sem_model: SentenceTransformer, reference: str, hypothesis: str) -> float:
    ref_sents = split_into_sentences(reference)
    hyp_sents = split_into_sentences(hypothesis)
    max_len = max(len(ref_sents), len(hyp_sents))
    # pad
    ref_sents += [""] * (max_len - len(ref_sents))
    hyp_sents += [""] * (max_len - len(hyp_sents))

    sims = []
    for r, h in zip(ref_sents, hyp_sents):
        # if both empty, skip
        if not r and not h:
            continue
        emb_r = sem_model.encode(r, convert_to_tensor=True)
        emb_h = sem_model.encode(h, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_r, emb_h))
        sims.append(sim)
    if not sims:
        return float('nan')
    return float(np.mean(sims))

def compute_hybrid_score(semantic: float, wer_v: float, bertf1: float, weights=HYBRID_WEIGHTS) -> float:
    # semantic: [0,1], wer_v: [0,inf) where lower better, bertf1: [0,1]
    # normalize wer as (1 - wer), clipped to [0,1]
    one_minus_wer = max(0.0, 1.0 - wer_v)
    # fill NaNs gracefully
    semantic = 0.0 if semantic is None or (isinstance(semantic, float) and math.isnan(semantic)) else semantic
    bertf1 = 0.0 if bertf1 is None or (isinstance(bertf1, float) and math.isnan(bertf1)) else bertf1
    hybrid = weights['semantic'] * semantic + weights['wer'] * one_minus_wer + weights['bertf1'] * bertf1
    return float(hybrid)

# ----------------------------
# Main processing
# ----------------------------
def process_file(input_csv: str, output_csv: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: audio_file, reference")

    cols = list(df.columns)
    audio_col = cols[0]
    ref_col = cols[1]
    hyp_cols = cols[2:]

    # Load semantic model (local)
    if not os.path.isdir(SEM_MODEL_PATH):
        raise FileNotFoundError(f"Semantic model directory not found: {SEM_MODEL_PATH}")
    sem_model = SentenceTransformer(SEM_MODEL_PATH)

    rows_out = []
    # Prepare BERTScore batches
    bert_refs = []
    bert_cands = []
    bert_row_indices = []

    # iterate rows
    for ridx in tqdm(range(len(df)), desc="Preparing examples"):
        audio = df.at[ridx, audio_col]
        ref_raw = df.at[ridx, ref_col]
        ref_norm = normalize_text(ref_raw)

        for hcol in hyp_cols:
            hyp_raw = df.at[ridx, hcol] if hcol in df.columns else ""
            if hyp_raw is None or str(hyp_raw).strip() == "":
                continue
            hyp_norm = normalize_text(hyp_raw)

            # compute scores that don't need heavy models
            bleu_v = compute_bleu(ref_norm, hyp_norm)
            wer_v = compute_wer(ref_norm, hyp_norm)
            rouge_v = compute_rouge_l(ref_norm, hyp_norm)

            # semantic similarity (local mpnet)
            try:
                sem_sim = compute_semantic_similarity(sem_model, ref_norm, hyp_norm)
            except Exception as e:
                print("Warning: semantic similarity failed for audio", audio, "->", e)
                sem_sim = float('nan')

            row = {
                "audio_file": audio,
                "hypothesis_name": hcol,
                "reference_raw": ref_raw,
                "hypothesis_raw": hyp_raw,
                "reference_norm": ref_norm,
                "hypothesis_norm": hyp_norm,
                "BLEU": bleu_v,
                "WER": wer_v,
                "ROUGE-L": rouge_v,
                "SemanticSim": sem_sim,
                "BERT-P": None,
                "BERT-R": None,
                "BERT-F1": None,
                "Hybrid": None
            }
            row_index = len(rows_out)
            rows_out.append(row)

            # queue for bertscore
            bert_refs.append(ref_norm)
            bert_cands.append(hyp_norm)
            bert_row_indices.append(row_index)

    # Run BERTScore in batches
    if bert_refs:
        P_list, R_list, F_list = compute_bertscore_batch(bert_refs, bert_cands)
        # fill into rows_out
        for i, ridx in enumerate(bert_row_indices):
            p = P_list[i] if i < len(P_list) else float('nan')
            r = R_list[i] if i < len(R_list) else float('nan')
            f = F_list[i] if i < len(F_list) else float('nan')
            rows_out[ridx]["BERT-P"] = p
            rows_out[ridx]["BERT-R"] = r
            rows_out[ridx]["BERT-F1"] = f

    # compute hybrid after we have bertf1
    for r in rows_out:
        sem = r.get("SemanticSim", float('nan'))
        w = r.get("WER", float('nan'))
        bf1 = r.get("BERT-F1", float('nan'))
        try:
            hy = compute_hybrid_score(semantic=sem, wer_v=w, bertf1=bf1, weights=HYBRID_WEIGHTS)
        except Exception:
            hy = float('nan')
        r["Hybrid"] = hy

    # persist CSV
    out_df = pd.DataFrame(rows_out)
    # desired column order
    cols_order = [
        "audio_file", "hypothesis_name",
        "reference_raw", "hypothesis_raw",
        "reference_norm", "hypothesis_norm",
        "BLEU", "WER", "ROUGE-L", "SemanticSim",
        "BERT-P", "BERT-R", "BERT-F1",
        "Hybrid"
    ]
    cols_existing = [c for c in cols_order if c in out_df.columns] + [c for c in out_df.columns if c not in cols_order]
    out_df = out_df[cols_existing]
    out_df.to_csv(output_csv, index=False)
    print(f"Done. Results written to: {output_csv}")

if __name__ == "__main__":
    try:
        process_file(INPUT_CSV, OUTPUT_CSV)
    except Exception as e:
        print("ERROR:", e)
        raise


Use the latest stable models if needed, also the transcripts are in indian dialect and can be multi languages.

Also the preprocessing/normalisation doesn't work in the code

Ask doubts if any, don't assume anything
