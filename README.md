"""
STT Evaluation Script for Indian Multi-language Transcripts

Models to download (HuggingFace):
1. Semantic Similarity: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
2. BERTScore: microsoft/deberta-v3-base (or use default bert-base-multilingual-cased)

Requirements:
    pip install sacrebleu jiwer rouge-score bert-score sentence-transformers pandas openpyxl tqdm numpy torch

Usage:
    1. Update MODEL_CACHE_DIR to your models location
    2. Set INPUT_CSV path
    3. Run script
"""

import os
import re
import warnings
from typing import List, Tuple
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

# Metrics
import sacrebleu
from jiwer import wer
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "/path/to/your/input.csv"  # UPDATE THIS
OUTPUT_CSV = "/path/to/your/output.csv"  # UPDATE THIS

# Directory where you'll download HuggingFace models
MODEL_CACHE_DIR = "/path/to/your/models"  # UPDATE THIS

# Model names (will load from MODEL_CACHE_DIR or download if not present)
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BERT_SCORE_MODEL = "microsoft/mdeberta-v3-base"  # Multilingual DeBERTa

# Batch sizes
BERT_BATCH_SIZE = 32
SEMANTIC_BATCH_SIZE = 16

# Hybrid score weights
HYBRID_WEIGHTS = {
    "semantic": 0.4,
    "wer": 0.25,
    "bleu": 0.15,
    "rouge": 0.1,
    "bertf1": 0.1
}

# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

# Indian number words (English + transliterated Hindi)
NUMBER_WORDS = {
    # English
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    
    # Indian scales
    "lakh": "100000", "lac": "100000", "lakhs": "100000", "lacs": "100000",
    "crore": "10000000", "crores": "10000000", "cr": "10000000",
    
    # Hindi transliterations (common in Indian English)
    "ek": "1", "do": "2", "teen": "3", "char": "4", "paanch": "5",
    "panch": "5", "chhe": "6", "saat": "7", "aath": "8", "nau": "9", "dus": "10"
}

# Currency normalization
CURRENCY_VARIANTS = {
    # EUR variants
    "eur": "euro", "euro": "euro", "euros": "euro", "€": "euro",
    
    # INR variants
    "inr": "rupee", "rupee": "rupee", "rupees": "rupee", "rs": "rupee",
    "₹": "rupee", "rupaye": "rupee", "rupaiya": "rupee",
    
    # USD variants
    "usd": "dollar", "dollar": "dollar", "dollars": "dollar", "$": "dollar",
    
    # Common in transcripts
    "rupaye": "rupee", "rupaiye": "rupee"
}

# Date/month normalization
MONTHS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "may": "may", "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "oct": "october", "nov": "november", "dec": "december",
    "january": "january", "february": "february", "march": "march",
    "april": "april", "june": "june", "july": "july", "august": "august",
    "september": "september", "october": "october", "november": "november",
    "december": "december"
}

# Common business terms normalization
BUSINESS_TERMS = {
    "rm": "relationship manager",
    "doc": "document", "docs": "documents",
    "fwd": "forward", "bkg": "booking",
    "maturity": "maturity", "tenor": "tenure", "tenure": "tenure"
}


def normalize_currency(text: str) -> str:
    """Normalize currency mentions"""
    text_lower = text.lower()
    for variant, standard in CURRENCY_VARIANTS.items():
        text_lower = re.sub(r'\b' + re.escape(variant) + r'\b', standard, text_lower)
    return text_lower


def normalize_numbers(text: str) -> str:
    """
    Normalize number representations including:
    - Written numbers to digits
    - Indian number scales (lakh, crore)
    - Decimal points
    """
    text = text.lower()
    
    # Handle "1 lakh", "2 crore" patterns
    text = re.sub(r'(\d+)\s*(?:lakhs?|lacs?)', r'\1 lakh', text)
    text = re.sub(r'(\d+)\s*(?:crores?|cr)', r'\1 crore', text)
    
    # Convert "one lakh" to "1 lakh"
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(r'\b' + word + r'\b', digit, text)
    
    # Normalize "95.56", "95 point 56", "95 point five six"
    def replace_point(match):
        return match.group(0).replace(' point ', '.')
    text = re.sub(r'\d+\s+point\s+\d+', replace_point, text)
    
    # Handle sequences like "nine five point six six" -> "95.66"
    # This is complex, so we'll do basic digit sequence handling
    
    # Normalize lakh/crore to actual numbers for better comparison
    text = re.sub(r'(\d+)\s*lakh', lambda m: str(int(m.group(1)) * 100000), text)
    text = re.sub(r'(\d+)\s*crore', lambda m: str(int(m.group(1)) * 10000000), text)
    
    return text


def normalize_dates(text: str) -> str:
    """Normalize date mentions"""
    text_lower = text.lower()
    
    # Normalize months
    for abbr, full in MONTHS.items():
        text_lower = re.sub(r'\b' + abbr + r'\b', full, text_lower)
    
    # Normalize "31st", "29th" to "31", "29"
    text_lower = re.sub(r'(\d+)(?:st|nd|rd|th)\b', r'\1', text_lower)
    
    return text_lower


def normalize_business_terms(text: str) -> str:
    """Normalize business abbreviations and terms"""
    text_lower = text.lower()
    for abbr, full in BUSINESS_TERMS.items():
        text_lower = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text_lower)
    return text_lower


def remove_filler_words(text: str) -> str:
    """Remove common filler words and sounds"""
    fillers = [
        r'\buh+\b', r'\bum+\b', r'\bhmm+\b', r'\bhm+\b',
        r'\bah+\b', r'\boh+\b', r'\byeah+\b', r'\byeah\b',
        r'\bok+\b', r'\bokay+\b', r'\balright+\b'
    ]
    text_lower = text.lower()
    for filler in fillers:
        text_lower = re.sub(filler, '', text_lower)
    return text_lower


def normalize_punctuation(text: str) -> str:
    """Standardize punctuation and spacing"""
    # Remove multiple punctuation
    text = re.sub(r'[.]{2,}', '', text)
    text = re.sub(r'[?!]+', '', text)
    
    # Remove most punctuation but keep decimal points and hyphens in numbers
    text = re.sub(r'(?<!\d)[.,!?;:"\'](?!\d)', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Apply all normalization steps
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Apply normalizations in order
    text = normalize_currency(text)
    text = normalize_numbers(text)
    text = normalize_dates(text)
    text = normalize_business_terms(text)
    text = remove_filler_words(text)
    text = normalize_punctuation(text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score"""
    if not reference or not hypothesis:
        return 0.0
    try:
        return float(sacrebleu.sentence_bleu(
            hypothesis, [reference], 
            smooth_method='exp',
            effective_order=True
        ).score)
    except:
        return 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate"""
    if not reference or not hypothesis:
        return 1.0
    try:
        return float(wer(reference, hypothesis))
    except:
        return 1.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score"""
    if not reference or not hypothesis:
        return 0.0
    try:
        scores = rouge_scorer_obj.score(reference, hypothesis)
        return float(scores['rougeL'].fmeasure)
    except:
        return 0.0


def compute_bertscore_batch(references: List[str], hypotheses: List[str], 
                           model_type: str) -> Tuple[List[float], List[float], List[float]]:
    """Compute BERTScore in batches"""
    if not references or not hypotheses:
        return [], [], []
    
    P_all, R_all, F_all = [], [], []
    
    for i in range(0, len(references), BERT_BATCH_SIZE):
        batch_refs = references[i:i+BERT_BATCH_SIZE]
        batch_hyps = hypotheses[i:i+BERT_BATCH_SIZE]
        
        try:
            P, R, F = bertscore_score(
                batch_hyps, batch_refs,
                model_type=model_type,
                lang="en",
                verbose=False,
                device='cpu'  # Change to 'cuda' if GPU available
            )
            P_all.extend([float(p) for p in P])
            R_all.extend([float(r) for r in R])
            F_all.extend([float(f) for f in F])
        except Exception as e:
            print(f"BERTScore batch failed: {e}")
            P_all.extend([0.0] * len(batch_refs))
            R_all.extend([0.0] * len(batch_refs))
            F_all.extend([0.0] * len(batch_refs))
    
    return P_all, R_all, F_all


def compute_semantic_similarity(model: SentenceTransformer, 
                                reference: str, hypothesis: str) -> float:
    """Compute semantic similarity using sentence embeddings"""
    if not reference or not hypothesis:
        return 0.0
    
    try:
        # Encode both texts
        emb_ref = model.encode(reference, convert_to_tensor=True)
        emb_hyp = model.encode(hypothesis, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.cos_sim(emb_ref, emb_hyp)
        return float(similarity[0][0])
    except Exception as e:
        print(f"Semantic similarity failed: {e}")
        return 0.0


def compute_hybrid_score(metrics: dict) -> float:
    """
    Compute weighted hybrid score
    """
    semantic = metrics.get('SemanticSim', 0.0)
    wer_val = metrics.get('WER', 1.0)
    bleu_val = metrics.get('BLEU', 0.0) / 100.0  # Normalize to 0-1
    rouge_val = metrics.get('ROUGE-L', 0.0)
    bertf1 = metrics.get('BERT-F1', 0.0)
    
    # Handle NaN values
    semantic = 0.0 if math.isnan(semantic) else semantic
    wer_val = 1.0 if math.isnan(wer_val) else wer_val
    bleu_val = 0.0 if math.isnan(bleu_val) else bleu_val
    rouge_val = 0.0 if math.isnan(rouge_val) else rouge_val
    bertf1 = 0.0 if math.isnan(bertf1) else bertf1
    
    # Invert WER (lower is better -> higher score)
    wer_score = max(0.0, 1.0 - wer_val)
    
    hybrid = (
        HYBRID_WEIGHTS['semantic'] * semantic +
        HYBRID_WEIGHTS['wer'] * wer_score +
        HYBRID_WEIGHTS['bleu'] * bleu_val +
        HYBRID_WEIGHTS['rouge'] * rouge_val +
        HYBRID_WEIGHTS['bertf1'] * bertf1
    )
    
    return float(hybrid)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_models():
    """Load required models"""
    print("Loading models...")
    
    # Set cache directory
    os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
    
    # Load semantic similarity model
    print(f"Loading semantic model: {SEMANTIC_MODEL}")
    sem_model = SentenceTransformer(SEMANTIC_MODEL, cache_folder=MODEL_CACHE_DIR)
    
    print("Models loaded successfully!")
    return sem_model


def process_csv(input_path: str, output_path: str):
    """Main processing function"""
    
    # Load data
    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path)
    
    # Identify columns
    columns = df.columns.tolist()
    if len(columns) < 2:
        raise ValueError("CSV must have at least 2 columns (reference + hypothesis)")
    
    reference_col = columns[0]  # First column is ground truth
    hypothesis_cols = columns[1:]  # Rest are hypotheses
    
    print(f"Reference column: {reference_col}")
    print(f"Hypothesis columns: {hypothesis_cols}")
    
    # Load models
    sem_model = load_models()
    
    # Prepare data for batch processing
    results = []
    
    # Collect all pairs for BERTScore batching
    bert_refs = []
    bert_hyps = []
    bert_indices = []
    
    print("\nProcessing transcripts...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        reference_raw = str(row[reference_col]) if pd.notna(row[reference_col]) else ""
        reference_norm = normalize_text(reference_raw)
        
        for hyp_col in hypothesis_cols:
            hypothesis_raw = str(row[hyp_col]) if pd.notna(row[hyp_col]) else ""
            
            if not hypothesis_raw or hypothesis_raw == "nan":
                continue
            
            hypothesis_norm = normalize_text(hypothesis_raw)
            
            # Compute individual metrics
            bleu_score = compute_bleu(reference_norm, hypothesis_norm)
            wer_score = compute_wer(reference_norm, hypothesis_norm)
            rouge_score = compute_rouge_l(reference_norm, hypothesis_norm)
            sem_sim = compute_semantic_similarity(sem_model, reference_norm, hypothesis_norm)
            
            result = {
                'Row': idx + 1,
                'Hypothesis_Name': hyp_col,
                'Reference_Raw': reference_raw,
                'Hypothesis_Raw': hypothesis_raw,
                'Reference_Normalized': reference_norm,
                'Hypothesis_Normalized': hypothesis_norm,
                'BLEU': bleu_score,
                'WER': wer_score,
                'ROUGE-L': rouge_score,
                'SemanticSim': sem_sim,
                'BERT-P': None,
                'BERT-R': None,
                'BERT-F1': None,
                'Hybrid': None
            }
            
            result_idx = len(results)
            results.append(result)
            
            # Queue for BERTScore
            bert_refs.append(reference_norm)
            bert_hyps.append(hypothesis_norm)
            bert_indices.append(result_idx)
    
    # Compute BERTScore in batches
    print("\nComputing BERTScore...")
    if bert_refs:
        P_list, R_list, F_list = compute_bertscore_batch(
            bert_refs, bert_hyps, BERT_SCORE_MODEL
        )
        
        for i, result_idx in enumerate(bert_indices):
            results[result_idx]['BERT-P'] = P_list[i]
            results[result_idx]['BERT-R'] = R_list[i]
            results[result_idx]['BERT-F1'] = F_list[i]
    
    # Compute hybrid scores
    print("Computing hybrid scores...")
    for result in results:
        result['Hybrid'] = compute_hybrid_score(result)
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Processed {len(results)} hypothesis pairs")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for hyp_col in hypothesis_cols:
        hyp_results = output_df[output_df['Hypothesis_Name'] == hyp_col]
        if len(hyp_results) > 0:
            print(f"\n{hyp_col}:")
            print(f"  BLEU:        {hyp_results['BLEU'].mean():.2f}")
            print(f"  WER:         {hyp_results['WER'].mean():.4f}")
            print(f"  ROUGE-L:     {hyp_results['ROUGE-L'].mean():.4f}")
            print(f"  SemanticSim: {hyp_results['SemanticSim'].mean():.4f}")
            print(f"  BERT-F1:     {hyp_results['BERT-F1'].mean():.4f}")
            print(f"  Hybrid:      {hyp_results['Hybrid'].mean():.4f}")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        process_csv(INPUT_CSV, OUTPUT_CSV)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
