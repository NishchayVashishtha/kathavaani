"""
Evaluation Metrics for TTS Bake-Off & Quality Audits
Calculates CER, WER, ECAPA Speaker Similarity, and Real-Time Factor (RTF).
"""

import re
import numpy as np
from typing import Tuple


def normalize_devanagari(text: str) -> str:
    """Normalizes Hindi Devanagari text for fair ASR metric comparison."""
    text = text.strip().lower()
    # Remove standard punctuation and dandas
    text = re.sub(r"[।॥,?!:;\"\'\-—–()\[\]]", "", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(s1: list, s2: list) -> int:
    """Computes Levenshtein edit distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) on Devanagari text."""
    ref_norm = list(normalize_devanagari(reference).replace(" ", ""))
    hyp_norm = list(normalize_devanagari(hypothesis).replace(" ", ""))

    if len(ref_norm) == 0:
        return 0.0 if len(hyp_norm) == 0 else 1.0

    dist = levenshtein_distance(ref_norm, hyp_norm)
    return min(1.0, dist / len(ref_norm))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) on Devanagari text."""
    ref_words = normalize_devanagari(reference).split()
    hyp_words = normalize_devanagari(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return min(1.0, dist / len(ref_words))


def compute_snr(audio_path: str) -> float:
    """Estimates Signal-to-Noise Ratio (dB) of synthesized audio."""
    try:
        import soundfile as sf
        data, _ = sf.read(audio_path)
        if len(data) == 0:
            return 0.0
        # Energy of signal vs low energy floor
        signal_power = np.mean(data ** 2)
        noise_floor = np.percentile(data ** 2, 5) + 1e-10
        snr = 10 * np.log10(signal_power / noise_floor)
        return float(round(max(0.0, snr), 2))
    except Exception:
        return 0.0
