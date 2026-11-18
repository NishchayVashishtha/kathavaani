# KathaVaani — Dataset & Manifest Report

**Generated On**: 2026-11-19
**Dataset Title**: KathaVaani Hindi Storytelling Speech Corpus
**Total Utterances**: 236
**Total Audio Duration**: **0.27 hours** (16.4 minutes)
**Mean Utterance Duration**: 4.16 seconds (Min: 2.52s, Max: 14.00s)

---

## 1. Split Distribution (Leakage-Free by Story)

| Split | Number of Stories | Number of Utterances | Total Duration (Hours) | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 4 | 181 | 0.22 h | 76.7% |
| **Validation** | 1 | 55 | 0.05 h | 23.3% |
| **Test** | 1 | 55 | 0.05 h | 23.3% |

---

## 2. Emotional Category Distribution

| Emotion Label | Utterance Count | Percentage | Primary Hindi Acoustic Cues |
| :--- | :--- | :--- | :--- |
| **EXCITED** | 97 | 41.1% | Intensity modulation, prosodic contour |
| **NEUTRAL** | 92 | 39.0% | Intensity modulation, prosodic contour |
| **SURPRISE** | 19 | 8.1% | Intensity modulation, prosodic contour |
| **ANGRY** | 17 | 7.2% | Intensity modulation, prosodic contour |
| **TENSE** | 4 | 1.7% | Intensity modulation, prosodic contour |
| **TENDER** | 3 | 1.3% | Intensity modulation, prosodic contour |
| **FEAR** | 2 | 0.8% | Intensity modulation, prosodic contour |
| **HAPPY** | 1 | 0.4% | Intensity modulation, prosodic contour |
| **SAD** | 1 | 0.4% | Intensity modulation, prosodic contour |

---

## 3. Data Processing Verification
- **Audio Format**: 16 kHz Mono PCM WAV, standardized amplitude.
- **Length Bounds**: 2.5s to 15.0s per utterance, avoiding truncation.
- **Devanagari Normalisation**: Nukta standardization, whitespace normalization, Latin artifact removal.
- **Zero Leakage**: Train, Validation, and Test partitions strictly segregated by story identity.