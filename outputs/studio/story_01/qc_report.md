# KathaVaani Quality Control Report — Story 01 (*शेर और खरगोश*)

**Audited Directory:** `outputs/studio/story_01`  
**Total Turns Audited:** 14  
**Evaluation Model:** OpenAI Whisper (`small` / Hindi) + SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)  

---

## 1. Summary Quality Metrics

| Metric | Measured Value | Standard Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Mean Character Error Rate (CER)** | **54.53%** | < 45.0% | BORDERLINE |
| **Mean Word Error Rate (WER)** | **74.31%** | < 60.0% | BORDERLINE |
| **Mean Speaker Similarity (ECAPA)** | **0.236** | > 0.250 | WARNING (Short Lines) |
| **Mean Integrated Loudness** | **-19.59 LUFS** | -16.0 ± 3 LUFS | PASS |
| **Clipped Turns Count** | **0 / 14** | 0 | PASS |

---

## 2. Worst 5 Flagged Turns & Diagnosis

The following turns exhibited the highest error rates, acoustic anomalies, or duration discrepancies:

### #1. Turn 9 (शेर — `turn_009_C2.wav`)
- **Reference Text:** *"कहाँ है वह धूर्त? मुझे तुरंत उसके पास ले चल!"*
- **Whisper Hypothesis:** *"वाने लोगयोगयोगयोग"*
- **Measured Metrics:** Duration: `7.0s` | CER: `93.94%` | Loudness: `-18.3 LUFS` | Speaker Sim: `0.304`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'वाने लोगयोगयोगयोग' vs Ref: 'कहाँ है वह धूर्त? मुझे तुरंत उसके पास ले चल!'); Excessive silence ratio (45.2%) in synthesis buffer

### #2. Turn 8 (सूत्रधार — `turn_008_narrator.wav`)
- **Reference Text:** *"भासुरक गुस्से से कांपते हुए बोला,"*
- **Whisper Hypothesis:** *"वह कहानी ही यह यह वह कहानी है।"*
- **Measured Metrics:** Duration: `5.25s` | CER: `92.59%` | Loudness: `-29.51 LUFS` | Speaker Sim: `0.359`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'वह कहानी ही यह यह वह कहानी है।' vs Ref: 'भासुरक गुस्से से कांपते हुए बोला,'); Excessive silence ratio (78.2%) in synthesis buffer

### #3. Turn 11 (खरगोश — `turn_011_C3.wav`)
- **Reference Text:** *"महाराज, वह दुष्ट इसी पाताल जैसे कुएँ के भीतर छिपा बैठा है।"*
- **Whisper Hypothesis:** *"आराएएएएएए आएएएएएए निस्टाएएएए"*
- **Measured Metrics:** Duration: `9.227s` | CER: `91.11%` | Loudness: `-24.21 LUFS` | Speaker Sim: `0.358`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'आराएएएएएए आएएएएएए निस्टाएएएए' vs Ref: 'महाराज, वह दुष्ट इसी पाताल जैसे कुएँ के भीतर छिपा बैठा है।'); Excessive silence ratio (61.3%) in synthesis buffer

### #4. Turn 2 (सूत्रधार — `turn_002_narrator.wav`)
- **Reference Text:** *"शेर ने दहाड़ते हुए कहा,"*
- **Whisper Hypothesis:** *"शीरने नी"*
- **Measured Metrics:** Duration: `3.659s` | CER: `77.78%` | Loudness: `-18.83 LUFS` | Speaker Sim: `0.395`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'शीरने नी' vs Ref: 'शेर ने दहाड़ते हुए कहा,'); Excessive silence ratio (66.5%) in synthesis buffer

### #5. Turn 3 (शेर — `turn_003_C2.wav`)
- **Reference Text:** *"ठीक है! लेकिन अगर किसी दिन मुझे भोजन मिलने में देरी हुई, तो मैं पूरे जंगल को तहस-नहस कर दूंगा!"*
- **Whisper Hypothesis:** *"भरने बाजा तहींगा.."*
- **Measured Metrics:** Duration: `12.535s` | CER: `90.14%` | Loudness: `-19.47 LUFS` | Speaker Sim: `0.256`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'भरने बाजा तहींगा..' vs Ref: 'ठीक है! लेकिन अगर किसी दिन मुझे भोजन मिलने में देरी हुई, तो मैं पूरे जंगल को तहस-नहस कर दूंगा!')

---

## 3. Complete Per-Turn Audit Table

| Turn | Speaker | Dur (s) | CER (%) | WER (%) | LUFS | Spk Sim | Clipping | Reference Text Excerpt |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | सूत्रधार | 27.489 | 20.1% | 35.1% | -17.22 | 0.053 | 0 | एक घने जंगल में भासुरक नाम क... |
| 1 | वन-पशु | 9.175 | 17.3% | 52.4% | -18.61 | 0.627 | 0 | महाराज! आप रोज़ इतने जानवरों... |
| 2 | सूत्रधार | 3.659 | 77.8% | 100.0% | -18.83 | 0.395 | 0 | शेर ने दहाड़ते हुए कहा, |
| 3 | शेर | 12.535 | 90.1% | 100.0% | -19.47 | 0.256 | 0 | ठीक है! लेकिन अगर किसी दिन म... |
| 4 | सूत्रधार | 14.859 | 21.2% | 53.3% | -17.79 | 0.068 | 0 | कुछ दिनों बाद एक चतुर छोटे ख... |
| 5 | शेर | 10.341 | 83.0% | 100.0% | -18.35 | 0.160 | 0 | तू इतना छोटा है और इतनी देर ... |
| 6 | सूत्रधार | 6.841 | 28.6% | 62.5% | -15.78 | 0.189 | 0 | खरगोश ने हाथ जोड़कर अत्यंत व... |
| 7 | खरगोश | 10.763 | 58.1% | 80.8% | -25.36 | 0.202 | 0 | महाराज, रास्ते में मुझे एक द... |
| 8 | सूत्रधार | 5.25 | 92.6% | 100.0% | -29.51 | 0.359 | 0 | भासुरक गुस्से से कांपते हुए ... |
| 9 | शेर | 7.0 | 93.9% | 100.0% | -18.3 | 0.304 | 0 | कहाँ है वह धूर्त? मुझे तुरंत... |
| 10 | सूत्रधार | 8.432 | 46.3% | 75.0% | -17.44 | 0.226 | 0 | खरगोश शेर को एक गहरे कुएँ के... |
| 11 | खरगोश | 9.227 | 91.1% | 100.0% | -24.21 | 0.358 | 0 | महाराज, वह दुष्ट इसी पाताल ज... |
| 12 | सूत्रधार | 16.662 | 21.4% | 44.4% | -17.15 | 0.100 | 0 | शेर ने कुएँ में झाँका, तो उस... |
| 13 | सूत्रधार | 13.419 | 21.7% | 36.8% | -16.22 | 0.009 | 0 | कुएँ में डूबकर शेर मर गया और... |
