# KathaVaani Quality Control Report — Story 01 (*शेर और खरगोश*)

**Audited Directory:** `outputs/studio/story_02`  
**Total Turns Audited:** 12  
**Evaluation Model:** OpenAI Whisper (`small` / Hindi) + SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)  

---

## 1. Summary Quality Metrics

| Metric | Measured Value | Standard Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Mean Character Error Rate (CER)** | **55.91%** | < 45.0% | BORDERLINE |
| **Mean Word Error Rate (WER)** | **78.35%** | < 60.0% | BORDERLINE |
| **Mean Speaker Similarity (ECAPA)** | **0.264** | > 0.250 | PASS |
| **Mean Integrated Loudness** | **-18.77 LUFS** | -16.0 ± 3 LUFS | PASS |
| **Clipped Turns Count** | **0 / 12** | 0 | PASS |

---

## 2. Worst 5 Flagged Turns & Diagnosis

The following turns exhibited the highest error rates, acoustic anomalies, or duration discrepancies:

### #1. Turn 1 (सूत्रधार — `turn_001_narrator.wav`)
- **Reference Text:** *"दादी माँ ने कांपते हुए हाथ जोड़कर कहा,"*
- **Whisper Hypothesis:** *"शीक तुम्यें शानवाफुर्ट � curved ीटे 바로 koy champagne? और ज़ों बो tendencies and ऑादी generating भीच मुआ लिच ज़ों शố<|hi|> nove खेड मैं आपकांз सोचमसे नहीं। Brand하세요 jana जुय्ं मॉसी GM A"*
- **Measured Metrics:** Duration: `6.045s` | CER: `100.00%` | Loudness: `-22.97 LUFS` | Speaker Sim: `0.321`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'शीक तुम्यें शानवाफुर्ट � curved ीटे 바로 koy champagne? और ज़ों बो tendencies and ऑादी generating भीच मुआ लिच ज़ों शố<|hi|> nove खेड मैं आपकांз सोचमसे नहीं। Brand하세요 jana जुय्ं मॉसी GM A' vs Ref: 'दादी माँ ने कांपते हुए हाथ जोड़कर कहा,'); Excessive silence ratio (72.6%) in synthesis buffer

### #2. Turn 8 (पिंकी — `turn_008_C3.wav`)
- **Reference Text:** *"हाँ मेरी बच्ची, तुम्हें तुम्हारा घर भी मिलेगा और तुम्हारी गुड़िया भी सुरक्षित रहेगी।"*
- **Whisper Hypothesis:** *"वह बाजी तुमें तुमागर।"*
- **Measured Metrics:** Duration: `5.43s` | CER: `81.16%` | Loudness: `-21.36 LUFS` | Speaker Sim: `0.318`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'वह बाजी तुमें तुमागर।' vs Ref: 'हाँ मेरी बच्ची, तुम्हें तुम्हारा घर भी मिलेगा और तुम्हारी गुड़िया भी सुरक्षित रहेगी।'); Excessive silence ratio (67.7%) in synthesis buffer

### #3. Turn 6 (पिंकी — `turn_006_C3.wav`)
- **Reference Text:** *"राजा जी! क्या हमें हमारा घर वापस मिल जाएगा? मुझे अपनी गुड़िया बहुत याद आ रही है।"*
- **Whisper Hypothesis:** *"हे बदब ञ़ई कर थसो बछी न है championship कर बतर बाचहक कर ठीMEXT? नहीं Still on Instagram था ड clearly"*
- **Measured Metrics:** Duration: `10.754s` | CER: `100.00%` | Loudness: `-20.43 LUFS` | Speaker Sim: `0.290`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'हे बदब ञ़ई कर थसो बछी न है championship कर बतर बाचहक कर ठीMEXT? नहीं Still on Instagram था ड clearly' vs Ref: 'राजा जी! क्या हमें हमारा घर वापस मिल जाएगा? मुझे अपनी गुड़िया बहुत याद आ रही है।')

### #4. Turn 4 (राजा विक्रमादित्य — `turn_004_C2.wav`)
- **Reference Text:** *"किसकी इतनी मजाल जो हमारे राज्य में एक लाचार वृद्धा और अबोध बच्ची पर जुल्म करे? सिपाहियों, उस सेठ को तुरंत बेड़ियों में बांधकर हाजिर करो!"*
- **Whisper Hypothesis:** *"पि उछ तानन Filipitos लिपि में लिएा ज़ि cade merry... ढदा अचा न yarns faq mishe statement आचा डराइश .... ... Ti renew"*
- **Measured Metrics:** Duration: `10.999s` | CER: `92.59%` | Loudness: `-18.6 LUFS` | Speaker Sim: `0.218`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'पि उछ तानन Filipitos लिपि में लिएा ज़ि cade merry... ढदा अचा न yarns faq mishe statement आचा डराइश .... ... Ti renew' vs Ref: 'किसकी इतनी मजाल जो हमारे राज्य में एक लाचार वृद्धा और अबोध बच्ची पर जुल्म करे? सिपाहियों, उस सेठ को तुरंत बेड़ियों में बांधकर हाजिर करो!'); Low acoustic timbre similarity (0.218) against reference voice

### #5. Turn 9 (सूत्रधार — `turn_009_narrator.wav`)
- **Reference Text:** *"दादी माँ की आँखों से कृतज्ञता के आँसू बह निकले और वे भावुक होकर बोलीं,"*
- **Whisper Hypothesis:** *"तादी मागर के आखो से क्रितगता के आसुद बहनिकले और वे भावो खोखर पुजि"*
- **Measured Metrics:** Duration: `10.763s` | CER: `41.82%` | Loudness: `-18.88 LUFS` | Speaker Sim: `0.224`
- **Diagnosed Cause:** Excessive silence ratio (49.7%) in synthesis buffer; Low acoustic timbre similarity (0.224) against reference voice

---

## 3. Complete Per-Turn Audit Table

| Turn | Speaker | Dur (s) | CER (%) | WER (%) | LUFS | Spk Sim | Clipping | Reference Text Excerpt |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | सूत्रधार | 20.939 | 18.6% | 60.0% | -17.42 | 0.157 | 0 | उज्जैन के राजदरबार में राजा ... |
| 1 | सूत्रधार | 6.045 | 100.0% | 100.0% | -22.97 | 0.321 | 0 | दादी माँ ने कांपते हुए हाथ ज... |
| 2 | दादी माँ | 16.183 | 41.4% | 72.0% | -17.66 | 0.504 | 0 | महाराज! रक्षा कीजिए! पड़ोस क... |
| 3 | सूत्रधार | 8.769 | 15.9% | 60.0% | -17.66 | 0.173 | 0 | यह सुनकर राजा विक्रमादित्य क... |
| 4 | राजा विक्रमादित्य | 10.999 | 92.6% | 96.2% | -18.6 | 0.218 | 0 | किसकी इतनी मजाल जो हमारे राज... |
| 5 | सूत्रधार | 10.818 | 56.6% | 78.6% | -19.12 | 0.227 | 0 | तभी नन्ही पिंकी ने डरते-डरते... |
| 6 | पिंकी | 10.754 | 100.0% | 100.0% | -20.43 | 0.290 | 0 | राजा जी! क्या हमें हमारा घर ... |
| 7 | सूत्रधार | 10.721 | 30.0% | 66.7% | -17.21 | 0.176 | 0 | राजा का क्रोध पल भर में पिघल... |
| 8 | पिंकी | 5.43 | 81.2% | 100.0% | -21.36 | 0.318 | 0 | हाँ मेरी बच्ची, तुम्हें तुम्... |
| 9 | सूत्रधार | 10.763 | 41.8% | 73.3% | -18.88 | 0.224 | 0 | दादी माँ की आँखों से कृतज्ञत... |
| 10 | दादी माँ | 8.897 | 41.4% | 64.3% | -16.95 | 0.403 | 0 | महाराज, भगवान आपकी उम्र लंबी... |
| 11 | सूत्रधार | 9.142 | 51.6% | 69.2% | -17.01 | 0.153 | 0 | पूरा राजदरबार राजा विक्रमादि... |
