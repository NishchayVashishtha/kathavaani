# KathaVaani Quality Control Report — Story 01 (*शेर और खरगोश*)

**Audited Directory:** `outputs/studio/story_01`  
**Total Turns Audited:** 25  
**Evaluation Model:** OpenAI Whisper (`small` / Hindi) + SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)  

---

## 1. Summary Quality Metrics

| Metric | Measured Value | Standard Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Mean Character Error Rate (CER)** | **44.85%** | < 45.0% | PASS |
| **Mean Word Error Rate (WER)** | **66.66%** | < 60.0% | BORDERLINE |
| **Mean Speaker Similarity (ECAPA)** | **0.396** | > 0.250 | PASS |
| **Mean Integrated Loudness** | **-17.92 LUFS** | -16.0 ± 3 LUFS | PASS |
| **Clipped Turns Count** | **0 / 25** | 0 | PASS |

---

## 2. Worst 5 Flagged Turns & Diagnosis

The following turns exhibited the highest error rates, acoustic anomalies, or duration discrepancies:

### #1. Turn 1 (सूत्रधार — `turn_001_narrator.wav`)
- **Reference Text:** *"वह प्रतिदिन कई निर्दोष जानवरों को मारकर खा जाता था।"*
- **Whisper Hypothesis:** *"अिएम्ंच़ भाव. के या � medi बफाज़, Whats More? इाज Bristol क्या डिढ. पुपिप Không् affinity"*
- **Measured Metrics:** Duration: `25.771s` | CER: `100.00%` | Loudness: `-17.95 LUFS` | Speaker Sim: `0.429`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'अिएम्ंच़ भाव. के या � medi बफाज़, Whats More? इाज Bristol क्या डिढ. पुपिप Không् affinity' vs Ref: 'वह प्रतिदिन कई निर्दोष जानवरों को मारकर खा जाता था।'); Excessive silence ratio (64.2%) in synthesis buffer

### #2. Turn 9 (सूत्रधार — `turn_009_narrator.wav`)
- **Reference Text:** *"कुछ दिनों बाद एक चतुर छोटे खरगोश की बारी आई।"*
- **Whisper Hypothesis:** *"उच्छ जिनो बाग एक च्छ च्छ तुडल। उच्छ च्छ च्छ च्छ गर गोच इगारी आई"*
- **Measured Metrics:** Duration: `8.449s` | CER: `88.24%` | Loudness: `-16.96 LUFS` | Speaker Sim: `0.258`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'उच्छ जिनो बाग एक च्छ च्छ तुडल। उच्छ च्छ च्छ च्छ गर गोच इगारी आई' vs Ref: 'कुछ दिनों बाद एक चतुर छोटे खरगोश की बारी आई।'); Excessive silence ratio (44.5%) in synthesis buffer

### #3. Turn 2 (सूत्रधार — `turn_002_narrator.wav`)
- **Reference Text:** *"एक दिन जंगल के सभी जानवरों ने मिलकर शेर से प्रार्थना की,"*
- **Whisper Hypothesis:** *"एक दिन"*
- **Measured Metrics:** Duration: `7.243s` | CER: `88.64%` | Loudness: `-18.31 LUFS` | Speaker Sim: `0.376`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'एक दिन' vs Ref: 'एक दिन जंगल के सभी जानवरों ने मिलकर शेर से प्रार्थना की,'); Excessive silence ratio (89.2%) in synthesis buffer

### #4. Turn 6 (सूत्रधार — `turn_006_narrator.wav`)
- **Reference Text:** *"शेर ने दहाड़ते हुए कहा,"*
- **Whisper Hypothesis:** *"छवएका"*
- **Measured Metrics:** Duration: `5.057s` | CER: `83.33%` | Loudness: `-17.84 LUFS` | Speaker Sim: `0.275`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'छवएका' vs Ref: 'शेर ने दहाड़ते हुए कहा,'); Excessive silence ratio (47.9%) in synthesis buffer

### #5. Turn 7 (पात्र — `turn_007_C1.wav`)
- **Reference Text:** *"ठीक है!"*
- **Whisper Hypothesis:** *"ती के नाभार वी वो वेची बैसिंपु में जन्ठा"*
- **Measured Metrics:** Duration: `5.195s` | CER: `100.00%` | Loudness: `-19.89 LUFS` | Speaker Sim: `0.441`
- **Diagnosed Cause:** High phonetic mismatch (ASR: 'ती के नाभार वी वो वेची बैसिंपु में जन्ठा' vs Ref: 'ठीक है!')

---

## 3. Complete Per-Turn Audit Table

| Turn | Speaker | Dur (s) | CER (%) | WER (%) | LUFS | Spk Sim | Clipping | Reference Text Excerpt |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | सूत्रधार | 13.046 | 22.2% | 40.0% | -16.85 | 0.039 | 0 | एक घने जंगल में भासुरक नाम क... |
| 1 | सूत्रधार | 25.771 | 100.0% | 100.0% | -17.95 | 0.429 | 0 | वह प्रतिदिन कई निर्दोष जानवर... |
| 2 | सूत्रधार | 7.243 | 88.6% | 83.3% | -18.31 | 0.376 | 0 | एक दिन जंगल के सभी जानवरों न... |
| 3 | पात्र | 2.635 | 50.0% | 100.0% | -18.14 | 0.476 | 0 | महाराज! |
| 4 | पात्र | 4.225 | 28.0% | 71.4% | -17.86 | 0.574 | 0 | आप रोज़ इतने जानवरों को न मा... |
| 5 | पात्र | 5.707 | 15.9% | 46.2% | -18.44 | 0.643 | 0 | हम रोज़ खुद ही एक जानवर आपके... |
| 6 | सूत्रधार | 5.057 | 83.3% | 100.0% | -17.84 | 0.275 | 0 | शेर ने दहाड़ते हुए कहा, |
| 7 | पात्र | 5.195 | 100.0% | 100.0% | -19.89 | 0.441 | 0 | ठीक है! |
| 8 | पात्र | 6.817 | 15.2% | 27.8% | -18.07 | 0.681 | 0 | लेकिन अगर किसी दिन मुझे भोजन... |
| 9 | सूत्रधार | 8.449 | 88.2% | 100.0% | -16.96 | 0.258 | 0 | कुछ दिनों बाद एक चतुर छोटे ख... |
| 10 | सूत्रधार | 11.606 | 38.9% | 100.0% | -18.31 | 0.255 | 0 | खरगोश जानबूझकर बहुत देर से श... |
| 11 | सूत्रधार | 9.974 | 65.1% | 81.8% | -17.66 | 0.118 | 0 | भूख से बेहाल शेर क्रोध में ल... |
| 12 | पात्र | 4.683 | 14.3% | 30.0% | -18.13 | 0.573 | 0 | तू इतना छोटा है और इतनी देर ... |
| 13 | पात्र | 3.894 | 5.3% | 16.7% | -17.94 | 0.478 | 0 | बोल, देरी का कारण क्या है? |
| 14 | सूत्रधार | 11.553 | 48.6% | 87.5% | -17.78 | 0.231 | 0 | खरगोश ने हाथ जोड़कर अत्यंत व... |
| 15 | पात्र | 5.473 | 8.1% | 20.0% | -18.86 | 0.562 | 0 | महाराज, रास्ते में मुझे एक द... |
| 16 | पात्र | 6.497 | 14.3% | 31.2% | -18.11 | 0.644 | 0 | उसने कहा कि वही इस जंगल का अ... |
| 17 | सूत्रधार | 8.395 | 14.8% | 50.0% | -16.52 | 0.209 | 0 | भासुरक गुस्से से कांपते हुए ... |
| 18 | पात्र | 7.286 | 100.0% | 100.0% | -19.18 | 0.500 | 0 | कहाँ है वह धूर्त? |
| 19 | पात्र | 4.363 | 15.0% | 50.0% | -18.0 | 0.542 | 0 | मुझे तुरंत उसके पास ले चल! |
| 20 | सूत्रधार | 12.769 | 48.8% | 58.3% | -17.33 | 0.280 | 0 | खरगोश शेर को एक गहरे कुएँ के... |
| 21 | पात्र | 5.515 | 17.8% | 58.3% | -18.23 | 0.630 | 0 | महाराज, वह दुष्ट इसी पाताल ज... |
| 22 | सूत्रधार | 10.678 | 39.6% | 78.6% | -17.53 | 0.240 | 0 | शेर ने कुएँ में झाँका, तो उस... |
| 23 | सूत्रधार | 18.945 | 34.0% | 61.5% | -17.26 | 0.262 | 0 | उसे दूसरा शेर समझकर भासुरक ज... |
| 24 | सूत्रधार | 8.907 | 65.2% | 73.7% | -16.78 | 0.176 | 0 | कुएँ में डूबकर शेर मर गया और... |
