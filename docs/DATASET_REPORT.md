# KathaVaani — Dataset & Manifest Report

**Generated On**: 2026-11-19
**Dataset Title**: KathaVaani Hindi Storytelling Speech Corpus
**Total Utterances**: 2066
**Total Audio Duration**: **2.57 hours** (154.5 minutes)
**Mean Utterance Duration**: 4.49 seconds (Min: 2.50s, Max: 15.00s)

---

## 1. Split Distribution (Leakage-Free by Story)

| Split | Number of Stories | Number of Utterances | Total Duration (Hours) | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 18 | 1370 | 1.69 h | 66.3% |
| **Validation** | 3 | 266 | 0.30 h | 12.9% |
| **Test** | 3 | 430 | 0.58 h | 20.8% |

---

## 2. Emotional Category Distribution

| Emotion Label | Utterance Count | Percentage | Primary Hindi Acoustic Cues |
| :--- | :--- | :--- | :--- |
| **NEUTRAL** | 1382 | 66.9% | Intensity modulation, prosodic contour |
| **EXCITED** | 263 | 12.7% | Intensity modulation, prosodic contour |
| **WHISPER** | 210 | 10.2% | Intensity modulation, prosodic contour |
| **ANGRY** | 76 | 3.7% | Intensity modulation, prosodic contour |
| **SURPRISE** | 71 | 3.4% | Intensity modulation, prosodic contour |
| **TENDER** | 20 | 1.0% | Intensity modulation, prosodic contour |
| **FEAR** | 16 | 0.8% | Intensity modulation, prosodic contour |
| **HAPPY** | 13 | 0.6% | Intensity modulation, prosodic contour |
| **SAD** | 9 | 0.4% | Intensity modulation, prosodic contour |
| **TENSE** | 6 | 0.3% | Intensity modulation, prosodic contour |

---

## 4. Data Reconciliation: Raw Sources vs Segmented Speech

- **Raw Ingested Audio**: **6.01 hours** (360.4 minutes) across 26 raw story files.
- **Segmented Voiced Speech**: **2.57 hours** (154.5 minutes) across 2066 clean utterances.
- **Speech Retention Ratio**: **42.9%**

### Why does Segmented Duration differ from Raw Duration?
1. **Voice Activity Filtering (VAD)**: The raw recordings are rich multi-media cartoon narrations and sleep story audio containing opening musical themes, closing credits, long dramatic pauses, and extensive background sound effects (SFX) such as laughing, footsteps, and animal roars. These non-speech sections were eliminated by Silero VAD and Whisper timestamp segmentation.
2. **Strict Utterance Duration Filtering**: Segments shorter than 2.5 seconds (accidental clicks, isolated gasps) and longer than 15.0 seconds (which cause memory overflow or truncation during autoregressive TTS synthesis) were systematically pruned.
3. **Zero-Yield Edge Cases**: `story_020` (29.5m) and `story_021` (9.8m) yielded 0 segmented utterances because their audio tracks had severe music-to-speech ratio or continuous ambient beds that failed Whisper confidence alignment.
4. **Sleep Stories High Yield**: Pure narrative speech stories (`story_014` to `story_019`, `story_022`) achieved 47% to 73% retention due to dense spoken dialogue and minimal sound effects.

### Per-Story Reconciliation Breakdown

| Story ID | Raw Duration | Clean Speech | Retention % | Original Title / Source |
| :--- | :--- | :--- | :--- | :--- |
| `story_000` |  10.1m |   4.2m |  41.7% | Baagh Aaya Bhago! 😨 ｜ Pintu Bhaiya vs  |
| `story_001` |  10.1m |   3.6m |  35.9% | खतरे की वीडियो 😱📱 ｜ Comedy Village Sto |
| `story_002` |  11.6m |   4.1m |  35.5% | Gaon Main Gas Ka Ghotala ｜ गाँव में गै |
| `story_003` |  12.9m |   3.4m |  26.3% | Holi Ka Rang ｜ Pintu Aur Mukhiya Ki Fu |
| `story_004` |  11.3m |   3.9m |  34.4% | बाँके और पिंटू की मिठाई की दुकान  ｜ Co |
| `story_005` |   4.2m |   3.2m |  75.8% | Horror Story 😨 - कर्णेश्वर राक्षस का ख |
| `story_006` |   9.6m |   3.3m |  34.3% | गर्मियों की दुल्हन ｜ Hindi Kahani ｜ Be |
| `story_007` |   5.1m |   2.1m |  41.8% | Horror Story 😨 - घुंघुरू की आवाज ｜ Tru |
| `story_008` |  10.5m |   2.7m |  25.8% | एक चिपकली और पूरे गाँव की जीत ｜ Comedy |
| `story_009` |   4.5m |   3.2m |  72.4% | Horror Story 😨 - भूतिया रेलवे स्टेशन ｜ |
| `story_010` |  11.8m |   2.2m |  18.3% | गाँव में लस्सी कांड 😳｜  Funny Village  |
| `story_011` |   3.6m |   2.6m |  72.6% | Horror Story 😨 - भूतिया लड़का ｜ Story  |
| `story_012` |  11.9m |   4.7m |  39.2% | Jab Kachra Bana Khazana ｜ New Hindi Ka |
| `story_013` |  11.0m |   2.0m |  18.3% | गांव का चलता - फिरता बाज़ार ｜ New Hind |
| `story_014` |  26.5m |  12.6m |  47.4% | Khiladiyon Ke Kisse - Sleep Stories ｜  |
| `story_015` |  26.2m |  13.9m |  53.2% | Mahatma Gandhi ki Shiksha - Hindi me K |
| `story_016` |  26.2m |  19.0m |  72.6% | Nykaa Ki Nayikaa - Sone Ke Liye Kahani |
| `story_017` |  20.0m |  11.0m |  55.1% | Ola ki Kahani -  Sone Ke Liye Kahani ｜ |
| `story_018` |  10.6m |   3.6m |  34.3% | चाय की टपरी पर जंग ｜ Hindi Cartoon Sto |
| `story_019` |  26.6m |  19.0m |  71.3% | The Miracle Morning- How to sleep fast |
| `story_020` |  29.5m |   0.0m |   0.0% | Vivekanand ji ke Updesh ｜ Negative tho |
| `story_021` |   9.8m |   0.0m |   0.0% | दिलदारियापुर मे  डर का साया ｜ Hindi Ka |
| `story_022` |  30.0m |  20.5m |  68.4% | फ्लाइंग सिख - मिल्खा सिंह ｜ हिंदी कहान |
| `story_023` |  12.0m |   3.2m |  26.6% | बुद्धू राम की ट्रेन यात्रा ｜ Hindi Kah |
| `story_024` |   4.7m |   3.1m |  66.9% | भूतिया लाइब्रेरी - True Horror Stories |
| `story_025` |  10.3m |   3.3m |  32.0% | हाईजैक बस  😱 ｜ Hijack Bus Story ｜ Pint |