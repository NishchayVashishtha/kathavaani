"""
Standardized 30-Sentence Hindi Storytelling Benchmark Suite for TTS Bake-Off.
Spans narration, dialogue, honorifics, and varied emotional delivery.
"""

BENCHMARK_SENTENCES = [
    # 1-5: Neutral Narration
    {
        "id": "SENT_01",
        "category": "narration",
        "emotion": "neutral",
        "text": "एक पुराने गांव के किनारे एक घना और शांत बरगद का पेड़ था।",
    },
    {
        "id": "SENT_02",
        "category": "narration",
        "emotion": "neutral",
        "text": "राजा विक्रम सिंह रोज़ सुबह अपनी प्रजा का हालचाल जानने महल से बाहर निकलते थे।",
    },
    {
        "id": "SENT_03",
        "category": "narration",
        "emotion": "neutral",
        "text": "सूरज ढलने के बाद नदी का पानी बिल्कुल शांत और चमकीला दिखाई देता था।",
    },
    {
        "id": "SENT_04",
        "category": "narration",
        "emotion": "neutral",
        "text": "दादी माँ ने अपनी संदूक से एक पुरानी पीली किताब निकाली और मेज़ पर रख दी।",
    },
    {
        "id": "SENT_05",
        "category": "narration",
        "emotion": "neutral",
        "text": "बाज़ार में सभी व्यापारी अपनी दुकानों को बंद करने की तैयारी कर रहे थे।",
    },

    # 6-10: Happy / Joy / Excited Dialogue
    {
        "id": "SENT_06",
        "category": "dialogue",
        "emotion": "happy",
        "text": "देखो भैया! आज हमारे खेत में सोने जैसी सुनहरी फसल लहलहा रही है!",
    },
    {
        "id": "SENT_07",
        "category": "dialogue",
        "emotion": "excited",
        "text": "अरे वाह! हम सब मिलकर इस बार मेले में सबसे बड़ा झूला झूलेंगे!",
    },
    {
        "id": "SENT_08",
        "category": "dialogue",
        "emotion": "happy",
        "text": "माँ, मेरी परीक्षा का परिणाम आ गया और मैंने पूरे विद्यालय में पहला स्थान पाया है!",
    },
    {
        "id": "SENT_09",
        "category": "dialogue",
        "emotion": "excited",
        "text": "जल्दी करो, राजा साहब ने पूरे गांव को शाही दावत के लिए आमंत्रित किया है!",
    },
    {
        "id": "SENT_10",
        "category": "dialogue",
        "emotion": "happy",
        "text": "यह मिठाई सचमुच बहुत स्वादिष्ट है, ऐसा स्वाद मैंने पहले कभी नहीं चखा।",
    },

    # 11-15: Sad / Tender / Grief
    {
        "id": "SENT_11",
        "category": "dialogue",
        "emotion": "sad",
        "text": "कास मैं उस वक़्त वहाँ होता, तो शायद आज यह भयानक अनर्थ न हुआ होता।",
    },
    {
        "id": "SENT_12",
        "category": "dialogue",
        "emotion": "tender",
        "text": "चिंता मत करो बेटा, जब तक मैं तुम्हारे साथ हूँ, तुम्हें कोई दुःख नहीं होगा।",
    },
    {
        "id": "SENT_13",
        "category": "dialogue",
        "emotion": "sad",
        "text": "हमारा सब कुछ उस बाढ़ में बह गया, अब हमारे सिर पर छत भी नहीं बची।",
    },
    {
        "id": "SENT_14",
        "category": "dialogue",
        "emotion": "tender",
        "text": "दादी की आँखें भर आईं जब उन्होंने अपने पोते को इतने सालों बाद गले से लगाया।",
    },
    {
        "id": "SENT_15",
        "category": "dialogue",
        "emotion": "sad",
        "text": "उसने धीमे स्वर में कहा, 'शायद हमारी तक़दीर में बिछड़ना ही लिखा था।'",
    },

    # 16-20: Angry / Outraged / Harsh
    {
        "id": "SENT_16",
        "category": "dialogue",
        "emotion": "angry",
        "text": "तुम्हारी इतनी हिम्मत कैसे हुई कि तुम मेरी अनुमति के बिना इस कक्ष में कदम रखो!",
    },
    {
        "id": "SENT_17",
        "category": "dialogue",
        "emotion": "angry",
        "text": "झूठ बोलना बंद करो! मैं तुम्हारी चालबाज़ी को बहुत अच्छी तरह समझ चुका हूँ!",
    },
    {
        "id": "SENT_18",
        "category": "dialogue",
        "emotion": "angry",
        "text": "सेनापति! इस गद्दार को तुरंत बंदी बना लो और कालकोठरी में डाल दो!",
    },
    {
        "id": "SENT_19",
        "category": "dialogue",
        "emotion": "angry",
        "text": "बस बहुत हो चुका! अब तुम्हारा एक भी बहाना नहीं सुना जाएगा!",
    },
    {
        "id": "SENT_20",
        "category": "dialogue",
        "emotion": "disgust",
        "text": "छिः! अपने ही भाई के साथ ऐसा विश्वासघात करते हुए तुम्हें ज़रा भी लज्जा नहीं आई?",
    },

    # 21-25: Fear / Panic / Whispering
    {
        "id": "SENT_21",
        "category": "dialogue",
        "emotion": "fear",
        "text": "चुप रहो! दरवाज़े के बाहर कोई भारी कदमों से इधर ही बढ़ रहा है!",
    },
    {
        "id": "SENT_22",
        "category": "dialogue",
        "emotion": "whisper",
        "text": "धीरे बोलो, अगर पहरेदारों ने हमारी आवाज़ सुन ली तो हम ज़िंदा नहीं बचेंगे।",
    },
    {
        "id": "SENT_23",
        "category": "dialogue",
        "emotion": "fear",
        "text": "हे भगवान! वह परछाईं दीवार पर कैसे हिल रही है? वहाँ तो कोई भी नहीं है!",
    },
    {
        "id": "SENT_24",
        "category": "dialogue",
        "emotion": "whisper",
        "text": "मैंने खिड़की के पीछे किसी की लाल आँखें चमकती हुई देखी हैं...",
    },
    {
        "id": "SENT_25",
        "category": "dialogue",
        "emotion": "fear",
        "text": "भागो! भागो यहाँ से! जंगल में आदमखोर बाघ आ गया है!",
    },

    # 26-30: Suspense / Mystery / Surprise
    {
        "id": "SENT_26",
        "category": "narration",
        "emotion": "tense",
        "text": "हवेली के तहखाने का दरवाज़ा अचानक खुद-ब-खुद चरमराते हुए खुल गया।",
    },
    {
        "id": "SENT_27",
        "category": "dialogue",
        "emotion": "surprise",
        "text": "क्या? क्या यह वही जादुई अंगूठी है जो सौ साल पहले खो गई थी?",
    },
    {
        "id": "SENT_28",
        "category": "narration",
        "emotion": "tense",
        "text": "रात के सन्नाटे में दूर पहाड़ी पर बजने वाले घुंघरूओं की आवाज़ साफ़ सुनाई दे रही थी।",
    },
    {
        "id": "SENT_29",
        "category": "dialogue",
        "emotion": "surprise",
        "text": "असंभव! जिसे हम सब मरा हुआ समझ रहे थे, वह हमारे सामने ज़िंदा खड़ा था!",
    },
    {
        "id": "SENT_30",
        "category": "narration",
        "emotion": "tense",
        "text": "सड़क के मोड़ पर एक अजनबी काला लबादा ओढ़े चुपचाप पहरा दे रहा था।",
    },
]
