import React, { useState } from 'react';
import { BookOpen, Sparkles, Wand2, PlayCircle, Loader2 } from 'lucide-react';

interface StoryInputProps {
  onAnalyze: (text: string) => Promise<void>;
  onSynthesize: () => Promise<void>;
  isAnalyzing: boolean;
  isSynthesizing: boolean;
  hasAnalyzed: boolean;
}

const PRESET_STORIES = [
  {
    id: '01',
    title: 'पंचतंत्र: शेर और चतुर खरगोश (क्रूरता, चालाकी, कुआँ)',
    text: `एक घने जंगल में भासुरक नाम का एक बहुत क्रूर और शक्तिशाली शेर रहता था। वह प्रतिदिन कई निर्दोष जानवरों को मारकर खा जाता था।
एक दिन जंगल के सभी जानवरों ने मिलकर शेर से प्रार्थना की, "महाराज! आप रोज़ इतने जानवरों को न मारें। हम रोज़ खुद ही एक जानवर आपके भोजन के लिए भेज दिया करेंगे।"
शेर ने दहाड़ते हुए कहा, "ठीक है! लेकिन अगर किसी दिन मुझे भोजन मिलने में देरी हुई, तो मैं पूरे जंगल को तहस-नहस कर दूंगा!"
कुछ दिनों बाद एक चतुर छोटे खरगोश की बारी आई। खरगोश जानबूझकर बहुत देर से शेर के पास पहुँचा।
भूख से बेहाल शेर क्रोध में लाल-पीला हो गया और चिल्लाया, "तू इतना छोटा है और इतनी देर से आया है? बोल, देरी का कारण क्या है?"
खरगोश ने हाथ जोड़कर अत्यंत विनम्रता से कहा, "महाराज, रास्ते में मुझे एक दूसरा शेर मिल गया था। उसने कहा कि वही इस जंगल का असली राजा है और उसने मुझे रोक लिया था।"
भासुरक गुस्से से कांपते हुए बोला, "कहाँ है वह धूर्त? मुझे तुरंत उसके पास ले चल!"
खरगोश शेर को एक गहरे कुएँ के पास ले गया और फुसफुसाया, "महाराज, वह दुष्ट इसी पाताल जैसे कुएँ के भीतर छिपा बैठा है।"
शेर ने कुएँ में झाँका, तो उसे पानी में अपनी ही परछाईं दिखाई दी। उसे दूसरा शेर समझकर भासुरक ज़ोर से दहाड़ा और कुएँ में कूद पड़ा।
कुएँ में डूबकर शेर मर गया और नन्हे खरगोश ने अपनी चतुराई से पूरे जंगल को भयमुक्त कर दिया।`,
  },
  {
    id: '02',
    title: 'अकबर-बीरबल: बीरबल की खिचड़ी (न्याय, चतुराई)',
    text: `सर्दियों की एक ठंडी रात में शहंशाह अकबर और बीरबल यमुना नदी के किनारे टहल रहे थे। पानी बर्फ की तरह ठंडा था।
अकबर ने कहा, "बीरबल! क्या कोई इंसान दौलत के लिए पूरी रात इस बर्फीले पानी में खड़ा रह सकता है?"
बीरबल ने मुस्कुरा कर कहा, "जहाँपनाह, इंसान भूख और ज़रूरत के लिए कुछ भी कर सकता है।"
अगले दिन एक निर्धन ब्राह्मण आया जिसने सौ सोने के सिक्कों के लिए पूरी रात यमुना के ठंडे पानी में बिताई।
अकबर ने ब्राह्मण से पूछा, "तुम इतनी ठंड में कैसे जीवित रहे?"
ब्राह्मण ने झुककर कहा, "हुज़ूर, दूर महल की छत पर एक दीपक जल रहा था। मैं उसी दीपक को देखकर ठंड भूल गया।"
अकबर ने कठोरता से कहा, "इसका मतलब तुमने महल के दीपक की गरमी से रात काटी! तुम्हें कोई इनाम नहीं मिलेगा।"
ब्राह्मण रोता हुआ बीरबल के पास गया। अगले दिन बीरबल दरबार नहीं आए।
अकबर खुद बीरबल के घर पहुंचे। उन्होंने देखा कि ज़मीन पर आग जल रही है और उससे पाँच हाथ ऊपर हाँडी टंगी है।
अकबर ने हँसते हुए पूछा, "बीरबल, क्या यह खिचड़ी कभी पकेगी?"
बीरबल ने नम्रता से कहा, "जहाँपनाह, अगर महल के एक छोटे से दीपक की गरमी से यमुना में खड़ा ब्राह्मण बच सकता है, तो इस आग से मेरी खिचड़ी भी ज़रूर पकेगी।"
अकबर को अपनी गलती समझ आ गई और उन्होंने ब्राह्मण को बुलाकर पूरा इनाम दिया।`,
  },
  {
    id: '05',
    title: 'भूतिया हवेली का रहस्य (रहस्य, सस्पेंस, डर)',
    text: `काली अमावस्या की रात थी। गाँव के अंतिम छोर पर स्थित पुरानी हवेली में सन्नाटा पसरा हुआ था।
अमित ने कांपते हाथों से टॉर्च जलाई और फुसफुसाया, "रोहन, मुझे लगता है हमें वापस गाँव लौट जाना चाहिए। यहाँ कुछ अजीब है।"
रोहन ने आगे बढ़ते हुए धीमी आवाज़ में कहा, "डरपोक मत बनो! हवेली के उस कमरे में खज़ाने का नक्शा छिपा है।"
तभी हवेली के ऊपर वाले कमरे से किसी के चलने की भारी आहट सुनाई दी—छम... छम... छम...
अमित की सांसें अटक गईं। उसने चिल्लाकर कहा, "भागो रोहन! कोई हमारे पीछे खड़ा है!"
अचानक टॉर्च की रोशनी बुझ गई और सामने एक परछाईं आकर रुक गई।
उस परछाईं ने धीमी, भारी आवाज़ में कहा, "यहाँ से कोई ज़िंदा वापस नहीं जाता..."`,
  },
];

export const StoryInput: React.FC<StoryInputProps> = ({
  onAnalyze,
  onSynthesize,
  isAnalyzing,
  isSynthesizing,
  hasAnalyzed,
}) => {
  const [selectedPreset, setSelectedPreset] = useState('01');
  const [storyText, setStoryText] = useState(PRESET_STORIES[0].text);

  const handlePresetChange = (presetId: string) => {
    setSelectedPreset(presetId);
    const story = PRESET_STORIES.find((s) => s.id === presetId);
    if (story) {
      setStoryText(story.text);
    }
  };

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-amber-400" />
          <h2 className="text-base font-bold text-slate-100 font-devanagari">
            कहानी चयन एवं पटकथा (Story Input & Script Analysis)
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Preset:</span>
          <select
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="text-xs bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-amber-500 font-devanagari"
          >
            {PRESET_STORIES.map((p) => (
              <option key={p.id} value={p.id}>
                {p.title}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="relative">
        <textarea
          rows={7}
          value={storyText}
          onChange={(e) => setStoryText(e.target.value)}
          placeholder="यहाँ अपनी हिन्दी कहानी दर्ज करें (100–500 शब्द)..."
          className="w-full text-sm bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-slate-200 focus:outline-none focus:border-amber-500 font-devanagari leading-relaxed resize-y"
        />
        <div className="absolute right-3 bottom-3 text-[10px] font-mono text-slate-500 bg-slate-900/80 px-2 py-0.5 rounded">
          {storyText.trim().split(/\s+/).length} words • {storyText.length} chars
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
        <div className="text-xs text-slate-400 font-mono">
          Model: <span className="text-amber-400 font-bold">XTTS v2</span> • GPU:{' '}
          <span className="text-emerald-400 font-bold">NVIDIA H200 (18GB)</span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onAnalyze(storyText)}
            disabled={isAnalyzing || isSynthesizing || !storyText.trim()}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 font-semibold text-xs flex items-center gap-2 border border-slate-700 transition-all disabled:opacity-50"
          >
            {isAnalyzing ? (
              <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
            ) : (
              <Sparkles className="w-4 h-4 text-amber-400" />
            )}
            <span>
              {isAnalyzing
                ? 'विश्लेषण चल रहा है...'
                : '1. पात्र एवं भाव विश्लेषण (Analyze & Direct)'}
            </span>
          </button>

          <button
            onClick={onSynthesize}
            disabled={!hasAnalyzed || isSynthesizing || isAnalyzing}
            className="px-5 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs flex items-center gap-2 transition-all disabled:opacity-40 shadow-lg shadow-amber-500/10 active:scale-95"
          >
            {isSynthesizing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <PlayCircle className="w-4 h-4" />
            )}
            <span>
              {isSynthesizing
                ? 'ध्वनि निर्माण जारी है...'
                : '2. पूर्ण ऑडियो निर्माण (Synthesize Audio)'}
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};
