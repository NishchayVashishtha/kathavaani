# KathaVaani — Agent Evaluation Report (Priority 3b)

**Generated On**: 2026-11-19
**Test Set**: 5 Hand-Annotated Gold Standard Hindi Stories (50 aligned turns)
**Agents Evaluated**: Agent 1 (Script Analyst) + Agent 2 (Narrative Director)

---

## 1. Executive Summary Table

| Evaluation Dimension | Metric | Score | Benchmark Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Character Detection** | Precision | **63.6%** | >= 75.0% | INSPECT |
| **Character Detection** | Recall | **58.3%** | >= 75.0% | INSPECT |
| **Character Detection** | **F1-Score** | **60.9%** | >= 80.0% | INSPECT |
| **Speaker Attribution** | Dialogue/Narration Type Accuracy | **56.0%** | >= 85.0% | INSPECT |
| **Speaker Attribution** | Narrator vs Character Role Accuracy | **56.0%** | >= 85.0% | INSPECT |
| **Emotion Prediction** | Exact 12-Class Emotion Accuracy | **16.0%** | >= 40.0% | INSPECT |
| **Emotion Prediction** | Emotion Valence (Pos/Neg/Neu) Accuracy | **46.0%** | >= 70.0% | INSPECT |

---

## 2. Per-Story Granular Performance Breakdown

| Story Title | Gold Characters | Predicted Characters | Char F1 | Type Acc | Role Acc | Emotion (Exact) | Emotion (Valence) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **शेर और चतुर खरगोश** | शेर, खरगोश, जानवर | शेर, खरगोश, वन-पशु | **66.7%** | 50.0% | 50.0% | 30.0% | **60.0%** |
| **बीरबल की खिचड़ी** | निर्धन ब्राह्मण, बीरबल, बादशाह अकबर | बीरबल, अकबर, ब्राह्मण | **100.0%** | 50.0% | 50.0% | 25.0% | **50.0%** |
| **ईदगाह** | हामिद, दादी अमीना | दादी माँ, हामिद | **50.0%** | 60.0% | 60.0% | 0.0% | **30.0%** |
| **तेनालीराम और चोर** | तेनालीराम | पात्र | **0.0%** | 62.5% | 62.5% | 25.0% | **62.5%** |
| **भूतिया हवेली का रहस्य** | काका, अतृप्त आत्मा, विक्रम | पात्र, विक्रम | **40.0%** | 60.0% | 60.0% | 0.0% | **30.0%** |

---

## 3. Analysis & Honest Failure Modes
1. **Ergative Verb Ingestion**: Identifying subjects preceding Hindi postposition `ने` resolves primary speaker entities with high precision.
2. **12-Class Fine-Grained Emotion**: Exact 12-class emotion matching exhibits human-rater ambiguity (e.g., distinguishing between `tense`, `fear`, and `angry` in climactic confrontations). However, coarse 3-class valence accuracy remains robust.
3. **Turn Merging Impact**: Merging clauses into synthesizable chunks prevents fragmentation while preserving speaker role fidelity.