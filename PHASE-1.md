# **Mental Wellness Pattern Analyzer - Phase 1 Submission**

### **Problem Framing + Dataset Strategy**

---

## **1. Problem Statement**

**Mental wellness drift emerges from subtle changes in emotional expression, cognitive load, behavioral rhythms, and personality tendencies. These micro-patterns are difficult for individuals to self-detect. Our goal is to build an AI system that reveals these early, enabling non-clinical, proactive mental wellness awareness.**

This system goes beyond mood tracking it identifies **cognitive signatures**, maps **behavioral rhythms**, and predicts **emotional trajectories**, creating a _cognitive observability framework_ for everyday mental wellbeing.

---

## **2. Core Vision**

Most existing tools measure mood.
We measure **patterns that produce the mood**.

Our system analyzes:

- **Emotional micro-patterns**
- **Cognitive load signatures**
- **Behavioral rhythm cycles**
- **Personality-modulated tendencies**

This allows us to uncover hidden dynamics like stress loops, burnout buildup, sleep-emotion mismatch, and personalized triggers that impact day-to-day wellness.

---

## **3. Dataset Strategy**

We adopt a **multi-modal dataset approach**, combining emotional, cognitive, behavioral, and personality-contextual data.

### **Datasets Used**

- **Emotional Signals**

  - RAVDESS
  - GoEmotions (Google)
  - DailyDialog
  - Emotional Tone Reddit datasets

- **Cognitive Load Signals**

  - Writing complexity datasets
  - Semantic coherence datasets
  - Readability & cognitive load corpora

- **Behavioral Rhythms**

  - Sleep-EDF
  - Activity / smartwatch datasets
  - Time-series human activity datasets
  - WESAD (stress + physiological patterns)

### **Dataset Purpose Table**

| Insight You Want            | Dataset Source               | Why It Matters                                                            |
| --------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| **Emotional tone**          | RAVDESS                      | Controlled speech → strong pretraining foundation for emotion recognition |
| **Natural emotional drift** | GoEmotions, DailyDialog      | Captures realistic emotional variability over time                        |
| **Behavioral rhythms**      | Sleep-EDF, activity datasets | Helps model circadian patterns, energy cycles, and wellness rhythms       |
| **Personality context**     | User-input MBTI              | Sets personalization priors for adaptation                                |

---

## **4. Key Concepts Incorporated**

- **Federated Learning:** Enables weekly per-user personalization without sharing raw data.
- **MBTI-based Personalization:** Personality acts as a soft prior for behavioral tendencies and emotional response patterns.

---

## **5. Target Patterns the System Will Detect**

### **Behavioral & Emotional Patterns**

- **Stress Loops:** Repeated negative spirals triggered by specific events or routines.
- **Burnout Trajectory:** Gradual decline in energy, motivation, and emotional stability.
- **Positive Habit Reinforcement:** Activities that reliably improve wellness.

### **Rhythmic & Cognitive Patterns**

- **Sleep-Emotion Mismatch:** Poor sleep correlates with emotional volatility.
- **Cognitive Overload Threshold:** Writing complexity drops → mental fatigue rising.
- **Social Withdrawal Signatures:** Reduced communication tied to emotional dips.
- **Routine Fracturing:** Sudden breaks in normal behavioral rhythms.

### **Personality-Linked Patterns**

- **MBTI-modulated Stress Response:** Personality traits affect how users react to stressors and recovery cycles.

---

## **6. Feature Design**

### **Emotional Micro-Patterns**

- Sentiment-shift velocity
- Emotional inertia (duration before mood changes)
- Volatility index (speed of emotional swings)
- Semantic dissonance (tone vs reported feeling mismatch)

### **Cognitive Load Signatures**

- Writing complexity decay
- Reduced vocabulary diversity
- Attention-span reduction patterns
- Repetitive language formations
- Break irregularity spikes

### **Behavioral Rhythm Maps**

- Sleep → activity → stress → recovery cycles
- Weekly emotional periodicity
- Burnout wave detection
- Social energy oscillations

### **Trigger-Response Correlation Engine**

Examples:

- Late-night screen-time → decreased next-day emotional stability
- Low-protein meal → increased irritability in journaling tone
- Coding late → negative emotion drift
- Post-deadline fatigue → 2-day recovery cycle

---

## **7. Personalization Strategy**

- A **global base model** learns general emotional + behavioral patterns.
- Each user receives a **lightweight personalization adapter**, updated weekly.
- This allows:

  - better accuracy with minimal user data
  - privacy (no raw data sharing)
  - modeling of unique personal rhythms and tendencies

The system adapts continuously, making insights more precise over time.

---

## **8. What Makes This System Different**

- We are _not_ tracking emotions.
- We are _not_ analyzing isolated text.

We are **discovering hidden cognitive patterns** that shape long-term mental wellness:

- Mapping **behavioral rhythm**
- Identifying **micro-patterns**
- Predicting **emotional trajectory**
- Delivering **human-readable insights**

This positions the system as a **non-clinical cognitive observability tool**, not a mental health diagnostic.

---
