/**
 * BAYMAX API Integration Service
 * Connects frontend to Python backend endpoints
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// API Endpoints
const ENDPOINTS = {
  // Text Embedding API
  health: `${API_BASE_URL}/health`,
  embedText: `${API_BASE_URL}/embed_text`,
  dayFeatures: `${API_BASE_URL}/day_features`,
  buildDayVector: `${API_BASE_URL}/build_day_vector`,
  multiDayDrift: `${API_BASE_URL}/multi_day_drift`,
  semanticDrift: `${API_BASE_URL}/semantic_drift`,
  
  // Fusion Model API (to be implemented)
  analyze: `${API_BASE_URL}/analyze`,
};

// Types
export interface DailyInput {
  journalText: string;
  sleepHours: number;
  screenTime: number;
  activityLevel: 'Low' | 'Moderate' | 'High' | 'Very High';
  audioFile?: File;
}

export interface AnalysisResult {
  patterns: {
    socialWithdrawal: number;
    sleepDisruption: number;
    emotionalVolatility: number;
    cognitiveDecline: number;
  };
  burnout: number;
  metrics: {
    stressLoopIntensity: number;
    burnoutTrajectory: number;
    emotionalVolatility: number;
    sleepEmotionAlignment: number;
    cognitiveLoadLevel: number;
    routineConsistency: number;
    socialEngagement: number;
  };
  insights: string[];
  topFeatures: string[];
  recommendation: string;
}

export interface TextEmbeddingResponse {
  embeddings: number[][];
  shape: number[];
}

export interface DayFeaturesResponse {
  lexical_diversity: number;
  avg_sentence_length: number;
  sentence_count: number;
}

export interface SemanticDriftResponse {
  drift_score: number;
  interpretation: string;
}

// API Functions
export const baymaxApi = {
  /**
   * Check if backend is healthy
   */
  async checkHealth(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await fetch(ENDPOINTS.health);
      if (!response.ok) throw new Error('Backend not responding');
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Unable to connect to BAYMAX backend. Please ensure Python server is running on port 8000.');
    }
  },

  /**
   * Get text embeddings from journal entry
   */
  async embedText(text: string): Promise<TextEmbeddingResponse> {
    const response = await fetch(ENDPOINTS.embedText, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error('Text embedding failed');
    return await response.json();
  },

  /**
   * Get daily text features (lexical diversity, sentence stats)
   */
  async getDayFeatures(texts: string[]): Promise<DayFeaturesResponse> {
    const response = await fetch(ENDPOINTS.dayFeatures, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts }),
    });
    if (!response.ok) throw new Error('Day features extraction failed');
    return await response.json();
  },

  /**
   * Build complete day feature vector (771 dims)
   */
  async buildDayVector(texts: string[]): Promise<{ feature_vector: number[]; shape: number[] }> {
    const response = await fetch(ENDPOINTS.buildDayVector, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts }),
    });
    if (!response.ok) throw new Error('Day vector building failed');
    return await response.json();
  },

  /**
   * Calculate semantic drift between days
   */
  async calculateDrift(texts1: string[], texts2: string[]): Promise<SemanticDriftResponse> {
    const response = await fetch(ENDPOINTS.semanticDrift, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts_day1: texts1, texts_day2: texts2 }),
    });
    if (!response.ok) throw new Error('Semantic drift calculation failed');
    return await response.json();
  },

  /**
   * Full analysis with fusion model
   * TODO: Implement backend endpoint
   */
  async analyze(input: DailyInput): Promise<AnalysisResult> {
    try {
      // For now, use mock data until backend endpoint is implemented
      console.warn('Using mock analysis - backend /analyze endpoint not yet implemented');
      
      // TODO: When backend /analyze endpoint is ready, use this:
      // const textData = await this.buildDayVector([input.journalText]);
      // const response = await fetch(ENDPOINTS.analyze, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({
      //     journal_text: input.journalText,
      //     text_embeddings: textData.feature_vector,
      //     sleep_hours: input.sleepHours,
      //     screen_time: input.screenTime,
      //     activity_level: input.activityLevel,
      //   }),
      // });
      // return await response.json();

      // Mock fusion model prediction
      const patterns = {
        socialWithdrawal: Math.random() * 0.3 + 0.2,
        sleepDisruption: input.sleepHours < 6 ? Math.random() * 0.5 + 0.5 : Math.random() * 0.3,
        emotionalVolatility: Math.random() * 0.4 + 0.3,
        cognitiveDecline: input.screenTime > 8 ? Math.random() * 0.4 + 0.4 : Math.random() * 0.3,
      };

      const burnout = (patterns.socialWithdrawal + patterns.sleepDisruption + 
                      patterns.emotionalVolatility + patterns.cognitiveDecline) / 4;

      return {
        patterns,
        burnout,
        metrics: {
          stressLoopIntensity: Math.random() * 40 + 40,
          burnoutTrajectory: burnout * 100,
          emotionalVolatility: patterns.emotionalVolatility * 100,
          sleepEmotionAlignment: (1 - patterns.sleepDisruption) * 100,
          cognitiveLoadLevel: patterns.cognitiveDecline * 100,
          routineConsistency: Math.random() * 30 + 60,
          socialEngagement: (1 - patterns.socialWithdrawal) * 100,
        },
        insights: this.generateInsights(patterns, burnout, input),
        topFeatures: this.getTopFeatures(patterns, input),
        recommendation: this.getRecommendation(burnout),
      };
    } catch (error) {
      console.error('Analysis failed:', error);
      throw new Error('Unable to analyze data. Please check backend connection.');
    }
  },

  /**
   * Generate insights from analysis
   */
  generateInsights(patterns: any, burnout: number, input: DailyInput): string[] {
    const insights: string[] = [];

    if (input.sleepHours < 6) {
      insights.push('⚠️ Low sleep detected. Consider earlier bedtime.');
    }
    if (input.screenTime > 8) {
      insights.push('📱 High screen time may impact cognitive function.');
    }
    if (patterns.socialWithdrawal > 0.5) {
      insights.push('🤝 Social engagement appears reduced.');
    }
    if (patterns.emotionalVolatility > 0.6) {
      insights.push('💭 Emotional patterns show variability.');
    }
    if (burnout > 0.7) {
      insights.push('🔴 High burnout risk detected - consider professional support.');
    } else if (burnout > 0.5) {
      insights.push('🟡 Moderate stress levels - implement self-care strategies.');
    } else {
      insights.push('🟢 Overall wellness indicators are positive.');
    }

    return insights;
  },

  /**
   * Get top contributing features
   */
  getTopFeatures(patterns: any, input: DailyInput): string[] {
    const features = [
      { name: 'Sleep Duration', value: input.sleepHours < 6 ? 0.8 : 0.3 },
      { name: 'Screen Time', value: input.screenTime > 8 ? 0.7 : 0.2 },
      { name: 'Social Withdrawal', value: patterns.socialWithdrawal },
      { name: 'Emotional Volatility', value: patterns.emotionalVolatility },
      { name: 'Activity Level', value: input.activityLevel === 'Low' ? 0.6 : 0.2 },
    ];

    return features
      .sort((a, b) => b.value - a.value)
      .slice(0, 3)
      .map(f => f.name);
  },

  /**
   * Get recommendation based on burnout score
   */
  getRecommendation(burnout: number): string {
    if (burnout > 0.7) {
      return 'High stress detected. Consider consulting a mental health professional and implementing immediate stress reduction techniques.';
    } else if (burnout > 0.5) {
      return 'Moderate stress levels. Focus on sleep hygiene, regular exercise, and mindfulness practices.';
    } else if (burnout > 0.3) {
      return 'Wellness indicators are good. Maintain current healthy habits and monitor for changes.';
    } else {
      return 'Excellent wellness profile. Continue your positive lifestyle patterns.';
    }
  },
};
