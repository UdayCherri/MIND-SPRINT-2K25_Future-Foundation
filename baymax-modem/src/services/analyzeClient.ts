/**
 * BAYMAX Analyze API Client
 * Handles communication with the backend /analyze endpoint
 */

const API_BASE_URL = import.meta.env.VITE_ANALYZE_API_URL || "http://localhost:8100";

// ============================================
// Type Definitions
// ============================================

export interface AnalyzeRequest {
  user_id: string;
  date: string;
  texts: string[];
  audio_embedding: number[];
  behavior: number[];
  include_explain?: boolean;
  max_trigger_lag_days?: number;
}

export interface PatternConfidence {
  pattern: string;
  probability: number;
}

export interface InsightCard {
  title: string;
  subtitle?: string;
  description: string;
  severity: "low" | "moderate" | "high";
  recommendation: string;
}

export interface FeatureContributor {
  feature: string;
  score: number;
  friendly_name?: string;
  direction?: string;
}

export interface ModalityContributions {
  audio: number;
  text: number;
  behavior: number;
}

export interface Explainability {
  top_contributors: FeatureContributor[];
  insight_card: InsightCard;
  modality_contributions: ModalityContributions;
  error?: string;
  details?: string;
}

export interface TriggerResponse {
  trigger: string;
  target: string;
  lag_days: number;
  correlation: number;
  pvalue: number;
  confidence: string;
  interpretation: string;
}

export interface AnalyzeResponse {
  user_id: string;
  date: string;
  patterns: PatternConfidence[];
  burnout_score: number;
  explainability?: Explainability;
  triggers?: TriggerResponse[];
  meta: {
    processing_time_s: number;
    model_loaded?: string;
    scaler_loaded?: string;
  };
}

export interface AudioUploadResponse {
  audio_embedding: number[];
  duration_seconds: number;
  sample_rate: number;
}

export interface AdapterRequest {
  user_id: string;
  audio_samples: number[][];
  text_samples: string[][];
  behavior_samples: number[][];
  target_patterns: number[][];
  target_burnout: number[];
  epochs?: number;
  batch_size?: number;
}

export interface AdapterStatusResponse {
  user_id: string;
  status: string;
  adapter_size_kb: number;
  training_samples: number;
  message: string;
}

// ============================================
// API Client Functions
// ============================================

/**
 * Check API health status
 */
export async function checkHealth(): Promise<{ status: string; model: string | null }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Main analyze endpoint - combines text, audio, behavior → predictions + explainability
 */
export async function analyze(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Add Authorization header if using JWT/API keys
      // "Authorization": `Bearer ${getToken()}`
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Analyze API error (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Upload audio file and get embedding
 */
export async function uploadAudio(audioBlob: Blob, filename = "recording.webm"): Promise<AudioUploadResponse> {
  const formData = new FormData();
  formData.append("file", audioBlob, filename);

  const response = await fetch(`${API_BASE_URL}/audio_upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Audio upload error (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Train personalized adapter for a user
 */
export async function updateAdapter(request: AdapterRequest): Promise<AdapterStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/adapter/update_adapter`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Adapter training error (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Load user's personalized adapter
 */
export async function loadAdapter(userId: string): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/adapter/load_adapter/${userId}`);

  if (!response.ok) {
    if (response.status === 404) {
      return { status: "not_found", message: "No adapter found for this user" };
    }
    const errorText = await response.text();
    throw new Error(`Adapter loading error (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Check if user has a trained adapter
 */
export async function checkAdapterStatus(userId: string): Promise<{
  user_id: string;
  has_adapter: boolean;
  adapter_path?: string;
  adapter_size_kb?: number;
  message?: string;
}> {
  const response = await fetch(`${API_BASE_URL}/adapter/status/${userId}`);

  if (!response.ok) {
    throw new Error(`Adapter status check failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================
// Helper Functions
// ============================================

/**
 * Get severity level and color from probability
 */
export function getSeverity(probability: number): { level: "low" | "moderate" | "high"; color: string } {
  if (probability < 0.3) {
    return { level: "low", color: "#10b981" }; // green
  } else if (probability < 0.6) {
    return { level: "moderate", color: "#f59e0b" }; // yellow
  } else {
    return { level: "high", color: "#ef4444" }; // red
  }
}

/**
 * Format pattern name for display
 */
export function formatPatternName(pattern: string): string {
  return pattern
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Create mock audio embedding for testing (256 floats)
 */
export function createMockAudioEmbedding(variance = 0.5): number[] {
  return Array.from({ length: 256 }, () => (Math.random() - 0.5) * variance);
}

/**
 * Create behavior vector from UI inputs
 * [avg_sleep, sleep_var, screen_time, sessions, breaks, activity]
 */
export function createBehaviorVector(inputs: {
  sleepHours?: number;
  avgSleep?: number;
  sleepVar?: number;
  screenTime?: number;
  sessions?: number;
  breaks?: number;
  activity?: number;
  activityLevel?: string;
}): number[] {
  // Support both avgSleep and sleepHours
  const avgSleep = inputs.avgSleep ?? inputs.sleepHours ?? 7.0;
  const sleepVar = inputs.sleepVar ?? 0.5;
  const screenTime = inputs.screenTime ?? 6.0;
  const sessions = inputs.sessions ?? 30;
  const breaks = inputs.breaks ?? 0.4;
  
  // Convert activity level string to numeric if provided
  let activity = inputs.activity ?? 30;
  if (inputs.activityLevel) {
    const activityMap: Record<string, number> = {
      'Low': 15,
      'Moderate': 30,
      'High': 60
    };
    activity = activityMap[inputs.activityLevel] ?? 30;
  }
  
  return [
    avgSleep,
    sleepVar,
    screenTime,
    sessions,
    breaks,
    activity,
  ];
}

/**
 * Get current date in YYYY-MM-DD format
 */
export function getCurrentDate(): string {
  const now = new Date();
  return now.toISOString().split("T")[0];
}

// ============================================
// Default Export (for convenience)
// ============================================

export const analyzeClient = {
  analyze,
  uploadAudio,
  updateAdapter,
  loadAdapter,
  checkAdapterStatus,
  checkHealth,
  getSeverity,
  formatPatternName,
  createMockAudioEmbedding,
  createBehaviorVector,
  getCurrentDate,
};

export default analyzeClient;
