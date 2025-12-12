import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import type { AnalyzeResponse } from '../services/analyzeClient';

// ============================================
// Type Definitions
// ============================================

export interface User {
  user_id: string;
  email: string;
  name: string;
}

export interface TimelineData {
  dates: string[];
  burnout_scores: number[];
  patterns: Record<string, number[]>;
}

export interface InsightsSummary {
  avg_burnout: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  top_patterns: Array<{ pattern: string; count: number }>;
  top_contributors: Array<{ feature: string; avg_score: number }>;
  total_analyses: number;
}

export interface AnalysisHistory {
  user_id: string;
  analyses: Array<{
    date: string;
    timestamp: string;
    patterns: any[];
    burnout_score: number;
    explainability?: any;
    meta: any;
  }>;
}

interface UserContextType {
  user: User | null;
  currentAnalysis: AnalyzeResponse | null;
  analysisHistory: AnalysisHistory | null;
  timelineData: TimelineData | null;
  insights: InsightsSummary | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (email: string, password: string, name: string) => Promise<boolean>;
  logout: () => void;
  setCurrentAnalysis: (analysis: AnalyzeResponse) => void;
  refreshHistory: () => Promise<void>;
  refreshTimeline: (days?: number) => Promise<void>;
  refreshInsights: () => Promise<void>;
}

// ============================================
// Context Creation
// ============================================

const UserContext = createContext<UserContextType | null>(null);

const API_BASE_URL = import.meta.env.VITE_ANALYZE_API_URL || "http://localhost:8100";

// ============================================
// Provider Component
// ============================================

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistory | null>(null);
  const [timelineData, setTimelineData] = useState<TimelineData | null>(null);
  const [insights, setInsights] = useState<InsightsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Login function
  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data.success && data.user) {
        setUser(data.user);
        // Load user data after login
        await refreshHistory(data.user.user_id);
        await refreshTimeline(30, data.user.user_id);
        await refreshInsights(data.user.user_id);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Register function
  const register = async (email: string, password: string, name: string): Promise<boolean> => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name }),
      });

      const data = await response.json();

      if (data.success && data.user) {
        setUser(data.user);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Register error:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    setUser(null);
    setCurrentAnalysis(null);
    setAnalysisHistory(null);
    setTimelineData(null);
    setInsights(null);
  };

  // Refresh history
  const refreshHistory = async (userId?: string) => {
    const targetUserId = userId || user?.user_id;
    if (!targetUserId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/data/history/${targetUserId}?limit=30`);
      if (response.ok) {
        const data = await response.json();
        setAnalysisHistory(data);
      }
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  // Refresh timeline
  const refreshTimeline = async (days: number = 30, userId?: string) => {
    const targetUserId = userId || user?.user_id;
    if (!targetUserId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/data/timeline/${targetUserId}?days=${days}`);
      if (response.ok) {
        const data = await response.json();
        setTimelineData(data);
      }
    } catch (error) {
      console.error('Failed to fetch timeline:', error);
    }
  };

  // Refresh insights
  const refreshInsights = async (userId?: string) => {
    const targetUserId = userId || user?.user_id;
    if (!targetUserId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/data/insights/${targetUserId}`);
      if (response.ok) {
        const data = await response.json();
        setInsights(data);
      }
    } catch (error) {
      console.error('Failed to fetch insights:', error);
    }
  };

  // Auto-refresh data when current analysis changes
  useEffect(() => {
    if (currentAnalysis && user) {
      refreshHistory();
      refreshTimeline();
      refreshInsights();
    }
  }, [currentAnalysis]);

  return (
    <UserContext.Provider
      value={{
        user,
        currentAnalysis,
        analysisHistory,
        timelineData,
        insights,
        isLoading,
        login,
        register,
        logout,
        setCurrentAnalysis,
        refreshHistory,
        refreshTimeline,
        refreshInsights,
      }}
    >
      {children}
    </UserContext.Provider>
  );
}

// ============================================
// Hook for consuming context
// ============================================

export function useUser() {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}
