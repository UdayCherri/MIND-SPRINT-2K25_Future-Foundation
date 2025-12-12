import { motion } from 'motion/react';
import { Page } from '../App';
import { Navigation } from '../components/Navigation';
import { AlertTriangle, Zap, Moon, Activity, TrendingDown, ArrowRight, Calendar, Brain } from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { formatPatternName } from '../services/analyzeClient';

interface InsightsPageProps {
  onNavigate: (page: Page) => void;
  onLogout: () => void;
}

export function InsightsPage({ onNavigate, onLogout }: InsightsPageProps) {
  const { currentAnalysis, insights, analysisHistory } = useUser();

  // Generate detected patterns from real data
  const detectedPatterns = currentAnalysis?.patterns.length
    ? currentAnalysis.patterns
        .filter(p => p.probability > 0.4)
        .map(p => ({
          title: formatPatternName(p.pattern),
          confidence: p.probability,
          icon: p.probability > 0.7 ? AlertTriangle : p.probability > 0.6 ? TrendingDown : Zap,
          color: p.probability > 0.7 ? 'from-red-500 to-pink-500' : p.probability > 0.6 ? 'from-orange-500 to-red-500' : 'from-yellow-500 to-orange-500',
          iconColor: p.probability > 0.7 ? 'text-red-400' : p.probability > 0.6 ? 'text-orange-400' : 'text-yellow-400',
          explanation: `Pattern detected with ${(p.probability * 100).toFixed(0)}% confidence based on your recent behavioral and emotional data.`,
          action: p.probability > 0.7 
            ? 'Consider seeking support or taking immediate steps to address this pattern.' 
            : 'Monitor this pattern and take preventive measures.',
          severity: (p.probability > 0.7 ? 'high' : p.probability > 0.6 ? 'high' : 'medium') as 'high' | 'medium' | 'low',
        }))
    : [
        {
          title: 'No Patterns Detected',
          confidence: 0,
          icon: Brain,
          color: 'from-gray-500 to-gray-600',
          iconColor: 'text-gray-400',
          explanation: 'Run an analysis to detect wellness patterns.',
          action: 'Go to Dashboard and complete an analysis.',
          severity: 'low' as const,
        },
      ];

  const correlationMap = currentAnalysis?.explainability?.top_contributors?.slice(0, 3).map(c => ({
    trigger: c.friendly_name || c.feature,
    response: 'Burnout Risk',
    strength: c.score,
  })) || [
    { trigger: 'Sleep Quality', response: 'Morning Energy', strength: 0.78 },
    { trigger: 'Screen Time', response: 'Cognitive Load', strength: 0.65 },
    { trigger: 'Physical Activity', response: 'Mood Stability', strength: 0.72 },
  ];

  const patternTimeline = analysisHistory?.analyses?.slice(-6).map((a, i) => ({
    date: new Date(a.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    pattern: a.burnout_score > 0.7 ? 'High Burnout Risk' : a.burnout_score > 0.4 ? 'Moderate Risk' : 'Low Risk',
    type: (a.burnout_score > 0.7 ? 'negative' : a.burnout_score > 0.4 ? 'negative' : 'positive') as 'positive' | 'negative',
  })) || [
    { date: 'Dec 7', pattern: 'Analysis Pending', type: 'positive' as const },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10">
        <Navigation activeTab="Insights" onNavigate={onNavigate} onLogout={onLogout} />
        
        <main className="container mx-auto px-4 py-6 max-w-[1600px]">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl mb-2">Insights Engine</h1>
            <p className="text-white/60">AI-powered pattern detection and actionable recommendations</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Detected Patterns */}
            <div className="lg:col-span-2 space-y-4">
              <h2 className="text-2xl mb-4">Detected Patterns</h2>
              
              {detectedPatterns.map((pattern, index) => {
                const Icon = pattern.icon;
                return (
                  <motion.div
                    key={pattern.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="rounded-2xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-xl hover:shadow-2xl transition-shadow"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className={`p-3 rounded-xl bg-slate-800/50 border border-white/10 shadow-lg`}>
                          <Icon className={`w-6 h-6 ${pattern.iconColor}`} />
                        </div>
                        <div>
                          <h3 className="text-xl text-white/90">{pattern.title}</h3>
                          <div className="flex items-center space-x-2 mt-1">
                            <div className="text-sm text-white/60">Confidence:</div>
                            <div className={`text-sm bg-gradient-to-r ${pattern.color} bg-clip-text text-transparent`}>
                              {Math.round(pattern.confidence * 100)}%
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Severity Badge */}
                      <div className={`px-3 py-1 rounded-full text-xs ${
                        pattern.severity === 'high'
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                          : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                      }`}>
                        {pattern.severity === 'high' ? 'High Priority' : 'Medium Priority'}
                      </div>
                    </div>

                    {/* Explanation */}
                    <div className="mb-4 p-4 rounded-xl bg-slate-800/30 border border-white/5">
                      <div className="text-sm text-white/70">{pattern.explanation}</div>
                    </div>

                    {/* Action */}
                    <div className={`p-4 rounded-xl bg-gradient-to-r ${pattern.color} bg-opacity-10 border border-white/10`}>
                      <div className="flex items-start space-x-2">
                        <Activity className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: pattern.iconColor.replace('text-', '#') }} />
                        <div>
                          <div className="text-sm mb-1 text-white/90">Suggested Action</div>
                          <div className="text-sm text-white/70">{pattern.action}</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* Right Column - Correlation Map & Timeline */}
            <div className="space-y-6">
              {/* Correlation Map */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="rounded-2xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-xl"
              >
                <h3 className="text-xl mb-4">Trigger → Response Map</h3>
                <p className="text-sm text-white/60 mb-6">Identified cause-effect patterns</p>
                
                <div className="space-y-4">
                  {correlationMap.map((correlation, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.4 + index * 0.1 }}
                      className="relative"
                    >
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 p-3 rounded-lg bg-slate-800/50 border border-white/10">
                          <div className="text-sm text-white/80">{correlation.trigger}</div>
                        </div>
                        <ArrowRight className="w-5 h-5 text-teal-400 flex-shrink-0" />
                        <div className="flex-1 p-3 rounded-lg bg-slate-800/50 border border-white/10">
                          <div className="text-sm text-white/80">{correlation.response}</div>
                        </div>
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="text-white/50">Correlation strength:</span>
                        <span className="text-teal-400">{Math.round(correlation.strength * 100)}%</span>
                      </div>
                      <div className="mt-1 h-1 rounded-full bg-slate-800/50 overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-teal-400 to-blue-400"
                          initial={{ width: 0 }}
                          animate={{ width: `${correlation.strength * 100}%` }}
                          transition={{ duration: 1, delay: 0.6 + index * 0.1 }}
                        />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>

              {/* Pattern Timeline */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="rounded-2xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-xl"
              >
                <h3 className="text-xl mb-4">Pattern Timeline</h3>
                <p className="text-sm text-white/60 mb-6">Last 30 days</p>
                
                <div className="space-y-3">
                  {patternTimeline.map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5 + index * 0.05 }}
                      className="flex items-center space-x-3"
                    >
                      <div className={`w-2 h-2 rounded-full ${
                        item.type === 'positive' ? 'bg-teal-400' : 'bg-orange-400'
                      } shadow-lg ${
                        item.type === 'positive' ? 'shadow-teal-400/50' : 'shadow-orange-400/50'
                      }`}></div>
                      <div className="flex-1 flex items-center justify-between p-3 rounded-lg bg-slate-800/30 border border-white/5">
                        <div className="flex items-center space-x-2">
                          <Calendar className="w-4 h-4 text-white/40" />
                          <span className="text-sm text-white/60">{item.date}</span>
                        </div>
                        <span className={`text-sm ${
                          item.type === 'positive' ? 'text-teal-400' : 'text-orange-400'
                        }`}>
                          {item.pattern}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
