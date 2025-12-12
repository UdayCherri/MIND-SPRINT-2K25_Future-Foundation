import { motion } from 'motion/react';
import { AlertCircle, TrendingUp, Shield, Zap, Brain, Activity } from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { formatPatternName } from '../services/analyzeClient';

export function InsightCards() {
  const { currentAnalysis, insights } = useUser();

  // Generate insights from real data if available
  const insightsList = currentAnalysis ? [
    // Top Pattern
    ...(currentAnalysis.patterns.length > 0 ? [{
      title: formatPatternName(currentAnalysis.patterns[0].pattern),
      metric: `${(currentAnalysis.patterns[0].probability * 100).toFixed(0)}% Confidence`,
      insight: `This pattern was detected with high probability in your recent analysis.`,
      icon: AlertCircle,
      color: currentAnalysis.patterns[0].probability > 0.6 ? 'from-red-500 to-orange-500' : 'from-yellow-500 to-orange-500',
      glow: currentAnalysis.patterns[0].probability > 0.6 ? 'shadow-red-500/20' : 'shadow-yellow-500/20',
      iconColor: currentAnalysis.patterns[0].probability > 0.6 ? 'text-red-400' : 'text-yellow-400',
    }] : []),
    // Burnout Risk
    {
      title: 'Burnout Risk',
      metric: `${(currentAnalysis.burnout_score * 100).toFixed(0)}%`,
      insight: currentAnalysis.burnout_score > 0.7 ? 'High risk detected - consider taking breaks' : 
               currentAnalysis.burnout_score > 0.4 ? 'Moderate risk - monitor your wellness' :
               'Low risk - keep up the good work!',
      icon: TrendingUp,
      color: currentAnalysis.burnout_score > 0.7 ? 'from-red-500 to-pink-500' : 
             currentAnalysis.burnout_score > 0.4 ? 'from-yellow-500 to-orange-500' : 
             'from-green-500 to-teal-500',
      glow: currentAnalysis.burnout_score > 0.7 ? 'shadow-red-500/20' : 
            currentAnalysis.burnout_score > 0.4 ? 'shadow-yellow-500/20' : 
            'shadow-green-500/20',
      iconColor: currentAnalysis.burnout_score > 0.7 ? 'text-red-400' : 
                 currentAnalysis.burnout_score > 0.4 ? 'text-yellow-400' : 
                 'text-green-400',
    },
    // Top Contributor
    ...(currentAnalysis.explainability?.top_contributors?.length > 0 ? [{
      title: 'Key Factor',
      metric: currentAnalysis.explainability.top_contributors[0].friendly_name || currentAnalysis.explainability.top_contributors[0].feature,
      insight: `Impact score: ${(currentAnalysis.explainability.top_contributors[0].score * 100).toFixed(0)}%`,
      icon: Brain,
      color: 'from-purple-500 to-blue-500',
      glow: 'shadow-purple-500/20',
      iconColor: 'text-purple-400',
    }] : []),
    // Analysis Trend
    ...(insights ? [{
      title: 'Overall Trend',
      metric: insights.trend.charAt(0).toUpperCase() + insights.trend.slice(1),
      insight: `Average burnout: ${(insights.avg_burnout * 100).toFixed(0)}% over ${insights.total_analyses} analyses`,
      icon: Activity,
      color: insights.trend === 'increasing' ? 'from-red-500 to-orange-500' : 
             insights.trend === 'decreasing' ? 'from-green-500 to-teal-500' : 
             'from-blue-500 to-indigo-500',
      glow: insights.trend === 'increasing' ? 'shadow-red-500/20' : 
            insights.trend === 'decreasing' ? 'shadow-green-500/20' : 
            'shadow-blue-500/20',
      iconColor: insights.trend === 'increasing' ? 'text-red-400' : 
                 insights.trend === 'decreasing' ? 'text-green-400' : 
                 'text-blue-400',
    }] : []),
  ] : [
    // Default insights when no analysis
    {
      title: 'No Analysis Yet',
      metric: 'Run Analysis',
      insight: 'Fill in the form and click Analyze to see insights.',
      icon: Shield,
      color: 'from-gray-500 to-gray-600',
      glow: 'shadow-gray-500/20',
      iconColor: 'text-gray-400',
    },
  ];

  // Ensure we always show 3-4 cards
  const displayInsights = insightsList.slice(0, 4);

  return (
    <div className="space-y-4">
      {displayInsights.map((insight, index) => {
        const Icon = insight.icon;
        
        return (
          <motion.div
            key={insight.title}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className="relative rounded-2xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-5 shadow-xl hover:shadow-2xl transition-shadow group"
          >
            {/* Glow Effect */}
            <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${insight.color} opacity-0 group-hover:opacity-10 transition-opacity blur-xl`}></div>
            
            <div className="relative z-10">
              {/* Icon */}
              <div className="flex items-start justify-between mb-3">
                <div className={`p-2 rounded-xl bg-slate-800/50 border border-white/10 ${insight.glow} shadow-lg`}>
                  <Icon className={`w-5 h-5 ${insight.iconColor}`} />
                </div>
                <div className={`w-2 h-2 rounded-full ${insight.iconColor} animate-pulse ${insight.glow}`}></div>
              </div>

              {/* Content */}
              <h3 className="text-white/90 mb-2">{insight.title}</h3>
              <div className={`text-xl mb-2 bg-gradient-to-r ${insight.color} bg-clip-text text-transparent`}>
                {insight.metric}
              </div>
              <p className="text-sm text-white/60">{insight.insight}</p>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
