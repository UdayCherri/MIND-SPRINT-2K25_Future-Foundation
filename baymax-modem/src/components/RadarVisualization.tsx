import { motion } from 'motion/react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { useUser } from '../contexts/UserContext';
import { formatPatternName } from '../services/analyzeClient';

export function RadarVisualization() {
  const { currentAnalysis } = useUser();

  // Use real data if available, otherwise show default view
  const data = currentAnalysis?.patterns.length
    ? currentAnalysis.patterns.map(p => ({
        metric: formatPatternName(p.pattern),
        value: Math.round(p.probability * 100),
        fullMark: 100,
      }))
    : [
        { metric: 'Stress Loop Intensity', value: 45, fullMark: 100 },
        { metric: 'Burnout Trajectory', value: 38, fullMark: 100 },
        { metric: 'Emotional Volatility', value: 42, fullMark: 100 },
        { metric: 'Sleep Disruption', value: 55, fullMark: 100 },
        { metric: 'Cognitive Decline', value: 35, fullMark: 100 },
        { metric: 'Social Withdrawal', value: 48, fullMark: 100 },
        { metric: 'Physical Symptoms', value: 32, fullMark: 100 },
      ];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="relative rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-8 shadow-2xl"
    >
      {/* Glow Effect */}
      <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-teal-500/5 to-blue-500/5 blur-xl"></div>
      
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl">Pattern Radar</h2>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${currentAnalysis ? 'bg-green-400' : 'bg-gray-400'} animate-pulse shadow-lg ${currentAnalysis ? 'shadow-green-400/50' : 'shadow-gray-400/50'}`}></div>
            <span className="text-sm text-white/60">{currentAnalysis ? 'Live Data' : 'Waiting for Analysis'}</span>
          </div>
        </div>

        <div className="relative">
          {/* Center Glow */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-gradient-to-br from-teal-400/20 to-blue-500/20 rounded-full blur-3xl"></div>
          
          <ResponsiveContainer width="100%" height={450}>
            <RadarChart data={data}>
              <defs>
                <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#39E6C6" stopOpacity={0.8} />
                  <stop offset="50%" stopColor="#3F77FF" stopOpacity={0.6} />
                  <stop offset="100%" stopColor="#9B5CFF" stopOpacity={0.4} />
                </linearGradient>
              </defs>
              <PolarGrid 
                stroke="rgba(255, 255, 255, 0.1)" 
                strokeWidth={1}
              />
              <PolarAngleAxis 
                dataKey="metric" 
                tick={{ fill: '#E8EBF0', fontSize: 12 }}
                stroke="rgba(255, 255, 255, 0.2)"
              />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 100]} 
                tick={{ fill: '#E8EBF0', fontSize: 10 }}
                stroke="rgba(255, 255, 255, 0.2)"
              />
              <Radar
                name="Wellness Metrics"
                dataKey="value"
                stroke="#39E6C6"
                fill="url(#radarGradient)"
                fillOpacity={0.6}
                strokeWidth={2}
                dot={{ r: 4, fill: '#39E6C6', strokeWidth: 0 }}
                className="drop-shadow-[0_0_10px_rgba(57,230,198,0.6)]"
              />
            </RadarChart>
          </ResponsiveContainer>

          {/* Animated Dots */}
          <motion.div
            className="absolute top-1/4 right-1/4 w-3 h-3 rounded-full bg-teal-400 shadow-lg shadow-teal-400/50"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.6, 1, 0.6],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          ></motion.div>
          <motion.div
            className="absolute bottom-1/3 left-1/4 w-2 h-2 rounded-full bg-blue-400 shadow-lg shadow-blue-400/50"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.6, 1, 0.6],
            }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5,
            }}
          ></motion.div>
        </div>
      </div>
    </motion.div>
  );
}
