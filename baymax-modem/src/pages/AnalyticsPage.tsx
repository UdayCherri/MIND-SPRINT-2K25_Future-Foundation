import { motion } from 'motion/react';
import { useState } from 'react';
import { Page } from '../App';
import { Navigation } from '../components/Navigation';
import { AreaChart, Area, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Calendar, Activity } from 'lucide-react';
import { useUser } from '../contexts/UserContext';

interface AnalyticsPageProps {
  onNavigate: (page: Page) => void;
  onLogout: () => void;
}

export function AnalyticsPage({ onNavigate, onLogout }: AnalyticsPageProps) {
  const [timeRange, setTimeRange] = useState<'week' | 'month'>('week');
  const { timelineData, insights, currentAnalysis } = useUser();

  // Multi-modal contribution from current analysis
  const multiModalContribution = currentAnalysis?.explainability?.modality_contributions
    ? [
        { source: 'Text Analysis', contribution: Math.round(currentAnalysis.explainability.modality_contributions.text * 100) },
        { source: 'Voice Patterns', contribution: Math.round(currentAnalysis.explainability.modality_contributions.audio * 100) },
        { source: 'Behavioral Data', contribution: Math.round(currentAnalysis.explainability.modality_contributions.behavior * 100) },
      ]
    : [
        { source: 'Text Analysis', contribution: 35 },
        { source: 'Voice Patterns', contribution: 28 },
        { source: 'Behavioral Data', contribution: 37 },
      ];

  // Timeline data for volatility chart
  const emotionalVolatilityData = timelineData?.dates?.length
    ? timelineData.dates.slice(-7).map((date, i) => ({
        day: new Date(date).toLocaleDateString('en-US', { weekday: 'short' }),
        volatility: Math.round(timelineData.burnout_scores[timelineData.dates.length - 7 + i] * 100),
      }))
    : [
        { day: 'Mon', volatility: 45 },
        { day: 'Tue', volatility: 52 },
        { day: 'Wed', volatility: 48 },
        { day: 'Thu', volatility: 65 },
        { day: 'Fri', volatility: 58 },
        { day: 'Sat', volatility: 42 },
        { day: 'Sun', volatility: 38 },
      ];

  // TODO: Replace with real user behavior rhythm data from backend
  // TODO: Make this time range configurable (24h, 7d, 30d)
  const behaviorRhythmData = [
    { time: '6AM', sleep: 8, activity: 2, screen: 0 },
    { time: '9AM', sleep: 0, activity: 5, screen: 3 },
    { time: '12PM', sleep: 0, activity: 4, screen: 4 },
    { time: '3PM', sleep: 0, activity: 3, screen: 6 },
    { time: '6PM', sleep: 0, activity: 6, screen: 5 },
    { time: '9PM', sleep: 0, activity: 2, screen: 7 },
    { time: '12AM', sleep: 7, activity: 0, screen: 2 },
  ];

  const cognitiveLoadData = emotionalVolatilityData; // Reuse same data for demo

  const colors = ['#39E6C6', '#3F77FF', '#9B5CFF'];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800/90 backdrop-blur-xl border border-white/20 rounded-xl p-3 shadow-xl">
          <p className="text-white/70 text-sm">{payload[0].payload.day || payload[0].payload.time}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: <span className="text-white">{entry.value}</span>
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10">
        <Navigation activeTab="Analytics" onNavigate={onNavigate} onLogout={onLogout} />
        
        <main className="container mx-auto px-4 py-6 max-w-[1600px]">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-4xl mb-2">Analytics</h1>
              <p className="text-white/60">Deep insights into your wellness patterns</p>
            </div>
            
            {/* Time Range Toggle */}
            <div className="flex items-center space-x-2 bg-slate-800/50 rounded-xl p-1 border border-white/10">
              <button
                onClick={() => setTimeRange('week')}
                className={`px-4 py-2 rounded-lg transition-all flex items-center space-x-2 ${
                  timeRange === 'week'
                    ? 'bg-gradient-to-r from-teal-500/20 to-blue-500/20 border border-teal-400/30 text-white'
                    : 'text-white/60 hover:text-white'
                }`}
              >
                <Calendar className="w-4 h-4" />
                <span>Weekly</span>
              </button>
              <button
                onClick={() => setTimeRange('month')}
                className={`px-4 py-2 rounded-lg transition-all flex items-center space-x-2 ${
                  timeRange === 'month'
                    ? 'bg-gradient-to-r from-teal-500/20 to-blue-500/20 border border-teal-400/30 text-white'
                    : 'text-white/60 hover:text-white'
                }`}
              >
                <Calendar className="w-4 h-4" />
                <span>Monthly</span>
              </button>
            </div>
          </div>

          {/* Comparison Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl mb-6"
          >
            <h2 className="text-xl mb-4">Week Comparison</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-slate-800/30 rounded-xl p-4 border border-white/5">
                <div className="text-sm text-white/60 mb-1">Avg. Volatility</div>
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">49.7</span>
                  <TrendingDown className="w-5 h-5 text-teal-400" />
                  <span className="text-sm text-teal-400">-8%</span>
                </div>
              </div>
              <div className="bg-slate-800/30 rounded-xl p-4 border border-white/5">
                <div className="text-sm text-white/60 mb-1">Sleep Quality</div>
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">82%</span>
                  <TrendingUp className="w-5 h-5 text-teal-400" />
                  <span className="text-sm text-teal-400">+5%</span>
                </div>
              </div>
              <div className="bg-slate-800/30 rounded-xl p-4 border border-white/5">
                <div className="text-sm text-white/60 mb-1">Activity Level</div>
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">3.8</span>
                  <TrendingUp className="w-5 h-5 text-teal-400" />
                  <span className="text-sm text-teal-400">+12%</span>
                </div>
              </div>
              <div className="bg-slate-800/30 rounded-xl p-4 border border-white/5">
                <div className="text-sm text-white/60 mb-1">Screen Time</div>
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">5.2h</span>
                  <TrendingDown className="w-5 h-5 text-orange-400" />
                  <span className="text-sm text-orange-400">-3%</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Emotional Volatility Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <h3 className="text-xl mb-4">Emotional Volatility Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={emotionalVolatilityData}>
                  <defs>
                    <linearGradient id="volatilityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3F77FF" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="#9B5CFF" stopOpacity={0.2} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                  <XAxis dataKey="day" stroke="rgba(255, 255, 255, 0.3)" tick={{ fill: '#E8EBF0' }} />
                  <YAxis stroke="rgba(255, 255, 255, 0.3)" tick={{ fill: '#E8EBF0' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="volatility"
                    stroke="#3F77FF"
                    strokeWidth={3}
                    dot={{ fill: '#3F77FF', r: 5 }}
                    activeDot={{ r: 7 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Cognitive Load Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <h3 className="text-xl mb-4">Cognitive Load Shifts</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={cognitiveLoadData}>
                  <defs>
                    <linearGradient id="loadGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#39E6C6" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="#39E6C6" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                  <XAxis dataKey="day" stroke="rgba(255, 255, 255, 0.3)" tick={{ fill: '#E8EBF0' }} />
                  <YAxis stroke="rgba(255, 255, 255, 0.3)" tick={{ fill: '#E8EBF0' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="load"
                    stroke="#39E6C6"
                    strokeWidth={2}
                    fill="url(#loadGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Behavioral Rhythm Map */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <h3 className="text-xl mb-4">Behavioral Rhythm Map</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={behaviorRhythmData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                  <XAxis dataKey="time" stroke="rgba(255, 255, 255, 0.3)" tick={{ fill: '#E8EBF0' }} />
                  <YAxis stroke="rgba(255, 255, 255, 0.3)" tick={{ fill: '#E8EBF0' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="sleep" fill="#39E6C6" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="activity" fill="#3F77FF" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="screen" fill="#9B5CFF" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center space-x-6 mt-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-teal-400"></div>
                  <span className="text-sm text-white/60">Sleep</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span className="text-sm text-white/60">Activity</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                  <span className="text-sm text-white/60">Screen Time</span>
                </div>
              </div>
            </motion.div>

            {/* Multi-Modal Contribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <h3 className="text-xl mb-4">Multi-Modal Contribution</h3>
              <p className="text-sm text-white/60 mb-6">How different data sources influenced predictions</p>
              
              <div className="space-y-4">
                {multiModalContribution.map((item, index) => (
                  <div key={item.source}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Activity className="w-4 h-4" style={{ color: colors[index] }} />
                        <span className="text-sm text-white/80">{item.source}</span>
                      </div>
                      <span className="text-sm" style={{ color: colors[index] }}>{item.contribution}%</span>
                    </div>
                    <div className="relative h-3 rounded-full bg-slate-800/50 overflow-hidden">
                      <motion.div
                        className="absolute inset-y-0 left-0 rounded-full"
                        style={{ backgroundColor: colors[index] }}
                        initial={{ width: 0 }}
                        animate={{ width: `${item.contribution}%` }}
                        transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-teal-500/10 to-blue-500/10 border border-teal-400/20">
                <p className="text-sm text-white/70">
                  <span className="text-teal-400">Insight:</span> Behavioral data showed strongest correlation with wellness patterns this week.
                </p>
              </div>
            </motion.div>
          </div>
        </main>
      </div>
    </div>
  );
}
