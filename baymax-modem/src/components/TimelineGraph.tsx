import { motion } from 'motion/react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useUser } from '../contexts/UserContext';

export function TimelineGraph() {
  const { timelineData } = useUser();

  // Use real data if available, otherwise show demo data
  const chartData = timelineData?.dates?.length
    ? timelineData.dates.map((date, i) => ({
        day: new Date(date).toLocaleDateString('en-US', { weekday: 'short' }),
        date: date,
        burnout: Math.round(timelineData.burnout_scores[i] * 100),
      }))
    : [
        { day: 'Mon', burnout: 72, date: '2025-12-09' },
        { day: 'Tue', burnout: 75, date: '2025-12-10' },
        { day: 'Wed', burnout: 68, date: '2025-12-11' },
        { day: 'Thu', burnout: 65, date: '2025-12-12' },
        { day: 'Fri', burnout: 78, date: '2025-12-13' },
        { day: 'Sat', burnout: 82, date: '2025-12-14' },
        { day: 'Sun', burnout: 79, date: '2025-12-15' },
      ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800/90 backdrop-blur-xl border border-white/20 rounded-xl p-3 shadow-xl">
          <p className="text-white/70 text-sm">{payload[0].payload.date}</p>
          <p className="text-teal-400">
            Burnout: <span className="text-white">{payload[0].value}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
      className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-8 shadow-2xl"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl mb-1">Burnout Timeline</h2>
          <p className="text-sm text-white/60">
            {timelineData?.dates?.length 
              ? `Last ${timelineData.dates.length} days` 
              : 'Sample data - run analysis to see your timeline'}
          </p>
        </div>
        <div className={`w-2 h-2 rounded-full ${timelineData ? 'bg-green-400' : 'bg-gray-400'} animate-pulse`}></div>
      </div>

      <div className="relative">
        {/* Background Glow */}
        <div className="absolute inset-0 bg-gradient-to-r from-teal-500/5 via-blue-500/5 to-purple-500/5 rounded-2xl blur-2xl"></div>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#39E6C6" />
                <stop offset="50%" stopColor="#3F77FF" />
                <stop offset="100%" stopColor="#9B5CFF" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
            <XAxis 
              dataKey="day" 
              stroke="rgba(255, 255, 255, 0.3)"
              tick={{ fill: '#E8EBF0' }}
            />
            <YAxis 
              domain={[0, 100]}
              stroke="rgba(255, 255, 255, 0.3)"
              tick={{ fill: '#E8EBF0' }}
              label={{ value: 'Burnout %', angle: -90, position: 'insideLeft', fill: '#E8EBF0' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="burnout"
              stroke="url(#lineGradient)"
              strokeWidth={3}
              dot={{ fill: '#39E6C6', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: '#39E6C6', stroke: '#fff', strokeWidth: 2 }}
              className="drop-shadow-[0_0_8px_rgba(57,230,198,0.6)]"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
