import { motion } from 'motion/react';
import { useEffect, useState } from 'react';

export function HeroSection() {
  const [points, setPoints] = useState<Array<{ x: number; y: number; vx: number; vy: number }>>([]);

  useEffect(() => {
    // Generate neural network points
    const newPoints = Array.from({ length: 30 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 40,
      vx: (Math.random() - 0.5) * 0.1,
      vy: (Math.random() - 0.5) * 0.1,
    }));
    setPoints(newPoints);

    const interval = setInterval(() => {
      setPoints((prevPoints) =>
        prevPoints.map((point) => ({
          x: (point.x + point.vx + 100) % 100,
          y: (point.y + point.vy + 40) % 40,
          vx: point.vx,
          vy: point.vy,
        }))
      );
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-8 shadow-2xl">
      {/* Neural Network Background */}
      <div className="absolute inset-0 overflow-hidden opacity-20">
        <svg className="w-full h-full" style={{ minHeight: '200px' }}>
          {/* Connections */}
          {points.map((point, i) => {
            return points.slice(i + 1).map((otherPoint, j) => {
              const distance = Math.sqrt(
                Math.pow(point.x - otherPoint.x, 2) + Math.pow(point.y - otherPoint.y, 2)
              );
              if (distance < 20) {
                return (
                  <line
                    key={`${i}-${j}`}
                    x1={`${point.x}%`}
                    y1={`${point.y}%`}
                    x2={`${otherPoint.x}%`}
                    y2={`${otherPoint.y}%`}
                    stroke="url(#gradient)"
                    strokeWidth="1"
                    opacity={1 - distance / 20}
                  />
                );
              }
              return null;
            });
          })}
          {/* Points */}
          {points.map((point, i) => (
            <circle
              key={i}
              cx={`${point.x}%`}
              cy={`${point.y}%`}
              r="2"
              fill="#39E6C6"
              className="drop-shadow-[0_0_4px_rgba(57,230,198,0.8)]"
            />
          ))}
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#39E6C6" />
              <stop offset="50%" stopColor="#3F77FF" />
              <stop offset="100%" stopColor="#9B5CFF" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Content */}
      <div className="relative z-10 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-4xl mb-3 tracking-tight">
            Your Mental Wellness Overview
          </h1>
        </motion.div>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-white/70 text-lg"
        >
          Real-time cognitive and emotional pattern analysis
        </motion.p>
      </div>

      {/* Waveform Animation */}
      <div className="absolute bottom-0 left-0 right-0 h-16 overflow-hidden opacity-30">
        <svg className="w-full h-full" viewBox="0 0 1200 100" preserveAspectRatio="none">
          <motion.path
            d="M0,50 Q150,20 300,50 T600,50 T900,50 T1200,50"
            fill="none"
            stroke="url(#waveGradient)"
            strokeWidth="2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear"
            }}
          />
          <defs>
            <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#39E6C6" />
              <stop offset="50%" stopColor="#3F77FF" />
              <stop offset="100%" stopColor="#9B5CFF" />
            </linearGradient>
          </defs>
        </svg>
      </div>
    </div>
  );
}
