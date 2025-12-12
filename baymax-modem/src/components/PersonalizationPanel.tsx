import { motion } from 'motion/react';
import { Brain, Check, Zap, Loader2, CheckCircle } from 'lucide-react';
import { useState, useEffect } from 'react';
import {
  checkAdapterStatus,
  updateAdapter,
  loadAdapter,
  type AdapterRequest,
} from '../services/analyzeClient';

interface PersonalizationPanelProps {
  userId?: string;
  onAdapterTrained?: () => void;
}

export function PersonalizationPanel({ userId = 'demo_user', onAdapterTrained }: PersonalizationPanelProps) {
  const [adapterStatus, setAdapterStatus] = useState<{
    hasAdapter: boolean;
    adapterSizeKb?: number;
  } | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const updates = [
    { label: 'Sleep Pattern Recognition', progress: 100, status: 'complete' },
    { label: 'Stress Trigger Mapping', progress: 85, status: 'learning' },
    { label: 'Emotional Baseline', progress: 92, status: 'learning' },
    { label: 'Cognitive Load Prediction', progress: 78, status: 'learning' },
  ];

  // Check adapter status on mount
  useEffect(() => {
    loadAdapterStatus();
  }, [userId]);

  const loadAdapterStatus = async () => {
    try {
      const status = await checkAdapterStatus(userId);
      setAdapterStatus(status);
    } catch (err) {
      console.error('Failed to check adapter status:', err);
    }
  };

  const handleTrainAdapter = async () => {
    setIsTraining(true);
    setError(null);
    setSuccessMessage(null);

    try {
      // Create mock training data (in production, collect from user's history)
      const mockTrainingData: AdapterRequest = {
        user_id: userId,
        audio_samples: [
          Array.from({ length: 256 }, () => Math.random() * 0.5),
          Array.from({ length: 256 }, () => Math.random() * 0.5 + 0.2),
          Array.from({ length: 256 }, () => Math.random() * 0.5 - 0.1),
        ],
        text_samples: [
          ['Feeling stressed today', 'Work was overwhelming'],
          ['Slept well', 'Feeling energetic'],
          ['Tired and anxious', "Can't focus"],
        ],
        behavior_samples: [
          [5.5, 1.2, 8.5, 35, 0.3, 25],
          [7.5, 0.5, 6.0, 40, 0.5, 45],
          [4.0, 2.0, 10.0, 55, 0.2, 15],
        ],
        target_patterns: [
          [0.6, 0.5, 0.7, 0.4],
          [0.2, 0.1, 0.2, 0.1],
          [0.8, 0.7, 0.8, 0.6],
        ],
        target_burnout: [0.6, 0.2, 0.8],
        epochs: 10,
        batch_size: 2,
      };

      const result = await updateAdapter(mockTrainingData);
      
      setSuccessMessage(
        `✓ Personalized model trained! Adapter size: ${result.adapter_size_kb}KB`
      );
      
      // Reload status
      await loadAdapterStatus();
      
      // Notify parent
      onAdapterTrained?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
      className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-8 shadow-2xl"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 shadow-lg shadow-purple-500/30">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <motion.div
              className="absolute inset-0 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 blur-md opacity-50"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.5, 0.8, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            ></motion.div>
          </div>
          <div>
            <h2 className="text-xl">Your Model Updates</h2>
          </div>
        </div>

        {/* Adapter Status Badge */}
        {adapterStatus?.hasAdapter && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30">
            <CheckCircle className="w-4 h-4 text-green-400" />
            <span className="text-xs text-green-400 font-medium">
              Personalized ({adapterStatus.adapterSizeKb?.toFixed(0)}KB)
            </span>
          </div>
        )}
      </div>

      <p className="text-white/70 text-sm mb-4">
        Adapter learning applied using this week's data.
      </p>

      {/* Messages */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm"
        >
          {error}
        </motion.div>
      )}
      {successMessage && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg text-green-400 text-sm"
        >
          {successMessage}
        </motion.div>
      )}

      {/* Personalize Button */}
      <button
        onClick={handleTrainAdapter}
        disabled={isTraining}
        className="w-full mb-6 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-all shadow-lg hover:shadow-xl disabled:shadow-none"
      >
        {isTraining ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Training Personalized Model...
          </>
        ) : (
          <>
            <Zap className="w-5 h-5" />
            {adapterStatus?.hasAdapter ? 'Retrain Personalized Model' : 'Train Personalized Model'}
          </>
        )}
      </button>

      {/* Model Updates Section */}
      <div className="mb-6">
        <h2 className="text-xl text-white mb-2">Your Model Updates</h2>
        <p className="text-white/70 text-sm">
          Adapter learning applied using this week's data.
        </p>
      </div>

      <div className="space-y-4">
        {updates.map((update, index) => (
          <motion.div
            key={update.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.6 + index * 0.1 }}
            className="space-y-2"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {update.status === 'complete' ? (
                  <div className="w-5 h-5 rounded-full bg-teal-500/20 border border-teal-400 flex items-center justify-center">
                    <Check className="w-3 h-3 text-teal-400" />
                  </div>
                ) : (
                  <div className="w-5 h-5 rounded-full border-2 border-purple-400/30 border-t-purple-400 animate-spin"></div>
                )}
                <span className="text-sm text-white/80">{update.label}</span>
              </div>
              <span className="text-sm text-white/60">{update.progress}%</span>
            </div>
            
            {/* Progress Bar */}
            <div className="relative h-2 rounded-full bg-slate-800/50 overflow-hidden">
              <motion.div
                className={`absolute inset-y-0 left-0 rounded-full ${
                  update.status === 'complete'
                    ? 'bg-gradient-to-r from-teal-400 to-blue-400'
                    : 'bg-gradient-to-r from-purple-400 to-pink-400'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${update.progress}%` }}
                transition={{ duration: 1, delay: 0.8 + index * 0.1 }}
              />
              {update.status === 'learning' && (
                <motion.div
                  className="absolute inset-y-0 w-8 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                  animate={{
                    left: ['-32px', '100%'],
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                />
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Status Badge */}
      <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-400/20">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></div>
          <span className="text-sm text-white/80">Model adapting to your patterns</span>
        </div>
      </div>
    </motion.div>
  );
}
