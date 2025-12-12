import { motion } from 'motion/react';
import { Mic, Send, Loader2, AlertCircle } from 'lucide-react';
import { useState } from 'react';
import { analyzeClient } from '../services/analyzeClient';
import { useUser } from '../contexts/UserContext';
import type { AnalyzeResponse } from '../services/analyzeClient';

interface InputPanelProps {
  onAnalysisComplete?: (result: AnalyzeResponse) => void;
}

export function InputPanel({ onAnalysisComplete }: InputPanelProps = {}) {
  const [sleepHours, setSleepHours] = useState(7);
  const [screenTime, setScreenTime] = useState(4);
  const [activityLevel, setActivityLevel] = useState('Moderate');
  const [journalText, setJournalText] = useState('');
  const [audioEmbedding, setAudioEmbedding] = useState<number[] | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { user } = useUser();

  const handleAnalyze = async () => {
    if (!journalText.trim()) {
      setError('Please enter a journal entry');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const behaviorVector = analyzeClient.createBehaviorVector({
        sleepHours,
        screenTime,
        activityLevel,
        socialInteractions: 3,
        physicalActivity: activityLevel === 'High' ? 2 : activityLevel === 'Moderate' ? 1.5 : 0.5,
        moodRating: 5
      });

      const response = await analyzeClient.analyze({
        user_id: user?.user_id || 'demo_user_123',
        date: analyzeClient.getCurrentDate(),
        texts: [journalText],
        audio_embedding: audioEmbedding || analyzeClient.createMockAudioEmbedding(),
        behavior: behaviorVector,
        include_explain: true
      });

      console.log('✅ Analysis complete:', response);
      
      if (onAnalysisComplete) {
        onAnalysisComplete(response);
      }

      // Show success feedback
      setError(null);
      
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(err.message || 'Analysis failed. Please check if backend is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRecordAudio = () => {
    setIsRecording(!isRecording);
    // For demo, just create mock audio embedding
    if (!isRecording) {
      setTimeout(() => {
        const mockEmbedding = analyzeClient.createMockAudioEmbedding();
        setAudioEmbedding(mockEmbedding);
        setIsRecording(false);
        console.log('🎤 Audio recorded (mock)');
      }, 2000);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
    >
      <h2 className="text-xl mb-6 flex items-center space-x-2">
        <div className="w-1.5 h-6 bg-gradient-to-b from-teal-400 to-blue-500 rounded-full shadow-lg shadow-teal-400/50"></div>
        <span>Daily Check-in</span>
      </h2>

      <div className="space-y-5">
        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 bg-red-500/10 border border-red-500/30 rounded-xl flex items-start gap-2 text-sm text-red-400"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </motion.div>
        )}

        {/* Success Message */}
        {audioEmbedding && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 bg-green-500/10 border border-green-500/30 rounded-xl flex items-center gap-2 text-sm text-green-400"
          >
            <Mic className="w-4 h-4" />
            <span>Audio recorded successfully</span>
          </motion.div>
        )}

        {/* Journal Text Box */}
        <div>
          <label className="block text-sm text-white/70 mb-2">Journal Entry</label>
          <div className="relative">
            <textarea
              value={journalText}
              onChange={(e) => setJournalText(e.target.value)}
              placeholder="How are you feeling today?"
              className="w-full h-32 px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all resize-none"
            />
          </div>
        </div>

        {/* Sleep Hours Slider */}
        <div>
          <label className="block text-sm text-white/70 mb-2">
            Sleep Hours: <span className="text-teal-400">{sleepHours}h</span>
          </label>
          <input
            type="range"
            min="0"
            max="12"
            step="0.5"
            value={sleepHours}
            onChange={(e) => setSleepHours(parseFloat(e.target.value))}
            className="w-full h-2 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #39E6C6 0%, #3F77FF ${(sleepHours / 12) * 100}%, rgba(255,255,255,0.1) ${(sleepHours / 12) * 100}%, rgba(255,255,255,0.1) 100%)`,
            }}
          />
        </div>

        {/* Screen Time Input */}
        <div>
          <label className="block text-sm text-white/70 mb-2">Screen Time (hours)</label>
          <input
            type="number"
            value={screenTime}
            onChange={(e) => setScreenTime(parseFloat(e.target.value))}
            min="0"
            max="24"
            step="0.5"
            className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
          />
        </div>

        {/* Activity Level Dropdown */}
        <div>
          <label className="block text-sm text-white/70 mb-2">Activity Level</label>
          <select
            value={activityLevel}
            onChange={(e) => setActivityLevel(e.target.value)}
            className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all cursor-pointer"
          >
            <option value="Low">Low</option>
            <option value="Moderate">Moderate</option>
            <option value="High">High</option>
            <option value="Very High">Very High</option>
          </select>
        </div>

        {/* Upload Audio Button */}
        <div>
          <label className="block text-sm text-white/70 mb-2">Voice Recording</label>
          <button 
            onClick={handleRecordAudio}
            disabled={isRecording}
            className={`w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white/70 hover:text-white hover:border-purple-400/50 transition-all flex items-center justify-center space-x-2 group ${
              isRecording ? 'border-purple-400/50 text-purple-400' : ''
            } ${audioEmbedding ? 'border-green-400/50 text-green-400' : ''}`}
          >
            {isRecording ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Mic className={`w-4 h-4 transition-colors ${
                audioEmbedding ? 'text-green-400' : 'group-hover:text-purple-400'
              }`} />
            )}
            <span>{isRecording ? 'Recording...' : audioEmbedding ? 'Audio Recorded' : 'Record Audio'}</span>
          </button>
        </div>

        {/* CTA Button */}
        <motion.button
          onClick={handleAnalyze}
          disabled={isAnalyzing || !journalText.trim()}
          whileHover={{ scale: isAnalyzing ? 1 : 1.02 }}
          whileTap={{ scale: isAnalyzing ? 1 : 0.98 }}
          className="w-full px-6 py-4 rounded-xl bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg shadow-teal-500/30 hover:shadow-teal-500/50 transition-all flex items-center justify-center space-x-2 relative overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-teal-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity"></div>
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 relative z-10 animate-spin" />
              <span className="relative z-10">Analyzing...</span>
            </>
          ) : (
            <>
              <Send className="w-5 h-5 relative z-10" />
              <span className="relative z-10">Analyze with BAYMAX</span>
            </>
          )}
        </motion.button>
      </div>
    </motion.div>
  );
}
