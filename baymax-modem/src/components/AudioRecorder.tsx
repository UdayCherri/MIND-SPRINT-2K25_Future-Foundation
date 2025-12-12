/**
 * Audio Recording Component
 * UI for recording voice with microphone
 */

import { Mic, Square, Play, Pause, Trash2, Upload } from "lucide-react";
import { useAudioRecorder, formatDuration } from "../hooks/useAudioRecorder";
import { useState } from "react";

interface AudioRecorderProps {
  onAudioReady: (embedding: number[]) => void;
  className?: string;
}

export function AudioRecorder({ onAudioReady, className = "" }: AudioRecorderProps) {
  const {
    state,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    clearRecording,
    uploadRecording,
  } = useAudioRecorder();

  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleUpload = async () => {
    setIsUploading(true);
    setUploadError(null);

    try {
      const result = await uploadRecording();
      onAudioReady(result.audio_embedding);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Recording Status */}
      {state.isRecording && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          <span className="text-red-400 font-medium">Recording</span>
          <span className="ml-auto text-gray-300 font-mono">
            {formatDuration(state.duration)}
          </span>
        </div>
      )}

      {/* Audio Player (when recorded) */}
      {state.audioUrl && !state.isRecording && (
        <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg space-y-3">
          <audio src={state.audioUrl} controls className="w-full" />
          <div className="text-sm text-gray-400">
            Duration: {formatDuration(state.duration)}
          </div>
        </div>
      )}

      {/* Error Display */}
      {(state.error || uploadError) && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {state.error || uploadError}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap gap-2">
        {!state.isRecording && !state.audioUrl && (
          <button
            onClick={startRecording}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors"
          >
            <Mic className="w-4 h-4" />
            Start Recording
          </button>
        )}

        {state.isRecording && !state.isPaused && (
          <>
            <button
              onClick={pauseRecording}
              className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-500 text-white rounded-lg transition-colors"
            >
              <Pause className="w-4 h-4" />
              Pause
            </button>
            <button
              onClick={stopRecording}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          </>
        )}

        {state.isRecording && state.isPaused && (
          <>
            <button
              onClick={resumeRecording}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors"
            >
              <Play className="w-4 h-4" />
              Resume
            </button>
            <button
              onClick={stopRecording}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          </>
        )}

        {state.audioUrl && !state.isRecording && (
          <>
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              <Upload className="w-4 h-4" />
              {isUploading ? "Uploading..." : "Use Recording"}
            </button>
            <button
              onClick={clearRecording}
              disabled={isUploading}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          </>
        )}
      </div>

      {/* Instructions */}
      {!state.isRecording && !state.audioUrl && (
        <div className="text-sm text-gray-400">
          <p>Click "Start Recording" to record your voice. Speak naturally for 5-30 seconds.</p>
        </div>
      )}
    </div>
  );
}
