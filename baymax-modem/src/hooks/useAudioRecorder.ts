/**
 * Audio Recording Hook for BAYMAX
 * Handles microphone recording and audio upload
 */

import { useState, useRef, useCallback } from "react";
import { uploadAudio, type AudioUploadResponse } from "../services/analyzeClient";

export interface AudioRecordingState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  audioUrl: string | null;
  error: string | null;
}

export interface UseAudioRecorderReturn {
  state: AudioRecordingState;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  pauseRecording: () => void;
  resumeRecording: () => void;
  clearRecording: () => void;
  uploadRecording: () => Promise<AudioUploadResponse>;
}

export function useAudioRecorder(): UseAudioRecorderReturn {
  const [state, setState] = useState<AudioRecordingState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    audioUrl: null,
    error: null,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const audioBlobRef = useRef<Blob | null>(null);

  /**
   * Start recording from microphone
   */
  const startRecording = useCallback(async () => {
    try {
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];

      // Create MediaRecorder
      const mimeType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/mp4";
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;

      // Handle data available
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      // Handle stop event
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        audioBlobRef.current = blob;
        const url = URL.createObjectURL(blob);
        
        setState((prev) => ({
          ...prev,
          isRecording: false,
          isPaused: false,
          audioUrl: url,
        }));

        // Stop timer
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }

        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
      };

      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms

      // Start timer
      timerRef.current = window.setInterval(() => {
        setState((prev) => ({
          ...prev,
          duration: prev.duration + 1,
        }));
      }, 1000);

      setState((prev) => ({
        ...prev,
        isRecording: true,
        isPaused: false,
        duration: 0,
        error: null,
      }));
    } catch (error) {
      console.error("Error starting recording:", error);
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : "Failed to start recording",
      }));
    }
  }, []);

  /**
   * Stop recording
   */
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  /**
   * Pause recording
   */
  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.pause();
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      setState((prev) => ({
        ...prev,
        isPaused: true,
      }));
    }
  }, []);

  /**
   * Resume recording
   */
  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "paused") {
      mediaRecorderRef.current.resume();

      // Restart timer
      timerRef.current = window.setInterval(() => {
        setState((prev) => ({
          ...prev,
          duration: prev.duration + 1,
        }));
      }, 1000);

      setState((prev) => ({
        ...prev,
        isPaused: false,
      }));
    }
  }, []);

  /**
   * Clear recording and reset state
   */
  const clearRecording = useCallback(() => {
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
    }

    audioBlobRef.current = null;
    chunksRef.current = [];

    setState({
      isRecording: false,
      isPaused: false,
      duration: 0,
      audioUrl: null,
      error: null,
    });
  }, [state.audioUrl]);

  /**
   * Upload recorded audio to backend and get embedding
   */
  const uploadRecording = useCallback(async (): Promise<AudioUploadResponse> => {
    if (!audioBlobRef.current) {
      throw new Error("No recording available to upload");
    }

    try {
      const result = await uploadAudio(audioBlobRef.current);
      return result;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : "Upload failed",
      }));
      throw error;
    }
  }, []);

  return {
    state,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    clearRecording,
    uploadRecording,
  };
}

/**
 * Format duration seconds to MM:SS
 */
export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}
