/**
 * Pattern Radar Chart Component
 * Visualizes mental health pattern probabilities in a radar chart
 */

import { useEffect, useRef } from "react";
import type { PatternConfidence } from "../services/analyzeClient";

interface PatternRadarProps {
  patterns: PatternConfidence[];
  className?: string;
}

export function PatternRadar({ patterns, className = "" }: PatternRadarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const size = 400;
    canvas.width = size;
    canvas.height = size;

    const centerX = size / 2;
    const centerY = size / 2;
    const maxRadius = size / 2 - 60;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    // Pattern labels (formatted)
    const labels = patterns.map((p) =>
      p.pattern
        .split("_")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ")
    );

    // Draw background circles
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach((ratio) => {
      ctx.beginPath();
      ctx.arc(centerX, centerY, maxRadius * ratio, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Draw axes
    const angleStep = (Math.PI * 2) / patterns.length;
    ctx.strokeStyle = "#4b5563";
    ctx.lineWidth = 1;

    patterns.forEach((_, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const x = centerX + Math.cos(angle) * maxRadius;
      const y = centerY + Math.sin(angle) * maxRadius;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();
    });

    // Draw data polygon
    ctx.fillStyle = "rgba(57, 230, 198, 0.15)";
    ctx.strokeStyle = "rgba(57, 230, 198, 0.9)";
    ctx.lineWidth = 2;

    ctx.beginPath();
    patterns.forEach((pattern, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const radius = maxRadius * pattern.probability;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Draw data points
    ctx.fillStyle = "rgba(63, 119, 255, 1)";
    patterns.forEach((pattern, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const radius = maxRadius * pattern.probability;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw labels
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    labels.forEach((label, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const labelRadius = maxRadius + 30;
      const x = centerX + Math.cos(angle) * labelRadius;
      const y = centerY + Math.sin(angle) * labelRadius;

      // Draw label with background
      const lines = label.split(" ");
      lines.forEach((line, lineIndex) => {
        const yOffset = (lineIndex - (lines.length - 1) / 2) * 14;
        ctx.fillText(line, x, y + yOffset);
      });

      // Draw probability value
      ctx.font = "bold 11px sans-serif";
      ctx.fillStyle = getSeverityColor(pattern.probability);
      const probY = y + (lines.length / 2) * 14 + 12;
      ctx.fillText(`${(patterns[i].probability * 100).toFixed(0)}%`, x, probY);
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#e5e7eb";
    });
  }, [patterns]);

  return (
    <div className={`flex flex-col items-center ${className}`}>
      <canvas ref={canvasRef} className="max-w-full h-auto" />
    </div>
  );
}

/**
 * Get color based on severity level
 */
function getSeverityColor(probability: number): string {
  if (probability < 0.3) return "#10b981"; // green
  if (probability < 0.6) return "#f59e0b"; // yellow
  return "#ef4444"; // red
}
