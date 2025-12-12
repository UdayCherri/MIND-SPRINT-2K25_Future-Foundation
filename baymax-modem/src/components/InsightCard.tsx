/**
 * Insight Card Component
 * Displays actionable insights from pattern analysis with explainability
 */

import { AlertCircle, CheckCircle, AlertTriangle, TrendingUp, Lightbulb } from "lucide-react";
import type { InsightCard as InsightCardType, FeatureContributor } from "../services/analyzeClient";

interface InsightCardProps {
  insight: InsightCardType;
  contributors?: FeatureContributor[];
  className?: string;
}

export function InsightCard({ insight, contributors, className = "" }: InsightCardProps) {
  const severityConfig = {
    low: {
      icon: CheckCircle,
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
      textColor: "text-green-400",
      iconColor: "text-green-500",
    },
    moderate: {
      icon: AlertTriangle,
      bgColor: "bg-yellow-500/10",
      borderColor: "border-yellow-500/30",
      textColor: "text-yellow-400",
      iconColor: "text-yellow-500",
    },
    high: {
      icon: AlertCircle,
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/30",
      textColor: "text-red-400",
      iconColor: "text-red-500",
    },
  };

  const config = severityConfig[insight.severity];
  const Icon = config.icon;

  return (
    <div
      className={`rounded-lg border-2 ${config.borderColor} ${config.bgColor} p-6 space-y-4 ${className}`}
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${config.bgColor}`}>
          <Icon className={`w-6 h-6 ${config.iconColor}`} />
        </div>
        <div className="flex-1">
          <h3 className={`text-lg font-semibold ${config.textColor}`}>
            {insight.title}
          </h3>
          {insight.subtitle && (
            <p className="text-sm text-gray-400 mt-1">{insight.subtitle}</p>
          )}
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-medium uppercase ${config.textColor} ${config.bgColor}`}
        >
          {insight.severity}
        </span>
      </div>

      {/* Description */}
      <div className="space-y-2">
        <p className="text-gray-300 text-sm leading-relaxed">{insight.description}</p>
      </div>

      {/* Top Contributors (if provided) */}
      {contributors && contributors.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-gray-400">
            <TrendingUp className="w-4 h-4" />
            <span>Top Contributing Factors</span>
          </div>
          <div className="space-y-1.5">
            {contributors.slice(0, 5).map((contributor, index) => (
              <div
                key={index}
                className="flex items-center gap-2 text-sm"
              >
                <div className="flex-1 flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span className="text-gray-300">
                    {contributor.friendly_name || contributor.feature}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-500 rounded-full"
                      style={{ width: `${Math.abs(contributor.score) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400 w-12 text-right">
                    {(contributor.score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendation */}
      <div className="space-y-2 pt-2 border-t border-gray-700">
        <div className="flex items-center gap-2 text-sm font-medium text-cyan-400">
          <Lightbulb className="w-4 h-4" />
          <span>Recommendation</span>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed pl-6">
          {insight.recommendation}
        </p>
      </div>
    </div>
  );
}

/**
 * Insight Cards Grid - Display multiple insight cards
 */
interface InsightCardsGridProps {
  insights: InsightCardType[];
  contributorsMap?: Map<string, FeatureContributor[]>;
  className?: string;
}

export function InsightCardsGrid({
  insights,
  contributorsMap,
  className = "",
}: InsightCardsGridProps) {
  if (insights.length === 0) {
    return (
      <div className={`text-center py-12 ${className}`}>
        <p className="text-gray-400">No insights available yet. Complete an analysis to get started.</p>
      </div>
    );
  }

  return (
    <div className={`grid gap-4 md:grid-cols-2 lg:grid-cols-1 ${className}`}>
      {insights.map((insight, index) => (
        <InsightCard
          key={index}
          insight={insight}
          contributors={contributorsMap?.get(insight.primary_pattern || "")}
        />
      ))}
    </div>
  );
}
