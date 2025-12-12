import { Page } from '../App';
import { Navigation } from '../components/Navigation';
import { HeroSection } from '../components/HeroSection';
import { RadarVisualization } from '../components/RadarVisualization';
import { InputPanel } from '../components/InputPanel';
import { InsightCards } from '../components/InsightCards';
import { TimelineGraph } from '../components/TimelineGraph';
import { PersonalizationPanel } from '../components/PersonalizationPanel';
import { useUser } from '../contexts/UserContext';

interface DashboardPageProps {
  onNavigate: (page: Page) => void;
  onLogout: () => void;
}

export function DashboardPage({ onNavigate, onLogout }: DashboardPageProps) {
  const { currentAnalysis, setCurrentAnalysis } = useUser();

  const handleAnalysisComplete = (result: any) => {
    setCurrentAnalysis(result);
    console.log('📊 Dashboard received analysis:', result);
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
        <Navigation activeTab="Dashboard" onNavigate={onNavigate} onLogout={onLogout} />
        
        <main className="container mx-auto px-4 py-6 max-w-[1600px]">
          <HeroSection />
          
          {/* Analysis Status Banner */}
          {currentAnalysis && (
            <div className="mb-6 px-6 py-4 rounded-2xl bg-gradient-to-r from-teal-500/10 to-blue-500/10 border border-teal-500/30 backdrop-blur-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 rounded-full bg-green-400 animate-pulse"></div>
                  <span className="text-white/90">Analysis Complete</span>
                </div>
                <div className="flex items-center space-x-4 text-sm">
                  <span className="text-white/70">
                    Burnout Risk: <span className={`font-semibold ${
                      currentAnalysis.burnout_score > 0.7 ? 'text-red-400' : 
                      currentAnalysis.burnout_score > 0.4 ? 'text-yellow-400' : 
                      'text-green-400'
                    }`}>{(currentAnalysis.burnout_score * 100).toFixed(1)}%</span>
                  </span>
                  <span className="text-white/70">
                    Patterns Detected: <span className="font-semibold text-teal-400">
                      {currentAnalysis.patterns.filter(p => p.probability > 0.5).length}
                    </span>
                  </span>
                </div>
              </div>
            </div>
          )}
          
          <div className="grid grid-cols-12 gap-6 mt-8">
            {/* Left Side - Input Panel */}
            <div className="col-span-12 lg:col-span-3">
              <InputPanel onAnalysisComplete={handleAnalysisComplete} />
            </div>
            
            {/* Center - Radar Visualization */}
            <div className="col-span-12 lg:col-span-6">
              <RadarVisualization />
            </div>
            
            {/* Right Side - Insight Cards */}
            <div className="col-span-12 lg:col-span-3">
              <InsightCards />
            </div>
          </div>
          
          {/* Bottom Section */}
          <div className="grid grid-cols-12 gap-6 mt-8">
            <div className="col-span-12 lg:col-span-8">
              <TimelineGraph />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <PersonalizationPanel />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
