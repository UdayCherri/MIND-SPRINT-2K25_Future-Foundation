import { Activity, BarChart3, Lightbulb, Settings, LogOut } from 'lucide-react';
import { Page } from '../App';

interface NavigationProps {
  activeTab: string;
  onNavigate: (page: Page) => void;
  onLogout: () => void;
}

export function Navigation({ activeTab, onNavigate, onLogout }: NavigationProps) {
  const tabs = [
    { name: 'Dashboard', icon: Activity, page: 'dashboard' as Page },
    { name: 'Analytics', icon: BarChart3, page: 'analytics' as Page },
    { name: 'Insights', icon: Lightbulb, page: 'insights' as Page },
    { name: 'Settings', icon: Settings, page: 'settings' as Page },
  ];

  return (
    <nav className="border-b border-white/10 backdrop-blur-xl bg-slate-900/50">
      <div className="container mx-auto px-4 max-w-[1600px]">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-400 to-blue-500 flex items-center justify-center shadow-lg shadow-teal-500/50">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-teal-400 to-blue-500 blur-md opacity-50 animate-pulse"></div>
            </div>
            <div>
              <div className="text-xl tracking-tight">
                <span className="bg-gradient-to-r from-teal-400 to-blue-400 bg-clip-text text-transparent">BAYMAX</span>
                <span className="text-white/90">: ModeM</span>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.name;
              
              return (
                <button
                  key={tab.name}
                  onClick={() => onNavigate(tab.page)}
                  className={`relative px-4 py-2 rounded-lg transition-all duration-300 flex items-center space-x-2 ${
                    isActive
                      ? 'text-white'
                      : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  {isActive && (
                    <div className="absolute inset-0 bg-gradient-to-r from-teal-500/20 to-blue-500/20 rounded-lg border border-teal-400/30 shadow-lg shadow-teal-500/20"></div>
                  )}
                  <Icon className={`w-4 h-4 relative z-10 ${isActive ? 'drop-shadow-[0_0_8px_rgba(57,230,198,0.6)]' : ''}`} />
                  <span className="relative z-10">{tab.name}</span>
                </button>
              );
            })}
            
            {/* Logout Button */}
            <button
              onClick={onLogout}
              className="relative px-4 py-2 rounded-lg transition-all duration-300 flex items-center space-x-2 text-white/60 hover:text-white/90 ml-4"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}