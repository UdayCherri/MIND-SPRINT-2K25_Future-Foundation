import { Component, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white flex items-center justify-center p-4">
          <div className="max-w-md w-full rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-8 shadow-2xl">
            <div className="flex items-center justify-center w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 mb-6 mx-auto">
              <AlertTriangle className="w-8 h-8 text-red-400" />
            </div>
            
            <h2 className="text-2xl font-bold text-center mb-4">Something went wrong</h2>
            
            <p className="text-white/70 text-center mb-6">
              We encountered an unexpected error. Please refresh the page and try again.
            </p>

            {this.state.error && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 mb-6">
                <p className="text-sm text-red-400 font-mono">
                  {this.state.error.message}
                </p>
              </div>
            )}

            <button
              onClick={() => window.location.reload()}
              className="w-full px-6 py-3 rounded-xl bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg hover:shadow-xl transition-all"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
